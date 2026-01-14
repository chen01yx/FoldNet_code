import logging
logger = logging.getLogger(__name__)

import json
import pathlib
from dataclasses import dataclass, asdict
import os
from typing import Optional
import subprocess

import omegaconf
import hydra
import numpy as np
import trimesh

from garmentds.foldenv.fold_env import FoldEnv, FoldEnvCfg, RenderProcess, env_timer
from garmentds.foldenv.policy.state.base import FoldStatePolicyCfg, FoldStatePolicy
from garmentds.foldenv.policy.state.tshirt import FoldStateTShirtPolicy, FoldStateTShirtPolicyCfg
from garmentds.foldenv.policy.state.tshirt_lr import FoldStateTShirtLRPolicyCfg, FoldStateTShirtLPolicy, FoldStateTShirtRPolicy
from garmentds.foldenv.policy.state.trousers import FoldStateTrousersPolicy, FoldStateTrousersPolicyCfg
from garmentds.foldenv.policy.visual import FoldVisualPolicy, FoldVisualPolicyCfg
from garmentds.foldenv.policy.hybrid import FoldHybridTShirtPolicy, FoldHybridTrousersPolicy, FoldHybridPolicyCfg, hybrid_action_post_process
from garmentds.foldenv.run_utils import *
import garmentds.common.utils as utils


@hydra.main(config_path="../config/run", config_name=pathlib.Path(__file__).stem, version_base='1.3')
def main(cfg: omegaconf.DictConfig):
    logger.info(f"main pid: {os.getpid()}")
    # setup
    utils.init_omegaconf()
    omegaconf.OmegaConf.resolve(cfg)
    cfg = utils.resolve_overwrite(cfg)
    omegaconf.OmegaConf.save(cfg, os.path.join(os.getcwd(), ".hydra", "resolved.yaml"))
    
    utils.seed_all(cfg.misc.seed)
    RenderProcess.tmp_dir = os.path.abspath(".tmp")
    env_timer.disable = not bool(cfg.misc.use_timer)

    # init run cfg
    @dataclass
    class RunCfg:
        traj_num: int
        generate_fold_gt_and_compute_error: bool = True
        generate_fold_gt_overwrite_render: Optional[bool] = False
        random_init_cloth: bool = True
        pre_and_post_overwrite_render: Optional[bool] = False
        generate_video: bool = True
        sync_per_traj: bool = True
        category: str = "tshirt"
        style: int = 0
    run_cfg = RunCfg(**cfg.run)
    
    # init
    env_cfg = FoldEnvCfg(**cfg.env)
    env = FoldEnv(env_cfg)
    env.reset()
    omegaconf.OmegaConf.save(
        omegaconf.DictConfig(json.loads(json.dumps(asdict(env_cfg), default=utils.custom_serializer))), 
        os.path.join(os.getcwd(), ".hydra", "env_cfg.yaml")
    )
    del env_cfg # this is a tmp object, delete it to prevent bugs

    # init policy to generate ground truth
    state_policy_cfg_cls = {
        ("tshirt", 0): FoldStateTShirtPolicyCfg, 
        ("tshirt", 1): FoldStateTShirtLRPolicyCfg, 
        ("tshirt", 2): FoldStateTShirtLRPolicyCfg, 
        ("trousers", 0): FoldStateTrousersPolicyCfg, 
    }[(run_cfg.category, run_cfg.style)]
    state_policy_cls = {
        ("tshirt", 0): FoldStateTShirtPolicy, 
        ("tshirt", 1): FoldStateTShirtLPolicy, 
        ("tshirt", 2): FoldStateTShirtRPolicy, 
        ("trousers", 0): FoldStateTrousersPolicy, 
    }[(run_cfg.category, run_cfg.style)]
    policy_gt_cfg: FoldStatePolicyCfg = state_policy_cfg_cls(cloth_scale=env.cloth_scale, **(cfg.policy.gt))
    policy_gt: FoldStatePolicy = state_policy_cls(policy_gt_cfg, env)
    omegaconf.OmegaConf.save(
        omegaconf.DictConfig(json.loads(json.dumps(asdict(policy_gt_cfg), default=utils.custom_serializer))), 
        os.path.join(os.getcwd(), ".hydra", "policy_gt_cfg.yaml")
    )
    del policy_gt_cfg # this is a tmp object, delete it to prevent bugs

    # init policy to rollout
    if cfg.policy.run.name == "state":
        policy_run_cfg = state_policy_cfg_cls(cloth_scale=env.cloth_scale, **(cfg.policy.run.state))
        policy_run = state_policy_cls(policy_run_cfg, env)
    elif cfg.policy.run.name == "visual":
        visual_cfg_dict_override: dict = omegaconf.OmegaConf.to_container(cfg.policy.run.visual)
        visual_cfg_dict_base: dict = omegaconf.OmegaConf.to_container(
            omegaconf.OmegaConf.load(utils.get_path_handler()(
                f"config/visual_policy/{visual_cfg_dict_override['name']}.yaml"
            ))
        )
        visual_cfg_dict_base.update(visual_cfg_dict_override)
        policy_run_cfg = FoldVisualPolicyCfg(**visual_cfg_dict_base)
        policy_run = FoldVisualPolicy(policy_run_cfg, env)
    elif cfg.policy.run.name == "hybrid":
        hybrid_cfg_dict_override: dict = omegaconf.OmegaConf.to_container(cfg.policy.run.hybrid)
        hybrid_cfg_dict_base: dict = omegaconf.OmegaConf.to_container(
            omegaconf.OmegaConf.load(utils.get_path_handler()(
                f"config/visual_policy/{hybrid_cfg_dict_override['name']}.yaml"
            ))
        )
        hybrid_cfg_dict_base.update(hybrid_cfg_dict_override)
        policy_run_cfg = FoldHybridPolicyCfg(**hybrid_cfg_dict_base)
        policy_run = {
            "tshirt": FoldHybridTShirtPolicy, "trousers": FoldHybridTrousersPolicy,
        }[run_cfg.category](policy_run_cfg, env)
    else:
        raise ValueError(f"Invalid policy name: {cfg.policy.run.name}")
    omegaconf.OmegaConf.save(
        omegaconf.DictConfig(json.loads(json.dumps(asdict(policy_run_cfg), default=utils.custom_serializer))), 
        os.path.join(os.getcwd(), ".hydra", "policy_run_cfg.yaml")
    )
    del policy_run_cfg # this is a tmp object, delete it to prevent bugs

    def callback():
        return

    def new_meta_info():
        return dict(category=run_cfg.category, fold_style=run_cfg.style)
    
    # start generate gt
    logger.info("Start generating fold ground truth ...")
    if run_cfg.generate_fold_gt_and_compute_error:
        all_render_dir = []
        fold_gt_traj_dir = os.path.abspath("fold_gt/traj")
        fold_gt_mesh_dir = os.path.abspath("fold_gt/mesh")
        old_skip_rotate = policy_gt.skip_rotate
        policy_gt.skip_rotate = True

        for i, (rot_z, flip_y) in enumerate(policy_gt.get_all_possible_rot_z_flip_y()):
            meta_info = new_meta_info()
            
            traj_dir = os.path.join(fold_gt_traj_dir, f"traj_{utils.format_int(i, 4)}")
            env_kwargs = dict(overwrite_render=run_cfg.generate_fold_gt_overwrite_render, callback=callback)
            env.set_render_output(traj_dir)
            env.perfect_init_cloth(rot_z, flip_y, **env_kwargs)
            while True:
                action = policy_gt.get_action()
                if action is not None:
                    env.step(**action.asdict_to_env(), **env_kwargs)
                else:
                    env.post_fold(**env_kwargs)
                    path = os.path.join(fold_gt_mesh_dir, f"{rot_z}_{flip_y}", "mesh.obj")
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    env.get_raw_mesh_curr().export(path)
                    meta_info["ik_fail_count"] = env.ik_fail_count
                    break
            
            env.reset()
            policy_gt.reset()

            append_render_dir(env, all_render_dir)
            export_meta_info_and_clear(traj_dir, meta_info, "meta_info_gt.json")

        env.sync()
        if run_cfg.generate_video:
            generate_video(env.get_render_fps(), all_render_dir)
        
        env.load_gt_mesh(fold_gt_mesh_dir)
        policy_gt.skip_rotate = old_skip_rotate
    
    # start rollout policy
    logger.info("Start collecting trajectories ...")
    all_render_dir = []
    for i in range(run_cfg.traj_num):
        meta_info = new_meta_info()

        logger.info("pre policy ...")
        traj_dir = f"traj_{utils.format_int(i, run_cfg.traj_num - 1)}"
        env.set_render_output(traj_dir)
        policy_run.set_save_dir(os.path.join(traj_dir, "policy"))
        pre_post_kwargs = dict(overwrite_render=run_cfg.pre_and_post_overwrite_render, callback=callback)
        if run_cfg.random_init_cloth:
            env.random_init_cloth(**pre_post_kwargs)
        else:
            env.perfect_init_cloth(0., False, **pre_post_kwargs)
        meta_info["start_step"] = env.current_step_idx

        logger.info("rollout policy ...")
        while True:
            state_path = os.path.join(traj_dir, "state", f"{env.current_step_idx - 1}.json")
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            utils.dump_json(state_path, policy_run.get_robot_state())

            action = policy_run.get_action()
            if action is not None:
                action_path = os.path.join(traj_dir, "action", f"{env.current_step_idx - 1}.json")
                os.makedirs(os.path.dirname(action_path), exist_ok=True)
                utils.dump_json(action_path, policy_run.delta_action(action).asdict_to_save())
                env.step(**action.asdict_to_env(), callback=callback)
            
            else:
                logger.info("post policy ...")
                meta_info["end_step"] = env.current_step_idx
                env.post_fold(**pre_post_kwargs)
                env.render()
                env.get_raw_mesh_curr().export(os.path.join(traj_dir, "final_mesh.obj"), include_texture=False)
                if run_cfg.generate_fold_gt_and_compute_error:
                    err = env.compute_error()
                    meta_info["err"] = err
                else:
                    meta_info["err"] = -1.
                meta_info["ik_fail_count"] = env.ik_fail_count
                meta_info["policy"] = policy_run.get_meta_info()
                break
        
        if cfg.policy.run.name == "hybrid":
            hybrid_action_post_process(os.path.join(traj_dir, "action"))
        
        env.reset()
        policy_run.reset()
        if run_cfg.sync_per_traj:
            env.sync()

        append_render_dir(env, all_render_dir)
        export_meta_info_and_clear(traj_dir, meta_info, "meta_info_demo.json")

    env.sync()
    if run_cfg.generate_video:
        generate_video(env.get_render_fps(), all_render_dir)

    # close env
    env.close()
    logger.info("Env closed ...")


if __name__ == '__main__':
    main()