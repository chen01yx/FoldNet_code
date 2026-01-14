import logging
logger = logging.getLogger(__name__)

import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional
import os
import tqdm
import time

import omegaconf

from garmentds.real_galbot.realapi import RealAPI, RealAPICfg
from garmentds.real_galbot.policy import FoldRealVisualPolicy, FoldRealVisualPolicyCfg
from garmentds.real_galbot.env import FoldRealEnv, FoldRealEnvCfg
from garmentds.foldenv.fold_env import RobotCfg


def easy_input(
    args_and_desc: Optional[dict[str, str]] = None,
    prompt_prefix: str = "Please enter",
):
    while True:
        prompt = prompt_prefix + ":\n" + "\n".join([f"'{k}': {v}" for k, v in args_and_desc.items()]) + "\n"
        user_input = input(prompt)
        if user_input in args_and_desc.keys() or args_and_desc is None:
            return user_input
        print(f"Invalid input: {user_input}, please try again.")


@dataclass
class RunCfg:
    step_dt: float = 0.2
    asynchronous: bool = True
    mode: Literal["test", "policy"] = "test"
    ask_before_policy_action: bool = True


class RealWorkspace:
    def __init__(self, realapi_cfg: omegaconf.DictConfig, env_cfg: omegaconf.DictConfig, policy_cfg: omegaconf.DictConfig, run_cfg: omegaconf.DictConfig):
        realapi_cfg_dataclass = RealAPICfg(
            robot_cfg=RobotCfg(
                device="cpu", ik_kwargs=dict(max_iter=16, square_err_th=1e-6),
                urdf_path="asset/galbot_one_charlie/urdf_nomtl.urdf"
            ), 
            **realapi_cfg,
        )
        self._api = RealAPI(realapi_cfg_dataclass)
        
        env_cfg_dataclass = FoldRealEnvCfg(**env_cfg)
        self._env = FoldRealEnv(env_cfg_dataclass, self._api)
        
        policy_cfg_dataclass = FoldRealVisualPolicyCfg(**policy_cfg)
        self._policy = FoldRealVisualPolicy(policy_cfg_dataclass, self._env)
        
        self._run_cfg = RunCfg(**run_cfg)
        
        # sanity check
        assert env_cfg_dataclass.use_mask == self._policy.use_masked_rgb, f"{env_cfg_dataclass.use_mask} {self._policy.use_masked_rgb}"
        
    def _run_test(self, run_cfg: RunCfg):
        self._api.moveinit()
        self._api.step(np.array([-0.4, -0.1, +0.2]), np.array([+0.4, -0.1, +0.2]))
        self._api.step(np.array([-0.4, -0.1, +0.1]), np.array([+0.4, -0.1, +0.1]))
        self._api.step(np.array([-0.4, -0.1, +0.05]), np.array([+0.4, -0.1, +0.05]))
        self._api.step(np.array([-0.4, -0.1, +0.01]), np.array([+0.4, -0.1, +0.01]))
        self._api.moveg(self._env.PICKER_CLOSE, self._env.PICKER_CLOSE)
        self._api.moveg(self._env.PICKER_OPEN, self._env.PICKER_OPEN)
    
    def _single_traj(self, traj_idx: int, run_cfg: RunCfg):
        logger.info(f"Run single trajectory {traj_idx}")
        traj_dir = f"traj_{traj_idx}"
        self._env.set_render_output(traj_dir)
        self._policy.set_save_dir(os.path.join(traj_dir, "policy"))
        self._env.before_trajectory()
        
        ans = easy_input({"": "continue", "q": "quit"}, "Press enter to start the trajectory, or q to quit")
        if ans == "q":
            return
        
        tqdm_bar = tqdm.tqdm()
        curr_step = 0
        while True:
            logger.info(f"Start running ... current step: {curr_step}")
            tic = time.time()
            action = self._policy.get_action()
            if run_cfg.ask_before_policy_action:
                ans = easy_input({"": "continue", "q": "quit"}, f"Whether to accept policy output action? {action}")
                if ans == "q":
                    break
            if action is None:
                break
            self._env.set_parameters_to_visualize(self._policy.get_action_pc())
            self._env.step(
                action.xyz_l, action.xyz_r, action.picker_l, action.picker_r, 
                self._run_cfg.asynchronous and curr_step > 0
            )
            toc = time.time()
            time.sleep(max(0, self._run_cfg.step_dt - (toc - tic)))
            
            tqdm_bar.update(1)
            curr_step += 1
        
        tqdm_bar.close()
        self._policy.reset()
    
    def _run_policy(self, run_cfg: RunCfg):
        traj_idx = 0
        while True:
            self._single_traj(traj_idx, run_cfg)
            traj_idx += 1
            
            ans = easy_input({"": "continue", "q": "quit"}, "Whether to continue to next trajectory?")
            if ans == "q":
                break
            
    def run(self):
        run_cfg = self._run_cfg
        if run_cfg.mode == "test":
            self._run_test(run_cfg)
        elif run_cfg.mode == "policy":
            self._run_policy(run_cfg)
        else:
            raise ValueError(f"Unknown mode: {run_cfg.mode}")