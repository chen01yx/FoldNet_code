import logging
logger = logging.getLogger(__name__)

from dataclasses import dataclass, field
from typing import Optional, Literal
from collections import defaultdict, deque
import os
import pprint

import numpy as np
import torch
import trimesh
import torchvision.transforms.v2 as transforms

from garmentds.foldenv.fold_env import FoldEnv, Picker
from garmentds.foldenv.policy.base import FoldPolicy, FoldPolicyCfg, FoldPolicyAction

import garmentds.common.utils as utils
from garmentds.foldenv.fold_learn import FoldPolicyModule, FoldPolicyDataset, load_json

timer = utils.Timer(name="visual_policy", logger=logger)


class TF:
    def __init__(self, height: int, width: int):
        self.rgb = transforms.Compose([
            transforms.Resize((height, width)), 
            transforms.Lambda(lambda x: torch.clamp(x, 0., 1.)),
        ])
        self.mask = transforms.Compose([transforms.Resize((height, width))])
        self.tcp = transforms.Compose([transforms.Resize((height, width))])
    
    def tf(self, rgb: np.ndarray, mask: np.ndarray, tcp: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb = self.rgb(torch.tensor(rgb))
        mask = self.mask(torch.tensor(mask))
        tcp = self.tcp(torch.tensor(tcp))
        return rgb, mask, tcp


@dataclass
class FoldVisualPolicyCfg(FoldPolicyCfg):
    name: str = None
    ckpt: str = None
    stop_delta_action_threshold: float = 1e-3
    stop_max_steps: int = 300
    action_smooth: float = 1e-3
    action_gripper_discrete: bool = True
    action_inference_every_n_step: Optional[int] = 4
    device: str = "cuda"
    ddpm_inference_timestep: int = 100
    log_network_results: bool = True

    def __post_init__(self):
        super().__post_init__()
        assert self.ckpt is not None, "ckpt must be specified"
        self.ckpt = utils.get_path_handler()(self.ckpt)
        assert self.name in ["act", "mlp", "dp"], f"name must be one of ['act', 'mlp', 'dp'], but got {self.name}"


@dataclass
class ActionCache:
    step: int # only for debug
    action: Optional[np.ndarray]

    def __post_init__(self):
        if self.action is not None:
            assert isinstance(self.action, np.ndarray), self.action


class FoldVisualPolicy(FoldPolicy):
    def __init__(self, cfg: FoldVisualPolicyCfg, env: FoldEnv):
        super().__init__(cfg, env)
        self._cfg: FoldVisualPolicyCfg

        self._model = FoldPolicyModule.load_from_checkpoint(
            checkpoint_path=cfg.ckpt, map_location=torch.device("cpu"),
        ).to(device=self._cfg.device).eval()
        self._height = self._model.height
        self._width = self._model.width
        self._dtype_np = np.dtype(self._model.dtype_str)
        self._dtype_th = getattr(torch, self._model.dtype_str)
        self._tf = TF(self._height, self._width)

        self._policy_cache = defaultdict(list)
        self._frame_idx_to_prepare = deque()
        self._current_action_step = 0

        if self._cfg.action_inference_every_n_step is None:
            self._action_inference_every_n_step = self._model.actpred_len // 2
        else:
            self._action_inference_every_n_step = int(self._cfg.action_inference_every_n_step)

        assert self._action_inference_every_n_step <= self._model.actpred_len, \
            f"action_inference_every_n_step:{self._action_inference_every_n_step} must be less than or equal to actpred_len:{self._model.actpred_len}"
    
    def _tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(dtype=self._dtype_th, device=self._cfg.device)
        else:
            return torch.tensor(np.array(x), dtype=self._dtype_th, device=self._cfg.device)

    def _is_stop_action(self, delta_action: FoldPolicyAction, absolute_action: FoldPolicyAction) -> bool:
        if self._current_action_step >= self._cfg.stop_max_steps:
            return True
        else:
            is_stop = (
                np.linalg.norm(delta_action.xyz_l) < self._cfg.stop_delta_action_threshold and
                np.linalg.norm(delta_action.xyz_r) < self._cfg.stop_delta_action_threshold and
                Picker.is_open_action(absolute_action.picker_l) and Picker.is_open_action(absolute_action.picker_r) and
                np.abs(delta_action.picker_l) < self._cfg.stop_delta_action_threshold and
                np.abs(delta_action.picker_r) < self._cfg.stop_delta_action_threshold
            )
            if is_stop:
                logger.info(f"Stop action detected at step {self._current_action_step}")
                logger.info(pprint.pformat(delta_action.__dict__, sort_dicts=False))
                logger.info(pprint.pformat(absolute_action.__dict__, sort_dicts=False))
            return is_stop
    
    @timer.timer
    def _prepare_obs(self, frame_idx: int, state: np.ndarray) -> dict[Literal["rgb", "mask", "tcp"], np.ndarray]:
        """sync rendering process and get frame_idx observation"""
        # frame_idx = last_render_frame_idx
        self._env.sync() # wait for rendering
        img_path = os.path.join(self._env.get_render_output(), "head_rgb_mask", f"{frame_idx:04d}.npy")
        cam_path = os.path.join(self._env.get_render_output(), "head_cam_param", f"{frame_idx:04d}.json")
        rgb, mask = FoldPolicyDataset.load_img_and_process(img_path, self._dtype_np)
        tcp = FoldPolicyDataset.compute_tcp_mask(
            load_json(cam_path), rgb.shape[1], rgb.shape[2], 
            state[0:3], state[3:6], self._dtype_np
        ) # [3, 480, 640]
        return dict(rgb=rgb, mask=mask, tcp=tcp)
    
    def _prepare_current_state(self):
        s = self.get_robot_state()
        return FoldPolicyDataset.state_dict_to_arr(s)
    
    def _after_average_action(self, action_avg: np.ndarray) -> tuple[FoldPolicyAction, FoldPolicyAction]:
        absolute_action = FoldPolicyAction(**FoldPolicyDataset.action_arr_to_dict(action_avg))
        if self._cfg.action_gripper_discrete:
            absolute_action.picker_l = Picker.OPEN if Picker.is_open_action(absolute_action.picker_l) else Picker.CLOSE
            absolute_action.picker_r = Picker.OPEN if Picker.is_open_action(absolute_action.picker_r) else Picker.CLOSE
        delta_action = self.delta_action(absolute_action)
        return delta_action, absolute_action
    
    def _save_action_seq_as_point_cloud(self, action_seq: np.ndarray, ply_dir: str):
        xyz = []
        rgb = []
        for a in action_seq:
            xyz.append(a[0:3].copy())
            xyz.append(a[3:6].copy())
            rgb.append([0.5 + 0.5 * max(min(a[6], 1.), 0.), 0., 0.])
            rgb.append([0., 0.5 + 0.5 * max(min(a[7], 1.), 0.), 0.])
        
        xyz_clothes = self._env.get_raw_mesh_curr().sample(1000)
        rgb_clothes = np.ones_like(xyz_clothes)

        xyz_robot = self._env.get_robot_mesh().sample(2000)
        rgb_robot = np.zeros_like(xyz_robot) + np.array([0., 1., 1.])

        trimesh.PointCloud(
            np.concatenate([xyz, xyz_clothes, xyz_robot]),
            colors=np.concatenate([rgb, rgb_clothes, rgb_robot]),
        ).export(ply_dir)
    
    @timer.timer
    def _model_compute_action_pred(self, net_input: list[torch.Tensor]):
        with torch.no_grad():
            action_seq, detail_info = self._model.compute_action_pred(
                net_input, mode="infer", ddpm_inference_timestep=self._cfg.ddpm_inference_timestep
            )
        return action_seq, detail_info
    
    @timer.timer
    def _log_network_results(self, detail_info: dict, action_seq_abs_np: np.ndarray, net_input: dict):
        self._model.log_img_eval(detail_info, os.path.join(self._save_dir, f"{self._current_action_step}.pdf"))
        self._save_action_seq_as_point_cloud(action_seq_abs_np, os.path.join(self._save_dir, f"{self._current_action_step}.ply"))
        np.save(os.path.join(self._save_dir, f"{self._current_action_step}.npy"), action_seq_abs_np)
        np.save(os.path.join(self._save_dir, f"{self._current_action_step}_net_input.npy"), utils.torch_dict_to_numpy_dict(net_input), allow_pickle=True)

    @timer.timer
    def _inference_net(self):
        net_input = defaultdict(list)
        for k in ["rgb", "mask", "tcp", "state"]:
            if k in ["rgb", "mask", "tcp"]:
                horizon = self._model.obs_horizon
            elif k == "state":
                horizon = self._model.sta_horizon
            else:
                raise ValueError(f"unknown key {k}")
            for i in range(horizon):
                idx = max(len(self._policy_cache[k]) - i - 1, 0)
                net_input[k].append(self._policy_cache[k][idx])
                logger.info(f"in _inference_net, put {k} cache at {idx} in net_input at {i}")
                
        rgb_th, mask_th, tcp_th = self._tf.tf(
            np.array(net_input["rgb"], dtype=self._dtype_np), 
            np.array(net_input["mask"], dtype=self._dtype_np),
            np.array(net_input["tcp"], dtype=self._dtype_np)
        )
        net_input["rgb"], net_input["mask"], net_input["tcp"] = rgb_th.unsqueeze(0), mask_th.unsqueeze(0), tcp_th.unsqueeze(0)
        net_input["state"] = torch.tensor(np.array(net_input["state"], dtype=self._dtype_np)[None, ...])
        for k, v in net_input.items():
            net_input[k] = v.to(device=self._model.device)
        
        # inference
        action_seq, detail_info = self._model_compute_action_pred(net_input)
        action_seq_abs = self._model.convert_pred_action_seq_to_abs_action_seq(action_seq, net_input["state"]).squeeze(0)
        action_seq_abs_np = utils.torch_to_numpy(action_seq_abs)

        # log network results
        if self._cfg.log_network_results:
            self._log_network_results(detail_info, action_seq_abs_np, net_input)

        return action_seq_abs_np
            
    def _compute_action_to_execute(self) -> tuple[FoldPolicyAction, FoldPolicyAction]:
        """delta to absolute action and average action"""
        action_list = []
        weight_list = []
        for i in range(self._model.actpred_len):
            if len(self._policy_cache["action"]) > i:
                idx = len(self._policy_cache["action"]) - i - 1
                action_seq: ActionCache = self._policy_cache["action"][idx]
                if action_seq.action is not None:
                    action_list.append(action_seq.action[i, ...])
                    weight_list.append(self._cfg.action_smooth ** i)
                    logger.info(f"in _compute_action_to_execute, use action cache at {idx} (step:{action_seq.step}) with index {i}, weight {weight_list[-1]}")
        logger.info(f"in _compute_action_to_execute, all weights: {weight_list}")
        action_avg = np.average(action_list, axis=0, weights=weight_list)
        delta_action, absolute_action = self._after_average_action(action_avg)
        logger.info(f"action_to_excute:\n{pprint.pformat(delta_action.asdict_to_env())}\n{pprint.pformat(absolute_action.asdict_to_env())}")
        return delta_action, absolute_action
    
    @timer.timer
    def get_action(self) -> Optional[FoldPolicyAction]:
        if self._current_action_step == 0:
            action = self._process_action(self._get_pre_action())

        else:
            # obs: append frame_idx to deque, render process and sim process is not synchronized to speed up
            self._frame_idx_to_prepare.append((self._env.last_render_frame_idx, len(self._policy_cache["state"])))
            # state: directly append current state
            self._policy_cache["state"].append(self._prepare_current_state())

            if len(self._policy_cache["action"]) % self._action_inference_every_n_step == 0:
                # prepare all observations and put into policy cache
                while len(self._frame_idx_to_prepare) > 0:
                    frame_idx, state_idx = self._frame_idx_to_prepare.popleft()
                    obs = self._prepare_obs(frame_idx, self._policy_cache["state"][state_idx])
                    self._policy_cache["rgb"].append(obs["rgb"])
                    self._policy_cache["mask"].append(obs["mask"])
                    self._policy_cache["tcp"].append(obs["tcp"])
                    logger.info(f"in get_action, put {frame_idx} into policy cache at {len(self._policy_cache['rgb']) - 1}")
                # inference net
                action_seq = self._inference_net()
                self._policy_cache["action"].append(ActionCache(self._current_action_step, action_seq))
            else:
                self._policy_cache["action"].append(ActionCache(self._current_action_step, None))
            
            delta_action, absolute_action = self._compute_action_to_execute()
            if self._is_stop_action(delta_action, absolute_action):
                action = None
            else:
                action = self._process_action(absolute_action)
        
        self._current_action_step += 1
        return action
    
    def reset(self):
        super().reset()
        self._policy_cache.clear()
        self._frame_idx_to_prepare.clear()
        self._current_action_step = 0