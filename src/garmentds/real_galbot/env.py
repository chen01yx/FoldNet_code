import logging
logger = logging.getLogger(__name__)

import os
from typing import Optional
from dataclasses import dataclass, field
import json

from PIL import Image
import numpy as np

import garmentds.common.utils as utils
from garmentds.foldenv.fold_env import FoldEnv
from garmentds.real_galbot.realapi import RealAPI
from garmentds.real.sam_utils import GroundedSAM
from garmentds.real_galbot.mask_app import App, AppCfg

real_env_timer = utils.Timer(name="real_env", logger=logger)
gsam = GroundedSAM() # init as a global variable to avoid hydra errors


@dataclass
class FoldRealEnvState:
    step_idx: int = field(init=False, default=0)
    render_frame_idx: int = field(init=False, default=0)
    
    def reset(self):
        self.step_idx = 0
        self.render_frame_idx = 0


@dataclass
class FoldRealEnvCfg:
    render_output_dir: str = "tmp_render_output"
    use_mask: bool = False
        

class FoldRealEnv(FoldEnv):
    def __init__(self, cfg: FoldRealEnvCfg, realapi: RealAPI):
        self._cfg = cfg
        self._api = realapi
        self._state = FoldRealEnvState()
        
        self._robot = self._api.robot_policy
        self._cache = dict()
    
    @real_env_timer.timer
    def _compute_mask(self, rgb: np.ndarray) -> np.ndarray:
        if self._cfg.use_mask:
            mask, _ = gsam.video_tracker.segment_and_update(rgb)
        else:
            mask = np.ones(rgb.shape[:2], dtype=rgb.dtype)
        return mask
    
    @real_env_timer.timer
    def _get_obs(self):
        rgbd = self._api.get_rgbd()
        camera_extrinsics = self._api.get_camera_extrinsics().tolist()
        camera_intrinsics = self._api.get_camera_intrinsics().tolist()
        rgb, depth = rgbd["color"], rgbd["depth"]
        mask = self._compute_mask(rgb)
        state = dict(
            tcp_xyz=self.get_tcp_xyz(),
            gripper_state=self.get_gripper_state(),
        )
        return rgb, depth, mask, camera_extrinsics, camera_intrinsics, state
    
    @real_env_timer.timer
    def _save_obs(self, step_idx: int, rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray, camera_extrinsics, camera_intrinsics, state):
        cfg = self._cfg
        
        with real_env_timer.context_manager("save_rgb"):
            rgb_path = os.path.join(cfg.render_output_dir, "head", f"{str(step_idx).zfill(4)}.png")
            os.makedirs(os.path.dirname(rgb_path), exist_ok=True)
            Image.fromarray(rgb).save(rgb_path)
            self._cache["rgb"] = rgb
            self._cache["depth"] = depth
        with real_env_timer.context_manager("save_npy"):
            npy_path = os.path.join(cfg.render_output_dir, "head_rgb_mask", f"{str(step_idx).zfill(4)}.npy")
            os.makedirs(os.path.dirname(npy_path), exist_ok=True)
            np.save(npy_path, np.concatenate([rgb.transpose(2, 0, 1), mask[None, :, :]], axis=0).astype(np.uint8))
            self._cache["mask"] = mask
        with real_env_timer.context_manager("save_mask"):
            png_path = os.path.join(cfg.render_output_dir, "head_mask_png", f"{str(step_idx).zfill(4)}.png")
            os.makedirs(os.path.dirname(png_path), exist_ok=True)
            Image.fromarray(np.where(np.tile(mask[:, :, None], (1, 1, 3)), rgb, 255)).save(png_path)
        with real_env_timer.context_manager("save_cam_param"):
            cam_param_path = os.path.join(cfg.render_output_dir, "head_cam_param", f"{str(step_idx).zfill(4)}.json")
            os.makedirs(os.path.dirname(cam_param_path), exist_ok=True)
            with open(cam_param_path, "w") as f:
                cam_param = dict(
                    camera_extrinsics=camera_extrinsics, 
                    camera_intrinsics=camera_intrinsics
                )
                json.dump(cam_param, f, indent=4)
            self._cache["cam_param"] = cam_param
        with real_env_timer.context_manager("save_state"):
            state_path = os.path.join(cfg.render_output_dir, "state", f"{step_idx}.json")
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            with open(state_path, "w") as f:
                json.dump(state, f, indent=4, default=utils.custom_serializer)
    
    @real_env_timer.timer
    def _get_and_save_obs(self, step_idx: int):
        obs = self._get_obs()
        self._save_obs(step_idx, *obs)
    
    def before_trajectory(self):
        self._api.moveinit()
        if self._cfg.use_mask:
            rgbd = self._api.get_rgbd()
            mask = App(AppCfg(), gsam).run(rgbd["color"]) # [480, 640], bool, False, True
            gsam.video_tracker.set_init_mask(mask)
    
    def get_point_cloud(self):
        """get depth sensor point cloud use cache"""
        try:
            rgb, depth, mask, cam_param = self._cache["rgb"], self._cache["depth"], self._cache["mask"], self._cache["cam_param"]
            H, W = depth.shape
            u, v = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
            w = np.ones((H, W))
            uvw = np.stack([u[np.where(mask)], v[np.where(mask)], w[np.where(mask)]], axis=0) # [3, N]
            xyz_c = np.linalg.inv(np.array(cam_param["camera_intrinsics"])) @ uvw
            xyzw_c = np.concatenate([xyz_c * depth[np.where(mask)], np.ones((1, xyz_c.shape[1]))], axis=0) # [4, N]
            xyzw_w = np.array(cam_param["camera_extrinsics"]) @ np.diag([1., -1., -1., 1.]) @ xyzw_c # [4, N]
            return xyzw_w[:3, :].T, np.clip(rgb[np.where(mask)] / 255., 0., 1.) # [N, 3]
        except KeyError:
            return None
        except Exception as e:
            raise e    
    
    def set_parameters_to_visualize(self, action_pc: Optional[np.ndarray]):
        self._api.set_parameters_to_visualize(
            action_pc,
            self.get_point_cloud(), 
            self._cache.get("cam_param", {}).get("camera_extrinsics", None), 
            self._cache.get("cam_param", {}).get("camera_intrinsics", None)
        )
    
    @real_env_timer.timer
    def step(
        self, 
        xyz_l: Optional[np.ndarray]=None, xyz_r: Optional[np.ndarray]=None, 
        picker_l: Optional[float]=None, picker_r: Optional[float]=None,
        asynchronous=False, 
    ):
        """overwrite original function"""
        state = self._state
        self._api.step(xyz_l, xyz_r, picker_l, picker_r, asynchronous=asynchronous)
        self._get_and_save_obs(state.step_idx) # we use file to transfer data between env and policy
        state.step_idx += 1
        state.render_frame_idx += 1
    
    def sync(self):
        """overwrite original function"""
        pass # sync is not needed for real env