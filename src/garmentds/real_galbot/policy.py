import logging
logger = logging.getLogger(__name__)

from dataclasses import dataclass, field
from collections import defaultdict
import os
import json
from typing import Optional, Literal

import trimesh
import numpy as np
import torch

import garmentds.common.utils as utils
from garmentds.foldenv.policy.visual import FoldVisualPolicyCfg, FoldVisualPolicy
from garmentds.foldenv.fold_learn import FoldPolicyDataset
from garmentds.real_galbot.env import FoldRealEnv

real_policy_timer = utils.Timer(name="real_policy", logger=logger)


@dataclass
class FoldRealVisualPolicyCfg(FoldVisualPolicyCfg):
    pass


class FoldRealVisualPolicy(FoldVisualPolicy):
    def __init__(self, cfg: FoldRealVisualPolicyCfg, env: FoldRealEnv):
        super().__init__(cfg, env)
        self._cfg: FoldRealVisualPolicyCfg
        self._env: FoldRealEnv
        self._action_pc: Optional[np.ndarray] = None
    
    def get_action_pc(self):
        return self._action_pc.copy() if self._action_pc is not None else None
    
    @real_policy_timer.timer
    def _save_action_seq_as_point_cloud(self, action_seq: np.ndarray, ply_dir: str):
        """overwrite original function"""
        xyz = []
        rgb = []
        for a in action_seq:
            xyz.append(a[0:3].copy())
            xyz.append(a[3:6].copy())
            rgb.append([0.5 + 0.5 * max(min(a[6], 1.), 0.), 0., 0.])
            rgb.append([0., 0.5 + 0.5 * max(min(a[7], 1.), 0.), 0.])
        self._action_pc = np.concatenate([xyz, rgb], axis=1)
        
        xyz_clothes, rgb_clothes = self._env.get_point_cloud()

        xyz_robot = self._env.get_robot_mesh().sample(2000)
        rgb_robot = np.zeros_like(xyz_robot) + np.array([0., 1., 1.])

        trimesh.PointCloud(
            np.concatenate([xyz, xyz_clothes, xyz_robot]),
            colors=np.concatenate([rgb, rgb_clothes, rgb_robot]),
        ).export(ply_dir)
    
    @real_policy_timer.timer
    def _prepare_obs(self, frame_idx: int, state: np.ndarray) -> dict[Literal["rgb", "mask", "tcp"], np.ndarray]:
        """overwrite original function"""
        obs = super()._prepare_obs(frame_idx, state)
        # obs["rgb"] = np.tile(np.mean(obs["rgb"], axis=0, keepdims=True), (3, 1, 1))
        return obs
    
    @property
    def use_masked_rgb(self):
        return self._model.use_masked_rgb