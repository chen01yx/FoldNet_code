import logging
logger = logging.getLogger(__name__)

import copy
from dataclasses import dataclass, asdict, field
import abc
from typing import Optional, Any
import pprint

import numpy as np

from garmentds.foldenv.fold_env import FoldEnv


@dataclass
class FoldPolicyAction:
    xyz_l: Optional[np.ndarray] = None
    xyz_r: Optional[np.ndarray] = None
    picker_l: Optional[float] = None
    picker_r: Optional[float] = None
    is_correct_action: bool = True

    def asdict(self):
        return asdict(self)

    def asdict_to_env(self):
        d = asdict(self)
        d.pop("is_correct_action")
        return d

    def asdict_to_save(self):
        return asdict(self)


@dataclass
class FoldPolicyCfg(abc.ABC):
    init_xyz_l: np.ndarray = field(default_factory=lambda: np.array([-0.4, -0.1, +0.2]))
    init_xyz_r: np.ndarray = field(default_factory=lambda: np.array([+0.4, -0.1, +0.2]))
    xyz_range: np.ndarray = field(default_factory=lambda: np.array([
        [-0.6, -0.4, 0.0],
        [+0.6, +0.4, 0.5],
    ]))

    def __post_init__(self):
        self.init_xyz_l = np.array(self.init_xyz_l)
        self.init_xyz_r = np.array(self.init_xyz_r)
        self.xyz_range = np.array(self.xyz_range)


class FoldPolicy(abc.ABC):
    _DEFLAUT_SAVE_DIR = "policy_output"
    def __init__(self, cfg: FoldPolicyCfg, env: FoldEnv):
        cfg = copy.deepcopy(cfg)
        self._cfg = cfg
        self._env = env
        self._save_dir: str = self._DEFLAUT_SAVE_DIR
        self._meta_info = dict()

    def _process_action(self, action: Optional[FoldPolicyAction]) -> Optional[FoldPolicyAction]:
        if action is not None:
            logger.info(f"process action raw: {pprint.pformat(action.asdict())}")
            for a in ["xyz_l", "xyz_r"]:
                v = getattr(action, a)
                if v is not None: # clip action xy
                    v = np.clip(v, self._cfg.xyz_range[0], self._cfg.xyz_range[1])
                setattr(action, a, v)
            for a in ["picker_l", "picker_r"]:
                v = getattr(action, a)
                if v is not None: # clip action picker
                    v = np.clip(v, 0., 1.)
                setattr(action, a, v)
            logger.info(f"process action processed: {pprint.pformat(action.asdict())}")
        else:
            logger.info("process action None action")
        return action
    
    def _get_pre_action(self):
        return FoldPolicyAction(
            xyz_l=self._cfg.init_xyz_l,
            xyz_r=self._cfg.init_xyz_r,
            picker_l=self._env.PICKER_OPEN,
            picker_r=self._env.PICKER_OPEN,
        )

    def expand_action(self, action: FoldPolicyAction):
        tcp_xyz = self._env.get_tcp_xyz()
        gripper_state = self._env.get_gripper_state()
        return FoldPolicyAction(
            xyz_l=(action.xyz_l if action.xyz_l is not None else tcp_xyz["left"]).copy(),
            xyz_r=(action.xyz_r if action.xyz_r is not None else tcp_xyz["right"]).copy(),
            picker_l=action.picker_l if action.picker_l is not None else gripper_state["left"],
            picker_r=action.picker_r if action.picker_r is not None else gripper_state["right"],
        )
    
    def delta_action(self, action: FoldPolicyAction) -> FoldPolicyAction:
        tcp_xyz = self._env.get_tcp_xyz()
        gripper_state = self._env.get_gripper_state()
        return FoldPolicyAction(
            xyz_l=(action.xyz_l - tcp_xyz["left"] if action.xyz_l is not None else np.zeros(3)),
            xyz_r=(action.xyz_r - tcp_xyz["right"] if action.xyz_r is not None else np.zeros(3)),
            picker_l=action.picker_l - gripper_state["left"] if action.picker_l is not None else 0.,
            picker_r=action.picker_r - gripper_state["right"] if action.picker_r is not None else 0.,
            is_correct_action=action.is_correct_action,
        )
    
    def absolute_action(self, action: FoldPolicyAction) -> FoldPolicyAction:
        tcp_xyz = self._env.get_tcp_xyz()
        gripper_state = self._env.get_gripper_state()
        return FoldPolicyAction(
            xyz_l=(action.xyz_l + tcp_xyz["left"] if action.xyz_l is not None else tcp_xyz["left"]).copy(),
            xyz_r=(action.xyz_r + tcp_xyz["right"] if action.xyz_r is not None else tcp_xyz["right"]).copy(),
            picker_l=action.picker_l + gripper_state["left"] if action.picker_l is not None else gripper_state["left"],
            picker_r=action.picker_r + gripper_state["right"] if action.picker_r is not None else gripper_state["right"],
            is_correct_action=action.is_correct_action,
        )
    
    def get_meta_info(self) -> dict:
        return copy.deepcopy(self._meta_info)

    def get_robot_state(self):
        return dict(
            tcp_xyz=self._env.get_tcp_xyz(),
            gripper_state=self._env.get_gripper_state(),
        )
    
    @abc.abstractmethod
    def get_action(self, *args, **kwargs):
        pass

    def set_save_dir(self, save_dir: str):
        self._save_dir = str(save_dir)

    def reset(self):
        self._save_dir = self._DEFLAUT_SAVE_DIR
        self._meta_info.clear()