import logging
logger = logging.getLogger(__name__)

from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Any
import pprint
import copy

import numpy as np
import trimesh

from garmentds.foldenv.fold_env import FoldEnv
import garmentds.foldenv.policy_utils as policy_utils
from garmentds.foldenv.policy.base import FoldPolicy, FoldPolicyCfg, FoldPolicyAction


PICKER_Z, GRASP_TH = 0.02, 0.02
@dataclass
class FoldStatePolicyCfg(FoldPolicyCfg):
    cloth_scale: float = None
    skip_rotate: bool = False

    single_step_length: float = 0.03
    
    enable_not_correct_action: bool = True
    not_correct_action_prob: float = 0.0
    not_correct_action_xyz_std: np.ndarray = (0.05, 0.05, 0.01)
    grasp_nothing_step_range: tuple[int, int] = (2, 5) # [2, 3, 4]

    move_away_l: np.ndarray = field(default_factory=lambda: np.array([-0.02, 0.02, +0.02])) # y'sign is depend on origin y'sign
    move_away_r: np.ndarray = field(default_factory=lambda: np.array([+0.02, 0.02, +0.02])) # y'sign is depend on origin y'sign

    def _scale(self):
        raise NotImplementedError

    def __post_init__(self):
        super().__post_init__()
        assert self.cloth_scale is not None, "cloth_scale must be set"
        self._scale()

        self.not_correct_action_xyz_std = np.array(self.not_correct_action_xyz_std)
        self.move_away_l = np.array(self.move_away_l)
        self.move_away_r = np.array(self.move_away_r)


class NotCorrectActionModule:
    def __init__(self, cfg: FoldStatePolicyCfg, env: FoldEnv, policy: "FoldStatePolicy"):
        self.cfg = cfg
        self.env = env
        self.policy = policy
        self.reset()
    
    def reset(self):
        self.prev_action = FoldPolicyAction()
        self.curr_action = FoldPolicyAction()
        self.force_correct_stage = None
        self.stage_detect_grasp_fail: set[int] = set()
        self.prepare_low_freq_random_action()
    
    def set_current_action(self, action: FoldPolicyAction):
        self.prev_action = copy.deepcopy(self.curr_action)
        self.curr_action = copy.deepcopy(action)
        self.curr_step += 1
    
    def get_current_action(self):
        if self.curr_action is None:
            return None
        
        self.detect_grasp_fail()
        action = self.random_not_correct_action(self.curr_action, self.prev_action)
        self.curr_action = copy.deepcopy(action)
        return self.curr_action
    
    @staticmethod
    def cos_function(wave: np.ndarray, x: np.ndarray):
        assert len(wave.shape) == 2 and wave.shape[1] == 3 and len(x.shape) == 1
        w = wave[None, :, 0]
        a = wave[None, :, 1]
        p = wave[None, :, 2]
        return np.sum(np.cos(2 * np.pi * (w * x[:, None] + p)) * a, axis=1)

    def prepare_low_freq_random_action(self):
        self.curr_step = np.array([0])

        N = 10
        deviation, randprob = self.cfg.not_correct_action_xyz_std, self.cfg.not_correct_action_prob
        self.wave_xyz = np.random.rand(2, 3, N, 3) * np.array([0.05, 1., 1.]) # 2 for left/right, 3 for xyz, 3 for omega, amplitude and phase
        self.wave_xyz[:, :, :, 1] *= deviation[None, :, None]
        self.wave_amp = np.random.rand(N, 3) * np.array([0.1, 1., 1.])

        # compute amp threshold
        S = 1000
        amps = self.cos_function(self.wave_amp, np.arange(0, S))
        self.amp_threshold = np.percentile(amps, (1 - randprob) * 100)
        self.amp_scale = 1. / (np.max(amps) - self.amp_threshold + 1e-10)
        # print(np.sum((amps - threshold) / amp_scale > 0.)) # 0.2 * S
    
    def random_not_correct_action(self, curr_action: FoldPolicyAction, prev_action: FoldPolicyAction):
        assert curr_action.is_correct_action
    
        amp = np.clip((float(self.cos_function(self.wave_amp, self.curr_step)) - self.amp_threshold) * self.amp_scale, 0., 1.)
        if self.cfg.not_correct_action_prob == 0.:
            amp = 0. # force zero, prevent future bug (e.g. step out of S may result non-zero amp)
        is_correct_action = (amp == 0.)
        logger.info(f"force_correct_stage:{self.force_correct_stage}, current_stage:{self.policy._current_stage} amp:{amp}")

        if is_correct_action:
            logger.info("case 1: return if is correct action")
            return curr_action
        
        if self.policy._current_stage == self.force_correct_stage:
            logger.info("case 2: avoid wrong action at the next stage after grasp failure")
            return curr_action
            
        if curr_action.xyz_l is None or curr_action.xyz_r is None:
            logger.info("case 3: not correct open/close action, use previous action state")
            assert curr_action.xyz_l is None and curr_action.xyz_r is None, "action.xyz_l and action.xyz_r must be both None or both not None"
            curr_action.is_correct_action = prev_action.is_correct_action
            return curr_action
        
        logger.info("case 4: not correct xyz action, add noise")
        curr_action.is_correct_action = False
        d_xyz_l = amp * np.array([float(self.cos_function(self.wave_xyz[0, i], self.curr_step)) for i in range(3)])
        d_xyz_r = amp * np.array([float(self.cos_function(self.wave_xyz[1, i], self.curr_step)) for i in range(3)])
        deviation = self.cfg.not_correct_action_xyz_std
        d_xyz_l = np.max([np.min([d_xyz_l, +deviation], axis=0), -deviation], axis=0)
        d_xyz_r = np.max([np.min([d_xyz_r, +deviation], axis=0), -deviation], axis=0)
        logger.info(f"d_xyz_l:{d_xyz_l} d_xyz_r:{d_xyz_r}")
        curr_action.xyz_l += d_xyz_l
        curr_action.xyz_r += d_xyz_r
        return curr_action
    
    def detect_grasp_fail(self):
        is_grasp_fail = self.env.is_grasp_fail()
        if is_grasp_fail["left"] or is_grasp_fail["right"]:
            logger.info(f"grasp fail detected, append some actions with random length and back to previous stage, current stage:{self.policy._current_stage}")
            if self.policy._current_stage in self.stage_detect_grasp_fail:
                logger.info("current_stage has already grasp fail, skip to avoid endless loop")
                return
            self.stage_detect_grasp_fail.add(self.policy._current_stage)
            
            action_list = list(self.policy._action_deque)
            self.force_correct_stage = self.policy._current_stage
            self.policy._action_deque.clear()
            self.policy._current_stage -= 1

            grasp_nothing_step = np.random.randint(*self.cfg.grasp_nothing_step_range)
            logger.info(f"grasp_nothing_step:{grasp_nothing_step}")
            for action_step in range(grasp_nothing_step - 1): # curr_action is already there, so -1
                if action_list[action_step].xyz_l is None or action_list[action_step].xyz_r is None:
                    break # (action with xyz_l=None or xyz_r is None) should be in action_list, so won't OutOfRangeError
                action = action_list[action_step]
                self.policy._append_action(action)
            self.policy._append_action(FoldPolicyAction(picker_l=self.env.PICKER_OPEN, picker_r=self.env.PICKER_OPEN))


class FoldStatePolicy(FoldPolicy):
    def __init__(self, cfg: FoldStatePolicyCfg, env: FoldEnv):
        super().__init__(cfg, env)
        self._cfg: FoldStatePolicyCfg

        self._current_stage: int = 0
        self._action_deque: deque[Optional[FoldPolicyAction]] = deque()
        self._policy_cache: dict[str, Any] = {}
        self._keypoint_names_for_shape_match = None
        if self._cfg.enable_not_correct_action:
            self._not_correct_action_module = NotCorrectActionModule(cfg, env, self)
    
    ### helper functions ###
    def _shape_match(self, rest_mesh: trimesh.Trimesh, curr_mesh: trimesh.Trimesh, kpid: dict):
        keypoint_names = self._keypoint_names_for_shape_match
        assert keypoint_names is not None, "keypoint_names_for_shape_match must be set"

        rot, trans, err = policy_utils.shape_match_xy(
            np.array([rest_mesh.vertices[kpid[name]][:2] for name in keypoint_names]),
            np.array([curr_mesh.vertices[kpid[name]][:2] for name in keypoint_names]),
        )
        rot_inv, trans_inv, err_inv = policy_utils.shape_match_xy(
            np.array([rest_mesh.vertices[kpid[name]][:2] for name in reversed(keypoint_names)]),
            np.array([curr_mesh.vertices[kpid[name]][:2] for name in keypoint_names]),
        )
        
        is_faceup = (err < err_inv)
        theta = policy_utils.theta_from_2d_rotation_matrix(rot if is_faceup else rot_inv)
        return is_faceup, theta
    
    def _local_coord(self, vec2: np.ndarray):
        ex = vec2 / max(self._env.eps, np.linalg.norm(vec2))
        ey = np.array([-ex[1], ex[0]])
        return ex, ey

    def _move_away_l(self, xyz: np.ndarray):
        return xyz + self._cfg.move_away_l * np.array([1., -np.sign(xyz[1]), 1.])

    def _move_away_r(self, xyz: np.ndarray):
        return xyz + self._cfg.move_away_r * np.array([1., -np.sign(xyz[1]), 1.])

    ### add action implementation ###
    def _append_action(self, action: Optional[FoldPolicyAction]):
        self._action_deque.append(self._process_action(action))
    
    def _interp_action_and_put_in_deque(
        self, 
        xyz_l_1: np.ndarray, xyz_l_2: np.ndarray, xyz_l_3: np.ndarray,
        xyz_r_1: np.ndarray, xyz_r_2: np.ndarray, xyz_r_3: np.ndarray,
    ):  
        # setup
        env, cfg = self._env, self._cfg
        tcp_curr = env.get_tcp_xyz()
        xyz_l_0 = tcp_curr["left"]
        xyz_r_0 = tcp_curr["right"]

        # adaptively set the step length
        ssl = cfg.single_step_length
        step_1 = 1 + int(max(
            np.linalg.norm(xyz_l_1 - xyz_l_0) / ssl,
            np.linalg.norm(xyz_r_1 - xyz_r_0) / ssl
        ))
        for xyz_l, xyz_r in zip(
                policy_utils.interpolate_bezier(np.array([xyz_l_0, xyz_l_1]), step_1),
                policy_utils.interpolate_bezier(np.array([xyz_r_0, xyz_r_1]), step_1),
            ):
            self._append_action(FoldPolicyAction(xyz_l=xyz_l, xyz_r=xyz_r))
        self._append_action(FoldPolicyAction(picker_l=env.PICKER_CLOSE, picker_r=env.PICKER_CLOSE))

        step_2 = 1 + int(max(
            (np.linalg.norm(xyz_l_3 - xyz_l_2) + np.linalg.norm(xyz_l_2 - xyz_l_1)) / ssl,
            (np.linalg.norm(xyz_r_3 - xyz_r_2) + np.linalg.norm(xyz_r_2 - xyz_r_1)) / ssl
        ))
        for xyz_l, xyz_r in zip(
            policy_utils.interpolate_bezier(np.array([xyz_l_1, xyz_l_2, xyz_l_3]), step_2),
            policy_utils.interpolate_bezier(np.array([xyz_r_1, xyz_r_2, xyz_r_3]), step_2),
        ):
            self._append_action(FoldPolicyAction(xyz_l=xyz_l, xyz_r=xyz_r))
        self._append_action(FoldPolicyAction(picker_l=env.PICKER_OPEN, picker_r=env.PICKER_OPEN))
        
        xyz_l_4, xyz_r_4 = self._move_away_l(xyz_l_3), self._move_away_r(xyz_r_3)
        step_3 = 1 + int(max(
            np.linalg.norm(xyz_l_4 - xyz_l_3) / ssl,
            np.linalg.norm(xyz_r_4 - xyz_r_3) / ssl
        ))
        for xyz_l, xyz_r in zip(
            policy_utils.interpolate_bezier(np.array([xyz_l_3, xyz_l_4]), step_3),
            policy_utils.interpolate_bezier(np.array([xyz_r_3, xyz_r_4]), step_3),
        ):
            self._append_action(FoldPolicyAction(xyz_l=xyz_l, xyz_r=xyz_r))
        
    def _add_action_pre_policy(self):
        tcp_curr = self._env.get_tcp_xyz()
        logger.info(f"add action pre policy, Current TCP position: {pprint.pformat(tcp_curr)}")
        self._append_action(self._get_pre_action())

    def _compute_new_action(self):
        raise NotImplementedError("This function should be implemented by subclass")
    
    def get_all_possible_rot_z_flip_y(self):
        return [(0, False), (180, False), (0, True), (180, True)]
    
    def get_action(self) -> Optional[FoldPolicyAction]:
        while len(self._action_deque) == 0:
            self._compute_new_action()
        action = self._action_deque.popleft()

        if self._cfg.enable_not_correct_action:
            self._not_correct_action_module.set_current_action(action)
            action = self._not_correct_action_module.get_current_action()

        return action

    @property
    def skip_rotate(self):
        return self._cfg.skip_rotate

    @skip_rotate.setter
    def skip_rotate(self, value: bool):
        self._cfg.skip_rotate = bool(value)
    
    def reset(self):
        super().reset()
        self._current_stage = 0
        self._action_deque.clear()
        self._policy_cache.clear()
        if self._cfg.enable_not_correct_action:
            self._not_correct_action_module.reset()