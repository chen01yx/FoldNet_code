import logging
logger = logging.getLogger(__name__)

from dataclasses import dataclass
from collections import deque
import os

import numpy as np

from garmentds.foldenv.fold_env import FoldEnv, Picker
from garmentds.foldenv.policy.base import FoldPolicyAction
from garmentds.foldenv.policy.visual import FoldVisualPolicy, FoldVisualPolicyCfg
import garmentds.foldenv.policy_utils as policy_utils
from garmentds.foldenv.policy.state.base import PICKER_Z
import garmentds.common.utils as utils


@dataclass
class FoldHybridPolicyCfg(FoldVisualPolicyCfg):
    grasp_nothing_step_range: tuple[int, int] = (2, 5) # [2, 3, 4]
    single_step_length: float = 0.03
    z_grasp: float = PICKER_Z


def hybrid_modify_is_correct_action(action_dict_list: list[dict]):
    """if one action is fail, then some previous action also fail"""
    fail_idx_list = []
    for action_idx, action in enumerate(action_dict_list):
        if not action["is_correct_action"]:
            fail_idx_list.append(action_idx)
    for fail_idx in fail_idx_list:
        for i in range(fail_idx - 1, -1, -1):
            if action_dict_list[i]["picker_l"] == -1. or action_dict_list[i]["picker_r"] == -1.:
                break
            action_dict_list[i]["is_correct_action"] = False
            logger.info(f"modify {i} th action because of {fail_idx} failed")
    return action_dict_list


def hybrid_action_post_process(action_dir: str):
    action_files = os.listdir(action_dir)
    action_files = sorted(action_files, key=lambda x: int(x.split(".")[0]))
    action_dict_list = []
    for action_file in action_files:
        action_dict_list.append(utils.load_json(os.path.join(action_dir, action_file)))
    
    action_dict_list = hybrid_modify_is_correct_action(action_dict_list)
    for action_file, action_dict in zip(action_files, action_dict_list):
        utils.dump_json(os.path.join(action_dir, action_file), action_dict)


class FoldHybridPolicy(FoldVisualPolicy):
    RECOVER_STATUS_NONE = 0
    RECOVER_STATUS_MOVE = 1
    def __init__(self, cfg: FoldHybridPolicyCfg, env: FoldEnv):
        super().__init__(cfg, env)
        self._cfg: FoldHybridPolicyCfg
        self._grasp_fail_info_list: list[dict[str, bool]] = []
        self._recover_action_deque: deque[FoldPolicyAction] = deque()
        self._reset()
    
    def _reset(self):
        self._grasp_fail_info_list.clear()
        self._recover_action_deque.clear()
        self._recover_status: int = self.RECOVER_STATUS_NONE
        self._grasp_nothing_step = np.random.randint(*self._cfg.grasp_nothing_step_range)
        logger.info(f"grasp_nothing_step:{self._grasp_nothing_step}")

    def get_possible_grasp_xyz(self) -> tuple[list[np.ndarray], list[str]]:
        raise NotImplementedError("This function should be implemented in subclass")
    
    def compute_recover_target_xyz(self) -> tuple[int, int]:
        possible_grasp_xyz, grasp_name = self.get_possible_grasp_xyz()
        tcp_xyz = self._env.get_tcp_xyz()
        xyz_l_0, xyz_r_0 = tcp_xyz["left"], tcp_xyz["right"]
        # find the closest possible keypoints
        idx_l = np.argmin(np.linalg.norm(xyz_l_0 - possible_grasp_xyz, axis=-1))
        xyz_l_1 = possible_grasp_xyz[idx_l]
        idx_r = np.argmin(np.linalg.norm(xyz_r_0 - possible_grasp_xyz, axis=-1))
        xyz_r_1 = possible_grasp_xyz[idx_r]
        logger.info(f"compute recover target xyz, left:{grasp_name[idx_l]}, right:{grasp_name[idx_r]}")
        xyz_l_1[2] = xyz_r_1[2] = self._cfg.z_grasp
        return idx_l, idx_r
    
    def get_grasp_fail_info(self):
        is_grasp_fail = self._env.is_grasp_fail()
        recover_target = self.compute_recover_target_xyz()
        return dict(is_grasp_fail=is_grasp_fail, recover_target=recover_target)
    
    def get_action(self):
        logger.info(f"current recover status:{self._recover_status}")
        # use policy network to compute action
        action = super().get_action()
        if action is None:
            return action
        
        if self._recover_status == self.RECOVER_STATUS_NONE:
            # compute action is correct or not
            # TODO: FIXME will_grasp_fail is different from is_grasp_fail
            will_grasp_fail = self._env.will_grasp_fail(action.xyz_l, action.xyz_r, action.picker_l, action.picker_r)
            action.is_correct_action = (not will_grasp_fail["left"]) and (not will_grasp_fail["right"])
            logger.info(f"hybrid get action, will_grasp_fail: {will_grasp_fail}")
            logger.info(f"hybrid get action, action: {action}")
 
            # if grasp fail detected, use heuristics to recover
            # let grasp_nothing_step = 1, the workflow would be like:
            # (get_action=close) -> (grasp_fail=false)[-3] -> 
            # (get_action=move) -> (grasp_fail=true)[-2] -> 
            # (get_action=move) -> (grasp_fail=false)[-1] -> (need_start_recover=true) -> (use open action instead)
            self._grasp_fail_info_list.append(self.get_grasp_fail_info())
            target_idx = self._grasp_nothing_step + 1
            need_recover = (len(self._grasp_fail_info_list) >= target_idx) and (
                self._grasp_fail_info_list[-target_idx]["is_grasp_fail"]["left"] or
                self._grasp_fail_info_list[-target_idx]["is_grasp_fail"]["right"]
            )

            if need_recover:
                logger.info(f"grasp fail detected, append some actions with random length and pick closest keypoint.")
                self._recover_status = self.RECOVER_STATUS_MOVE
                self._recover_action_deque.clear()

                tcp_xyz = self._env.get_tcp_xyz()
                action = FoldPolicyAction(
                    xyz_l=tcp_xyz["left"], xyz_r=tcp_xyz["right"],
                    picker_l=self._env.PICKER_OPEN, picker_r=self._env.PICKER_OPEN,
                    is_correct_action=True, 
                )
            
            return action
        
        elif self._recover_status == self.RECOVER_STATUS_MOVE:
            if len(self._recover_action_deque) == 0:
                tcp_xyz = self._env.get_tcp_xyz()
                xyz_l_0, xyz_r_0 = tcp_xyz["left"], tcp_xyz["right"]

                target_idx = self._grasp_nothing_step + 1
                idx_l, idx_r = self._grasp_fail_info_list[-target_idx]["recover_target"]
                possible_grasp_xyz = self.get_possible_grasp_xyz()[0]
                xyz_l_1, xyz_r_1 = possible_grasp_xyz[idx_l], possible_grasp_xyz[idx_r]

                ssl = self._cfg.single_step_length
                step_1 = 1 + int(max(
                    np.linalg.norm(xyz_l_1 - xyz_l_0) / ssl,
                    np.linalg.norm(xyz_r_1 - xyz_r_0) / ssl
                ))
                for xyz_l, xyz_r in zip(
                    policy_utils.interpolate_bezier(np.array([xyz_l_0, xyz_l_1]), step_1),
                    policy_utils.interpolate_bezier(np.array([xyz_r_0, xyz_r_1]), step_1),
                ):
                    self._recover_action_deque.append(self._process_action(FoldPolicyAction(
                        xyz_l=xyz_l, xyz_r=xyz_r,
                        picker_l=self._env.PICKER_OPEN, picker_r=self._env.PICKER_OPEN, 
                        is_correct_action=True, 
                    )))
                self._recover_action_deque.append(FoldPolicyAction(
                    xyz_l=xyz_l_1, xyz_r=xyz_r_1,
                    picker_l=self._env.PICKER_CLOSE, picker_r=self._env.PICKER_CLOSE, 
                    is_correct_action=True, 
                ))
            
            action = self._recover_action_deque.popleft()
            if len(self._recover_action_deque) == 0:
                self._recover_status = self.RECOVER_STATUS_NONE
                self._policy_cache["action"].clear()
            
            return action
        
        else:
            raise ValueError(self._recover_status)
    
    def reset(self):
        super().reset()
        self._reset()


class FoldHybridTShirtPolicy(FoldHybridPolicy):
    def get_possible_grasp_xyz(self):
        curr_mesh = self._env.get_raw_mesh_curr()
        curr_vert = curr_mesh.vertices
        kpid = self._env.get_keypoint_idx()
        return np.array([
            curr_vert[kpid["l_shoulder"]],
            curr_vert[kpid["r_shoulder"]],
            curr_vert[kpid["l_corner"]],
            curr_vert[kpid["r_corner"]],
            curr_vert[kpid["l_sleeve_bottom"]],
            curr_vert[kpid["r_sleeve_bottom"]],
            curr_vert[kpid["l_sleeve_top"]],
            curr_vert[kpid["r_sleeve_top"]],
            (curr_vert[kpid["l_sleeve_bottom"]] + curr_vert[kpid["l_sleeve_top"]]) / 2,
            (curr_vert[kpid["r_sleeve_bottom"]] + curr_vert[kpid["r_sleeve_top"]]) / 2,
        ]), [
            "l_shoulder",
            "r_shoulder",
            "l_corner",
            "r_corner",
            "l_sleeve_bottom",
            "r_sleeve_bottom",
            "l_sleeve_top",
            "r_sleeve_top",
            "l_sleeve_ctr",
            "r_sleeve_ctr",
        ]


class FoldHybridTrousersPolicy(FoldHybridPolicy):
    def get_possible_grasp_xyz(self):
        curr_mesh = self._env.get_raw_mesh_curr()
        curr_vert = curr_mesh.vertices
        kpid = self._env.get_keypoint_idx()
        return np.array([
            curr_vert[kpid["l_corner"]],
            curr_vert[kpid["r_corner"]],
            curr_vert[kpid["l_leg_o"]],
            curr_vert[kpid["r_leg_o"]],
            (curr_vert[kpid["l_leg_o"]] + curr_vert[kpid["l_leg_i"]]) / 2,
            (curr_vert[kpid["r_leg_o"]] + curr_vert[kpid["r_leg_i"]]) / 2,
            (curr_vert[kpid["top_ctr_f"]] + curr_vert[kpid["top_ctr_b"]]) / 2,
        ]), [
            "l_corner",
            "r_corner",
            "l_leg_o",
            "r_leg_o",
            "l_leg_c",
            "r_leg_c",
            "top_ctr",
        ]