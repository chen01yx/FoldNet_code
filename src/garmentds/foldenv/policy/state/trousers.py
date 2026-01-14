import logging
logger = logging.getLogger(__name__)

from dataclasses import dataclass, asdict, field
from collections import deque
from typing import Optional, Any
import pprint

import numpy as np
import trimesh

from garmentds.foldenv.fold_env import FoldEnv
import garmentds.foldenv.policy_utils as policy_utils
from garmentds.foldenv.policy.state.base import FoldStatePolicyCfg, FoldStatePolicy, GRASP_TH, PICKER_Z


@dataclass
class FoldStateTrousersPolicyCfg(FoldStatePolicyCfg):
    rotate_z_grasp: float = PICKER_Z
    rotate_z_move: float = PICKER_Z
    rotate_z_put: float = PICKER_Z

    fold1_z_grasp: float = PICKER_Z
    fold1_z_move: float = PICKER_Z + 0.15
    fold1_z_put: float = PICKER_Z + 0.04

    drag_z_grasp: float = PICKER_Z
    drag_z_move: float = PICKER_Z
    drag_z_put: float = PICKER_Z

    fold2_z_grasp: float = PICKER_Z
    fold2_z_move: float = PICKER_Z + 0.25
    fold2_z_put: float = PICKER_Z + 0.06
    fold2_x_offset: float = GRASP_TH * 3 # to avoid over folding

    skip_threshold: float = 0.04

    def _scale(self):
        logger.info(f"before scale: {self}")
        S = self.cloth_scale

        for attr_name in [
            "rotate_z_grasp", "rotate_z_move", "rotate_z_put",
            "fold1_z_grasp", "fold1_z_move", "fold1_z_put",
            "drag_z_grasp", "drag_z_move", "drag_z_put",
            "fold2_z_grasp", "fold2_z_move", "fold2_z_put",
        ]:
            new_val = (getattr(self, attr_name) - PICKER_Z) * S + PICKER_Z
            setattr(self, attr_name, new_val)
        logger.info(f"after scale: {self}")


class FoldStateTrousersPolicy(FoldStatePolicy):
    def __init__(self, cfg: FoldStateTrousersPolicyCfg, env: FoldEnv):
        super().__init__(cfg, env)
        self._cfg: FoldStateTrousersPolicyCfg
        self._keypoint_names_for_shape_match = ["l_corner", "l_leg_o", "r_leg_o", "r_corner"]

    def get_all_possible_rot_z_flip_y(self):
        return [(+90, True), (-90, True), (+90, False), (-90, False)]

    ### add action implementation ###
    def _add_action_rotate(self):
        env, cfg = self._env, self._cfg
        rest_mesh = env.get_raw_mesh_rest()
        rest_vert = rest_mesh.vertices
        curr_mesh = env.get_raw_mesh_curr()
        curr_vert = curr_mesh.vertices
        kpid = env.get_keypoint_idx()

        is_faceup, theta = self._shape_match(rest_mesh, curr_mesh, kpid)
        self._policy_cache["shape_match_for_rotation"] = (is_faceup, theta)
        logger.info(f"Shape match result: is_faceup={is_faceup}, theta={np.rad2deg(theta)}")

        l_cn, r_cn, l_lo, r_lo = "l_corner", "r_corner", "l_leg_o", "r_leg_o"
        if not is_faceup:
            l_cn, r_cn, l_lo, r_lo = r_cn, l_cn, r_lo, l_lo
        
        # if is_faceup:
        #     theta = 0 -> rot_angle = 45
        #     theta = +90 -> rot_angle = 90
        #     theta = +180 -> rot_angle = 135
        if theta > 0:
            xyl_1 = curr_vert[kpid[l_cn]][:2]
            xyr_1 = curr_vert[kpid[l_lo]][:2]
            rot = policy_utils.get_2d_rotation_matrix(+np.pi / 2 + (abs(theta) - np.pi / 2) * (1 / 3))
            xyl_3 = rest_vert[kpid["l_corner"]][:2] @ rot.T
            xyr_3 = rest_vert[kpid["l_leg_o"]][:2] @ rot.T
        else:
            xyl_1 = curr_vert[kpid[r_lo]][:2]
            xyr_1 = curr_vert[kpid[r_cn]][:2]
            rot = policy_utils.get_2d_rotation_matrix(-np.pi / 2 - (abs(theta) - np.pi / 2) * (1 / 3))
            xyl_3 = rest_vert[kpid["r_leg_o"]][:2] @ rot.T
            xyr_3 = rest_vert[kpid["r_corner"]][:2] @ rot.T
        xyl_3 += np.array([0., -0.1])
        xyr_3 += np.array([0., -0.1])
        
        xyz_l_1 = np.array([*xyl_1, cfg.rotate_z_grasp])
        xyz_r_1 = np.array([*xyr_1, cfg.rotate_z_grasp])
        xyz_l_2 = np.array([*((xyl_1 + xyl_3) / 2), cfg.rotate_z_move])
        xyz_r_2 = np.array([*((xyr_1 + xyr_3) / 2), cfg.rotate_z_move])
        xyz_l_3 = np.array([*xyl_3, cfg.rotate_z_put])
        xyz_r_3 = np.array([*xyr_3, cfg.rotate_z_put])

        move_dist_l = np.linalg.norm(xyl_1 - xyl_3)
        move_dist_r = np.linalg.norm(xyr_1 - xyr_3)
        logger.info(f"rotate move_dist_l={move_dist_l}, move_dist_r={move_dist_r}")
        if max(move_dist_l, move_dist_r) <= cfg.skip_threshold:
            return
        self._interp_action_and_put_in_deque(
            xyz_l_1, xyz_l_2, xyz_l_3,
            xyz_r_1, xyz_r_2, xyz_r_3,
        )
        
    def _add_action_fold1(self):
        env, cfg = self._env, self._cfg
        rest_mesh = env.get_raw_mesh_rest()
        rest_vert = rest_mesh.vertices
        curr_mesh = env.get_raw_mesh_curr()
        curr_vert = curr_mesh.vertices
        kpid = env.get_keypoint_idx()

        is_faceup, theta = self._policy_cache["shape_match_for_rotation"]

        l_cn, r_cn, l_lo, r_lo = "l_corner", "r_corner", "l_leg_o", "r_leg_o"
        if not is_faceup:
            l_cn, r_cn, l_lo, r_lo = r_cn, l_cn, r_lo, l_lo
        
        if theta > 0:
            xyl_1 = curr_vert[kpid[r_cn]][:2]
            xyr_1 = curr_vert[kpid[r_lo]][:2]
            xyl_3 = curr_vert[kpid[l_cn]][:2]
            xyr_3 = curr_vert[kpid[l_lo]][:2]
        else:
            xyl_1 = curr_vert[kpid[l_lo]][:2]
            xyr_1 = curr_vert[kpid[l_cn]][:2]
            xyl_3 = curr_vert[kpid[r_lo]][:2]
            xyr_3 = curr_vert[kpid[r_cn]][:2]
        
        xyz_l_1 = np.array([*xyl_1, cfg.fold1_z_grasp])
        xyz_r_1 = np.array([*xyr_1, cfg.fold1_z_grasp])
        xyz_l_2 = np.array([*((xyl_1 + xyl_3) / 2), cfg.fold1_z_move])
        xyz_r_2 = np.array([*((xyr_1 + xyr_3) / 2), cfg.fold1_z_move])
        xyz_l_3 = np.array([*xyl_3, cfg.fold1_z_put])
        xyz_r_3 = np.array([*xyr_3, cfg.fold1_z_put])

        self._interp_action_and_put_in_deque(
            xyz_l_1, xyz_l_2, xyz_l_3,
            xyz_r_1, xyz_r_2, xyz_r_3,
        )

    def _add_action_drag_to_center(self):
        env, cfg = self._env, self._cfg
        rest_mesh = env.get_raw_mesh_rest()
        rest_vert = rest_mesh.vertices
        curr_mesh = env.get_raw_mesh_curr()
        curr_vert = curr_mesh.vertices
        kpid = env.get_keypoint_idx()

        is_faceup, theta = self._policy_cache["shape_match_for_rotation"]
        
        if theta > 0:
            xyl_1 = (curr_vert[kpid["top_ctr_f"]] + curr_vert[kpid["top_ctr_b"]])[:2] / 2
            xyr_1 = (curr_vert[kpid["l_leg_i"]] + curr_vert[kpid["r_leg_i"]])[:2] / 2
            length = np.linalg.norm(xyl_1[:2] - xyr_1[:2])
            xyl_3 = np.array([-length / 2, -0.05])
            xyr_3 = np.array([+length / 2, -0.05])
        else:
            xyl_1 = (curr_vert[kpid["l_leg_i"]] + curr_vert[kpid["r_leg_i"]])[:2] / 2
            xyr_1 = (curr_vert[kpid["top_ctr_f"]] + curr_vert[kpid["top_ctr_b"]])[:2] / 2
            length = np.linalg.norm(xyl_1[:2] - xyr_1[:2])
            xyl_3 = np.array([-length / 2, -0.05])
            xyr_3 = np.array([+length / 2, -0.05])
        
        xyz_l_1 = np.array([*xyl_1, cfg.drag_z_grasp])
        xyz_r_1 = np.array([*xyr_1, cfg.drag_z_grasp])
        xyz_l_2 = np.array([*((xyl_1 + xyl_3) / 2), cfg.drag_z_move])
        xyz_r_2 = np.array([*((xyr_1 + xyr_3) / 2), cfg.drag_z_move])
        xyz_l_3 = np.array([*xyl_3, cfg.drag_z_put])
        xyz_r_3 = np.array([*xyr_3, cfg.drag_z_put])

        move_dist_l = np.linalg.norm(xyl_1 - xyl_3)
        move_dist_r = np.linalg.norm(xyr_1 - xyr_3)
        logger.info(f"drag move_dist_l={move_dist_l}, move_dist_r={move_dist_r}")
        if max(move_dist_l, move_dist_r) <= cfg.skip_threshold:
            return
        self._interp_action_and_put_in_deque(
            xyz_l_1, xyz_l_2, xyz_l_3,
            xyz_r_1, xyz_r_2, xyz_r_3,
        )

    def _add_action_fold2(self):
        env, cfg = self._env, self._cfg
        rest_mesh = env.get_raw_mesh_rest()
        rest_vert = rest_mesh.vertices
        curr_mesh = env.get_raw_mesh_curr()
        curr_vert = curr_mesh.vertices
        kpid = env.get_keypoint_idx()

        is_faceup, theta = self._policy_cache["shape_match_for_rotation"]

        if theta > 0:
            xyl_1 = (curr_vert[kpid["top_ctr_f"]] + curr_vert[kpid["top_ctr_b"]])[:2] / 2
            xyr_1 = (curr_vert[kpid["l_corner"]] + curr_vert[kpid["r_corner"]])[:2] / 2
        else:
            xyl_1 = (curr_vert[kpid["l_corner"]] + curr_vert[kpid["r_corner"]])[:2] / 2
            xyr_1 = (curr_vert[kpid["top_ctr_f"]] + curr_vert[kpid["top_ctr_b"]])[:2] / 2
        
        width = np.linalg.norm(xyl_1[:2] - xyr_1[:2])
        xyc_3 = (
            curr_vert[kpid["l_leg_i"]] + curr_vert[kpid["r_leg_i"]] +
            curr_vert[kpid["l_leg_o"]] + curr_vert[kpid["r_leg_o"]]
        )[:2] / 4
        
        if theta > 0:
            xyl_3 = xyc_3 + np.array([-cfg.fold2_x_offset, +width / 2])
            xyr_3 = xyc_3 + np.array([-cfg.fold2_x_offset, -width / 2])
        else:
            xyl_3 = xyc_3 + np.array([+cfg.fold2_x_offset, -width / 2])
            xyr_3 = xyc_3 + np.array([+cfg.fold2_x_offset, +width / 2])
        
        xyz_l_1 = np.array([*xyl_1, cfg.fold2_z_grasp])
        xyz_r_1 = np.array([*xyr_1, cfg.fold2_z_grasp])
        xyz_l_2 = np.array([*((xyl_1 + xyl_3) / 2), cfg.fold2_z_move])
        xyz_r_2 = np.array([*((xyr_1 + xyr_3) / 2), cfg.fold2_z_move])
        xyz_l_3 = np.array([*xyl_3, cfg.fold2_z_put])
        xyz_r_3 = np.array([*xyr_3, cfg.fold2_z_put])

        self._interp_action_and_put_in_deque(
            xyz_l_1, xyz_l_2, xyz_l_3,
            xyz_r_1, xyz_r_2, xyz_r_3,
        )
        
    ### high-level policy api ###
    def _compute_new_action(self):
        if self._current_stage == 0:
            self._add_action_pre_policy()
        elif self._current_stage == 1:
            self._add_action_rotate()
        elif self._current_stage == 2:
            self._add_action_fold1()
        elif self._current_stage == 3:
            self._add_action_drag_to_center()
        elif self._current_stage == 4:
            self._add_action_fold2()
        else:
            self._append_action(None)

        self._current_stage += 1