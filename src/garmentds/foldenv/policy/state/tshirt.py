import logging
logger = logging.getLogger(__name__)

from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Any
import pprint

import numpy as np
import trimesh

from garmentds.foldenv.fold_env import FoldEnv
import garmentds.foldenv.policy_utils as policy_utils
from garmentds.foldenv.policy.state.base import FoldStatePolicyCfg, FoldStatePolicy, GRASP_TH, PICKER_Z


@dataclass
class FoldStateTShirtPolicyCfg(FoldStatePolicyCfg):
    rotate_z_grasp: float = PICKER_Z
    rotate_z_move: float = PICKER_Z
    rotate_z_put: float = PICKER_Z
    rotate_theta_ratio: float = 2 / 3
    rotate_dist_relax: float = 0.95

    align_z_grasp: float = PICKER_Z
    align_z_move: float = PICKER_Z
    align_z_put: float = PICKER_Z
    align_y_offset: float = 0.02 # slightly stretch the garment
    align_angle_delta_max1: float = np.deg2rad(35)
    align_angle_delta_max2: float = np.deg2rad(65)
    align_skip_threshold: float = 0.04 # if the target position is too close to the current position, skip the alignment

    fold1_z_grasp: float = PICKER_Z
    fold1_z_move: float = PICKER_Z + 0.15
    fold1_z_put: float = PICKER_Z + 0.04
    fold1_z_move_proportional_to_dist: float = 0.1
    fold1_x_max_barrier: float = 0.04 # avoid gripper collision

    fold2_z_grasp: float = PICKER_Z
    fold2_z_move: float = PICKER_Z + 0.20
    fold2_z_put: float = PICKER_Z + 0.04
    fold2_y_offset: float = GRASP_TH / 2 # to avoid over folding
    fold2_x_relax: float = 0.95
    fold2_sleeve_fold_proj_coeff_th: float = 0.5
    
    def _scale(self):
        logger.info(f"before scale: {self}")
        S = self.cloth_scale

        for attr_name in [
            "rotate_z_grasp", "rotate_z_move", "rotate_z_put",
            "align_z_grasp", "align_z_move", "align_z_put",
            "fold1_z_grasp", "fold1_z_move", "fold1_z_put",
            "fold2_z_grasp", "fold2_z_move", "fold2_z_put",
        ]:
            new_val = (getattr(self, attr_name) - PICKER_Z) * S + PICKER_Z
            setattr(self, attr_name, new_val)
        logger.info(f"after scale: {self}")


class FoldStateTShirtPolicy(FoldStatePolicy):
    def __init__(self, cfg: FoldStateTShirtPolicyCfg, env: FoldEnv):
        super().__init__(cfg, env)
        self._cfg: FoldStateTShirtPolicyCfg
        self._keypoint_names_for_shape_match = ["l_shoulder", "l_corner", "r_corner", "r_shoulder"]

    ### add action implementation ###
    def _add_action_rotate_garment(self, rotate_step: int):
        env, cfg = self._env, self._cfg
        rest_mesh = env.get_raw_mesh_rest()
        rest_vert = rest_mesh.vertices
        curr_mesh = env.get_raw_mesh_curr()
        curr_vert = curr_mesh.vertices
        kpid = env.get_keypoint_idx()
        
        # rotate garment using more suitable matching
        if rotate_step == 0:
            is_faceup, theta = self._shape_match(rest_mesh, curr_mesh, kpid)
            self._policy_cache["shape_match_for_rotation"] = (is_faceup, theta)
            logger.info(f"Shape match result: is_faceup={is_faceup}, theta={np.rad2deg(theta)}")

            def interp(theta: float, theta_target: float):
                return theta_target * cfg.rotate_theta_ratio + theta * (1. - cfg.rotate_theta_ratio)

            if abs(theta) < np.pi / 2:
                if is_faceup:
                    xyl_start = curr_vert[kpid["l_corner"]][:2]
                    xyr_start = curr_vert[kpid["r_corner"]][:2]
                else:
                    xyl_start = curr_vert[kpid["r_corner"]][:2]
                    xyr_start = curr_vert[kpid["l_corner"]][:2]
                rot = policy_utils.get_2d_rotation_matrix(interp(theta, 0.))
                xyl_end = rest_vert[kpid["l_corner"]][:2] @ rot.T
                xyr_end = rest_vert[kpid["r_corner"]][:2] @ rot.T
            else:
                if is_faceup:
                    xyl_start = curr_vert[kpid["r_shoulder"]][:2]
                    xyr_start = curr_vert[kpid["l_shoulder"]][:2]
                else:
                    xyl_start = curr_vert[kpid["l_shoulder"]][:2]
                    xyr_start = curr_vert[kpid["r_shoulder"]][:2]
                rot = policy_utils.get_2d_rotation_matrix(interp(abs(theta), np.pi) * np.sign(theta))
                xyl_end = rest_vert[kpid["r_shoulder"]][:2] @ rot.T
                xyr_end = rest_vert[kpid["l_shoulder"]][:2] @ rot.T
            
            center, diff = (xyl_end + xyr_end) / 2, (xyl_end - xyr_end) / 2
            xyl_end = center + diff * cfg.rotate_dist_relax
            xyr_end = center - diff * cfg.rotate_dist_relax

            xyl_end = xyl_end + np.array([0., -0.1])
            xyr_end = xyr_end + np.array([0., -0.1])
        
        elif rotate_step == 1:
            shape_match_for_rotation: tuple[bool, float] = self._policy_cache["shape_match_for_rotation"]
            (is_faceup, theta) = shape_match_for_rotation

            if abs(theta) < np.pi / 2:
                if is_faceup:
                    xyl_start = curr_vert[kpid["l_shoulder"]][:2]
                    xyr_start = curr_vert[kpid["r_shoulder"]][:2]
                else:
                    xyl_start = curr_vert[kpid["r_shoulder"]][:2]
                    xyr_start = curr_vert[kpid["l_shoulder"]][:2]
                xyl_end = rest_vert[kpid["l_shoulder"]][:2]
                xyr_end = rest_vert[kpid["r_shoulder"]][:2]
            else:
                if is_faceup:
                    xyl_start = curr_vert[kpid["r_corner"]][:2]
                    xyr_start = curr_vert[kpid["l_corner"]][:2]
                else:
                    xyl_start = curr_vert[kpid["l_corner"]][:2]
                    xyr_start = curr_vert[kpid["r_corner"]][:2]
                rot = policy_utils.get_2d_rotation_matrix(np.pi)
                xyl_end = rest_vert[kpid["r_corner"]][:2] @ rot.T
                xyr_end = rest_vert[kpid["l_corner"]][:2] @ rot.T
        
        else:
            raise ValueError(f"Invalid rotate_step {rotate_step}")

        xyz_l_1 = np.array([*xyl_start, cfg.rotate_z_grasp])
        xyz_l_2 = np.array([*((xyl_start + xyl_end) / 2), cfg.rotate_z_move])
        xyz_l_3 = np.array([*xyl_end, cfg.rotate_z_put])
        xyz_r_1 = np.array([*xyr_start, cfg.rotate_z_grasp])
        xyz_r_2 = np.array([*((xyr_start + xyr_end) / 2), cfg.rotate_z_move])
        xyz_r_3 = np.array([*xyr_end, cfg.rotate_z_put])

        self._interp_action_and_put_in_deque(
            xyz_l_1, xyz_l_2, xyz_l_3,
            xyz_r_1, xyz_r_2, xyz_r_3,
        )

    def _add_action_align_sleeve(self, align_step: int):
        env, cfg = self._env, self._cfg
        rest_mesh = env.get_raw_mesh_rest()
        rest_vert = rest_mesh.vertices
        curr_mesh = env.get_raw_mesh_curr()
        curr_vert = curr_mesh.vertices
        kpid = env.get_keypoint_idx()

        is_faceup, theta = self._shape_match(rest_mesh, curr_mesh, kpid)

        lsb, rsb = "l_sleeve_bottom", "r_sleeve_bottom"
        lst, rst = "l_sleeve_top", "r_sleeve_top"
        lar, rar = "l_armpit", "r_armpit"
        lcn, rcn = "l_corner", "r_corner"
        if int(is_faceup) + int(abs(theta) < np.pi / 2) == 1:
            lsb, rsb = rsb, lsb
            lst, rst = rst, lst
            lar, rar = rar, lar
            lcn, rcn = rcn, lcn
        
        if abs(theta) < np.pi / 2:
            sign_y_l, sign_y_r = -1., +1.
        else:
            sign_y_l, sign_y_r = +1., -1.
        
        def compute_xy(cn: str, ar: str, sb: str, st: str, sign_y: float, angle_delta_max: float):
            def compute_angle(vec1: np.ndarray, vec2: np.ndarray):
                return np.arccos(np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0))
            angle_ar = compute_angle(curr_vert[kpid[cn]][:2] - curr_vert[kpid[ar]][:2], 
                                     curr_vert[kpid[sb]][:2] - curr_vert[kpid[ar]][:2])
            ex, ey = self._local_coord(curr_vert[kpid[cn]][:2] - curr_vert[kpid[ar]][:2])
            d1 = np.linalg.norm(rest_vert[kpid[sb]][:2] - rest_vert[kpid[ar]][:2])
            d2 = np.linalg.norm(rest_vert[kpid[st]][:2] - rest_vert[kpid[ar]][:2])
            a2 = compute_angle(rest_vert[kpid[st]][:2] - rest_vert[kpid[ar]][:2],
                               rest_vert[kpid[sb]][:2] - rest_vert[kpid[ar]][:2])

            angle = compute_angle(rest_vert[kpid[cn]][:2] - rest_vert[kpid[ar]][:2],
                                  rest_vert[kpid[sb]][:2] - rest_vert[kpid[ar]][:2])
            if angle_ar > angle: # move to inside
                a = max(angle, angle_ar - angle_delta_max)
                xy_1 = curr_vert[kpid[sb]][:2]
                xy_3 = curr_vert[kpid[ar]][:2] + (
                    + ex * (d1 * np.cos(a))
                    + ey * (d1 * np.sin(a) + cfg.align_y_offset) * sign_y
                )
            else: # move to outside
                a = min(angle, angle_ar + angle_delta_max)
                xy_1 = curr_vert[kpid[st]][:2]
                xy_3 = curr_vert[kpid[ar]][:2] + (
                    + ex * (d2 * np.cos(a + a2))
                    + ey * (d2 * np.sin(a + a2) + cfg.align_y_offset) * sign_y
                )
            logger.info(f"curr_vert:\n{pprint.pformat({k: curr_vert[kpid[k]][:2] for k in [lcn, rcn, lar, rar, lsb, rsb, lst, rst]}, sort_dicts=False)}")
            logger.info(f"rest_vert:\n{pprint.pformat({k: rest_vert[kpid[k]][:2] for k in [lcn, rcn, lar, rar, lsb, rsb, lst, rst]}, sort_dicts=False)}")
            logger.info(f"xy_1={xy_1}, xy_3={xy_3}")
            return xy_1, xy_3
        
        if align_step == 0:
            angle_delta_max = cfg.align_angle_delta_max1
        elif align_step == 1:
            angle_delta_max = cfg.align_angle_delta_max2
        else:
            raise ValueError(f"Invalid align_step {align_step}")

        xyl_1, xyl_3 = compute_xy(lcn, lar, lsb, lst, sign_y_l, angle_delta_max)
        xyr_1, xyr_3 = compute_xy(rcn, rar, rsb, rst, sign_y_r, angle_delta_max)
        z1, z2, z3 = cfg.align_z_grasp, cfg.align_z_move, cfg.align_z_put

        xyz_l_1 = np.array([*xyl_1, z1])
        xyz_l_2 = np.array([*((xyl_1 + xyl_3) / 2), z2])
        xyz_l_3 = np.array([*xyl_3, z3])
        xyz_r_1 = np.array([*xyr_1, z1])
        xyz_r_2 = np.array([*((xyr_1 + xyr_3) / 2), z2])
        xyz_r_3 = np.array([*xyr_3, z3])

        move_dist_l = np.linalg.norm(xyl_1 - xyl_3)
        move_dist_r = np.linalg.norm(xyr_1 - xyr_3)
        logger.info(f"align move_dist_l={move_dist_l}, move_dist_r={move_dist_r}")
        if max(move_dist_l, move_dist_r) <= cfg.align_skip_threshold:
            return
        self._interp_action_and_put_in_deque(
            xyz_l_1, xyz_l_2, xyz_l_3,
            xyz_r_1, xyz_r_2, xyz_r_3,
        )

    def _add_action_fold_garment(self, fold_step: int):
        env, cfg = self._env, self._cfg
        rest_mesh = env.get_raw_mesh_rest()
        rest_vert = rest_mesh.vertices
        curr_mesh = env.get_raw_mesh_curr()
        curr_vert = curr_mesh.vertices
        kpid = env.get_keypoint_idx()

        if fold_step == 0:
            # fold two sleeves
            is_faceup, theta = self._shape_match(rest_mesh, curr_mesh, kpid)
            self._policy_cache["shape_match_for_fold"] = (is_faceup, theta)

            lst, rst = "l_sleeve_top", "r_sleeve_top"
            lsb, rsb = "l_sleeve_bottom", "r_sleeve_bottom"
            lsh, rsh = "l_shoulder", "r_shoulder"
            lar, rar = "l_armpit", "r_armpit"
            lcn, rcn = "l_corner", "r_corner"
            lco, rco = "l_collar", "r_collar"
            if int(is_faceup) + int(abs(theta) < np.pi / 2) == 1:
                lst, rst = rst, lst
                lsb, rsb = rsb, lsb
                lsh, rsh = rsh, lsh
                lar, rar = rar, lar
                lcn, rcn = rcn, lcn
                lco, rco = rco, lco
            
            xyl_1 = (curr_vert[kpid[lst]] + curr_vert[kpid[lsb]])[:2] / 2
            xyr_1 = (curr_vert[kpid[rst]] + curr_vert[kpid[rsb]])[:2] / 2

            if abs(theta) < np.pi / 2:
                sign_y = -1.
            else:
                sign_y = +1.

            px_max = (
                np.linalg.norm(curr_vert[kpid[rsh]][:2] - curr_vert[kpid[lsh]][:2]) / 2 
                - cfg.fold1_x_max_barrier
            )
            ex0, ey0 = self._local_coord(curr_vert[kpid[rsh]][:2] - curr_vert[kpid[lsh]][:2])
            def compute_xy3(sh: str, st: str, sb: str, sign_x: float, sign_y: float):
                ex, ey = ex0 * sign_x, ey0 * sign_y
                cvst, cvsb, cvsh = curr_vert[kpid[st]], curr_vert[kpid[sb]], curr_vert[kpid[sh]]
                px0 = np.dot(((cvst + cvsb) / 2 - cvsh)[:2], -ex)
                py0 = np.dot(((cvst + cvsb) / 2 - cvsh)[:2], +ey)
                if px_max > px0:
                    px, py = px0, py0
                else:
                    px, py = px_max, (px0 ** 2 + py0 ** 2 - px_max ** 2) ** 0.5
                logger.info(f"px0={px0}, py0={py0}, px={px}, py={py}")

                return cvsh[:2] + ex * px + ey * py
            xyl_3 = compute_xy3(lsh, lst, lsb, +1., sign_y)
            xyr_3 = compute_xy3(rsh, rst, rsb, -1., sign_y)

            def proj_coeff(x: np.ndarray, a: np.ndarray, b: np.ndarray): 
                return np.dot(x - a, b - a) / np.sum((b - a) ** 2)
            c = proj_coeff(
                (xyl_3 + xyr_3) / 2, 
                (curr_vert[kpid[lcn]] + curr_vert[kpid[rcn]])[:2] / 2,
                (curr_vert[kpid[lco]] + curr_vert[kpid[rco]])[:2] / 2
            )
            self._policy_cache["sleeve_fold_proj_coeff"] = c
            self._meta_info["sleeve_fold_proj_coeff"] = c
            logger.info(f"sleeve_fold_proj_coeff={c}")

            xyl_2 = (curr_vert[kpid[lar]] + curr_vert[kpid[lsh]])[:2] / 2
            xyr_2 = (curr_vert[kpid[rar]] + curr_vert[kpid[rsh]])[:2] / 2
            move_dist_l = np.linalg.norm(xyl_1 - xyl_3)
            move_dist_r = np.linalg.norm(xyr_1 - xyr_3)
            logger.info(f"fold1 move_dist_l={move_dist_l}, move_dist_r={move_dist_r}")

            z1, z2l, z2r, z3 = cfg.fold1_z_grasp, cfg.fold1_z_move, cfg.fold1_z_move, cfg.fold1_z_put
            z2l += cfg.fold1_z_move_proportional_to_dist * move_dist_l
            z2r += cfg.fold1_z_move_proportional_to_dist * move_dist_r
            logger.info(f"z2={cfg.fold1_z_move} z2l={z2l}, z2r={z2r}")

        elif fold_step == 1:
            shape_match_for_fold: tuple[bool, float] = self._policy_cache["shape_match_for_fold"]
            (is_faceup, theta) = shape_match_for_fold
            sleeve_fold_proj_coeff: float = self._policy_cache["sleeve_fold_proj_coeff"]
            
            lco, rco = "l_collar", "r_collar"
            lcn, rcn = "l_corner", "r_corner"
            lsh, rsh = "l_shoulder", "r_shoulder"
            if int(is_faceup) + int(abs(theta) < np.pi / 2) == 1:
                lco, rco = rco, lco
                lcn, rcn = rcn, lcn
                lsh, rsh = rsh, lsh
            
            if sleeve_fold_proj_coeff > cfg.fold2_sleeve_fold_proj_coeff_th:
                # short sleeves, fold from corner to collar
                xyl_1 = curr_vert[kpid[lcn]][:2]
                xyr_1 = curr_vert[kpid[rcn]][:2]
                
                ex, ey = self._local_coord(curr_vert[kpid[rco]][:2] - curr_vert[kpid[lco]][:2])
                xyc_3 = (curr_vert[kpid[lco]][:2] + curr_vert[kpid[rco]][:2]) / 2
                xyd_3 = np.linalg.norm(rest_vert[kpid[lcn]][:2] - rest_vert[kpid[rcn]][:2])
                y0 = 0.
                if abs(theta) < np.pi / 2:
                    sign_y = -1.
                else:
                    sign_y = +1.
            else:
                # long sleeves, fold from shoulder to corner
                xyl_1 = curr_vert[kpid[lsh]][:2]
                xyr_1 = curr_vert[kpid[rsh]][:2]

                ex, ey = self._local_coord(curr_vert[kpid[rcn]][:2] - curr_vert[kpid[lcn]][:2])
                xyc_3 = (curr_vert[kpid[lcn]][:2] + curr_vert[kpid[rcn]][:2]) / 2
                xyd_3 = np.linalg.norm(curr_vert[kpid[lsh]][:2] - curr_vert[kpid[rsh]][:2])
                y0 = np.linalg.norm(
                    (rest_vert[kpid[lco]] + rest_vert[kpid[rco]])[:2] / 2 -
                    (rest_vert[kpid[lsh]] + rest_vert[kpid[rsh]])[:2] / 2
                )
                if abs(theta) < np.pi / 2:
                    sign_y = +1.
                else:
                    sign_y = -1.
            
            xyl_3 = xyc_3 - ex * xyd_3 / 2 * cfg.fold2_x_relax + ey * sign_y * (cfg.fold2_y_offset + y0)
            xyr_3 = xyc_3 + ex * xyd_3 / 2 * cfg.fold2_x_relax + ey * sign_y * (cfg.fold2_y_offset + y0)

            xyl_2 = (xyl_1 + xyl_3) / 2
            xyr_2 = (xyr_1 + xyr_3) / 2

            z1, z2l, z2r, z3 = cfg.fold2_z_grasp, cfg.fold2_z_move, cfg.fold2_z_move, cfg.fold2_z_put
        
        else:
            raise ValueError(f"Invalid fold_step {fold_step}")
        
        logger.info(f"Fold step {fold_step}")
        logger.info(f"xyl_1 {xyl_1}")
        logger.info(f"xyr_1 {xyr_1}")
        logger.info(f"xyl_2 {xyl_2}")
        logger.info(f"xyr_2 {xyr_2}")
        logger.info(f"xyl_3 {xyl_3}")
        logger.info(f"xyr_3 {xyr_3}")

        xyz_l_1 = np.array([*xyl_1, z1])
        xyz_l_2 = np.array([*xyl_2, z2l])
        xyz_l_3 = np.array([*xyl_3, z3])
        xyz_r_1 = np.array([*xyr_1, z1])
        xyz_r_2 = np.array([*xyr_2, z2r])
        xyz_r_3 = np.array([*xyr_3, z3])

        self._interp_action_and_put_in_deque(
            xyz_l_1, xyz_l_2, xyz_l_3,
            xyz_r_1, xyz_r_2, xyz_r_3,
        )

    ### high-level policy api ###
    def _compute_new_action(self):
        if not self._cfg.skip_rotate:
            if self._current_stage == 0:
                # pre-policy action
                self._add_action_pre_policy()
            elif self._current_stage in [1, 3]:
                # rotate garment
                self._add_action_rotate_garment({1: 0, 3: 1}[self._current_stage])
            elif self._current_stage in [2, 4]:
                # align sleeves
                self._add_action_align_sleeve({2: 0, 4: 1}[self._current_stage])
            elif self._current_stage in [5, 6]:
                # fold garment
                self._add_action_fold_garment({5: 0, 6: 1}[self._current_stage])
            else:
                # end of policy
                self._append_action(None)
        else:
            if self._current_stage == 0:
                # pre-policy action
                self._add_action_pre_policy()
            elif self._current_stage in [1, 2]:
                # fold garment
                self._add_action_fold_garment(self._current_stage - 1)
            else:
                # end of policy
                self._append_action(None)
        
        self._current_stage += 1