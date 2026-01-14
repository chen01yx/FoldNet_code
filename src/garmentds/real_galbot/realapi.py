import logging
logger = logging.getLogger(__name__)

from dataclasses import dataclass, field
from typing import Optional, Literal
import copy

import numpy as np
import trimesh.transformations as tra

from garmentds.real_galbot.gsocket_utils import GalbotClient
from garmentds.real_galbot.vis_o3d import O3dVisualizer
from garmentds.foldenv.fold_env import Robot, RobotCfg, Picker
import garmentds.common.utils as utils

realapi_timer = utils.Timer(name="real_api", logger=logger)


@dataclass
class RealAPICfg:
    robot_cfg: Optional[RobotCfg] = None
    
    robot_offset_l: np.ndarray = field(default_factory=lambda: np.array([0., 0., 0.]))
    robot_offset_r: np.ndarray = field(default_factory=lambda: np.array([0., 0., 0.]))
    grasp_offset_l: np.ndarray = field(default_factory=lambda: np.array([0., 0., 0.]))
    grasp_offset_r: np.ndarray = field(default_factory=lambda: np.array([0., 0., 0.]))
    grasp_force_z: float = 0.02
    # z_affine_coeff: np.ndarray = field(default_factory=lambda: np.array([0.02, 0.22, 0.02, 0.32]))
    press_enter_before_move: bool = True
    use_visualizer: bool = True
    
    def __post_init__(self):
        if self.robot_cfg is None:
            self.robot_cfg = RobotCfg()
        self.robot_offset_l = np.array(self.robot_offset_l)
        self.robot_offset_r = np.array(self.robot_offset_r)
        self.grasp_offset_l = np.array(self.grasp_offset_l)
        self.grasp_offset_r = np.array(self.grasp_offset_r)
        # self.z_affine_coeff = np.array(self.z_affine_coeff)


class DummyPicker(Picker):
    def __init__(self, *args, **kwargs):
        self._val = self.OPEN
        self._val_float = float(self._val)
    
    def set_tf(self, *args, **kwargs):
        """overwrite original function"""
        pass
    
    def set_action(self, action: Optional[float]):
        """overwrite original function"""
        """0 is open and 1 is close"""
        self._prev_val = self._val
        if action is None:
            action = self._prev_val
        self._val = self.OPEN if self.is_open_action(action) else self.CLOSE
        self._val_float = float(action)

    
class RealAPI:
    def __init__(self, cfg: Optional[RealAPICfg] = None):
        if cfg is None:
            cfg = RealAPICfg()
        self._cfg = copy.deepcopy(cfg)
        
        self._api = GalbotClient()
        self._api_joint_list = self._api.get_joints_name_list()
        if cfg.use_visualizer:
            self._vis = O3dVisualizer()
        self._scene_pc = None # for visualize only
        self._action_pc = None # for visualize only
        self._cam_ext = None # for visualize only
        self._cam_int = None # for visualize only
        
        # We maintain two set of urdfs, one for policy, another for the real robot.
        # Due to the offset we applied, the qpos are slightly different.
        self._robot_real = Robot(DummyPicker(), DummyPicker(), cfg.robot_cfg)
        self._robot_policy = Robot(DummyPicker(), DummyPicker(), cfg.robot_cfg)
        
        qpos = self._robot_real.get_qpos()
        for j, v in zip(self._api_joint_list, self._api.get_qpos()):
            qpos[j] = v
        self._robot_real.set_qpos(qpos, True)
        self._robot_policy.set_qpos(qpos, True)
        
        self._api_cache = dict(picker_l=None, picker_r=None)
    
    @realapi_timer.timer
    def _call_robot_ik_solver(self, robot: Robot, xyz_l: Optional[np.ndarray] = None, xyz_r: Optional[np.ndarray] = None) -> dict[str, float]:
        info = robot.set_target_xyz(1, xyz_l, xyz_r)
        qpos = robot.waypoints_qpos.pop()
        return qpos
    
    def _call_robot_modify_gripper_qpos(self, robot: Robot, picker_l: float, picker_r: float):
        qpos = robot.get_qpos()
        robot.modify_gripper_qpos(qpos, "left", picker_l)
        robot.modify_gripper_qpos(qpos, "right", picker_r)
        return qpos
    
    @realapi_timer.timer
    def _vis_movej_impl(self, qpos: dict[str, float]):
        self._vis.render(qpos, self._scene_pc, self._action_pc, self._cam_ext)
    
    def _vis_movej(self, qpos: dict[str, float]):
        if self._cfg.use_visualizer:
            self._vis_movej_impl(qpos)
    
    @realapi_timer.timer
    def _api_movej_impl(self, qpos: dict[str, float], asynchronous=False):
        logger.info(f"api_movej_impl: asynchronous={asynchronous}")
        self._api.set_qpos(
            [qpos[k] for k in self._api_joint_list], 
            speed=1.0, asynchronous=asynchronous,
        )
        
    def _api_movej(self, qpos: dict[str, float], asynchronous=False, press_enter_before_move_overwrite=None):
        if press_enter_before_move_overwrite is None:
            press_enter_before_move_overwrite = self._cfg.press_enter_before_move
        if press_enter_before_move_overwrite:
            input("press enter to move robot arms ...")
        self._api_movej_impl(qpos, asynchronous=asynchronous)
    
    @realapi_timer.timer
    def _api_moveg_impl(self, action_l: float, action_r: float):
        # in fold_env convention, 0 is open and 1 is close, in realapi convention, 0 is close and 1 is open
        logger.info(f"api_moveg_impl: {action_l} {action_r} {self._api_cache['picker_l']} {self._api_cache['picker_r']}")
        if (
            self._api_cache["picker_l"] == action_l and 
            self._api_cache["picker_r"] == action_r
        ):
            pass
        else:
            self._api_cache["picker_l"] = action_l
            self._api_cache["picker_r"] = action_r
            self._api.sync() # sync before manipulate grippers
            self._api.set_grippers_status(
                np.clip(1 - action_l, 0., 1.), 
                np.clip(1 - action_r, 0., 1.), 
                force=1.0
            )
    
    def _api_moveg(self, action_l: float, action_r: float, press_enter_before_move_overwrite=None):
        if press_enter_before_move_overwrite is None:
            press_enter_before_move_overwrite = self._cfg.press_enter_before_move
        if press_enter_before_move_overwrite:
            input("press enter to move grippers ...")
        self._api_moveg_impl(action_l, action_r)
    
    def movej(self, qpos_real: dict[str, float], qpos_policy: dict[str, float], asynchronous=False, press_enter_before_move_overwrite=None):
        qpos_real = copy.deepcopy(qpos_real)
        self._robot_real.set_qpos(qpos_real, True)
        self._robot_policy.set_qpos(qpos_policy, True)
        self._vis_movej(qpos_real)
        self._api_movej(qpos_real, asynchronous=asynchronous, press_enter_before_move_overwrite=press_enter_before_move_overwrite)
    
    def moveg(self, picker_l: Optional[float]=None, picker_r: Optional[float]=None, press_enter_before_move_overwrite=None):
        gripper_state = self._robot_policy.get_gripper_state()
        if picker_l is None:
            picker_l = gripper_state["left"]
        if picker_r is None:
            picker_r = gripper_state["right"]
        qpos_real = self._call_robot_modify_gripper_qpos(self._robot_real, picker_l, picker_r)
        qpos_policy = self._call_robot_modify_gripper_qpos(self._robot_policy, picker_l, picker_r)
        
        def set_robot(robot: Robot, qpos: dict[str, float]):
            robot.set_qpos(qpos, False)
            robot.picker["left"].set_action(picker_l)
            robot.picker["right"].set_action(picker_r)
            
        set_robot(self._robot_real, qpos_real)
        set_robot(self._robot_policy, qpos_policy)
        self._vis_movej(qpos_real)
        self._api_moveg(picker_l, picker_r, press_enter_before_move_overwrite=press_enter_before_move_overwrite)
    
    @realapi_timer.timer
    def get_rgbd(self):
        """
        Returns:
        - color: np.ndarray, shape=(480, 640, 3), dtype=np.uint8
        - depth: np.ndarray, shape=(480, 640), dtype=np.float16
        """
        return self._api.get_rgbd()
    
    @realapi_timer.timer
    def get_camera_intrinsics(self):
        return self._api.get_camera_intrinsics()
    
    @realapi_timer.timer
    def get_camera_extrinsics(self):
        raw_ext = self._api.get_camera_extrinsics()
        tf = (
            utils.torch_to_numpy(self._robot_real.urdf.link_transform_map[raw_ext.from_link])[0, :, :] @
            tra.translation_matrix(raw_ext.translation) @ 
            tra.quaternion_matrix(np.array(raw_ext.rotation_xyzw)[[3, 0, 1, 2]]) @ 
            tra.euler_matrix(np.pi, 0., 0.) #  this is due to different conventions of extrinsic matrix
        )
        return tf
    
    def _add_offset(self, xyz: np.ndarray, picker: float, hand: Literal["left", "right"]):
        grasp_offset = dict(left=self._cfg.grasp_offset_l, right=self._cfg.grasp_offset_r)[hand]
        robot_offset = dict(left=self._cfg.robot_offset_l, right=self._cfg.robot_offset_r)[hand]
        new_xyz = xyz.copy()
        
        if picker == Picker.CLOSE and self._api_cache[dict(left="picker_l", right="picker_r")[hand]] == Picker.OPEN:
            new_xyz[2] = self._cfg.grasp_force_z
            new_xyz += grasp_offset
        elif picker == Picker.CLOSE and self._api_cache[dict(left="picker_l", right="picker_r")[hand]] == Picker.CLOSE:
            new_xyz += robot_offset
        else:
            new_xyz += robot_offset
        return new_xyz
    
    def set_parameters_to_visualize(
        self, 
        action_pc: Optional[np.ndarray], 
        scene_pc: Optional[tuple[np.ndarray, np.ndarray]], 
        cam_ext: Optional[np.ndarray], 
        cam_int: Optional[np.ndarray]
    ):
        if action_pc is not None:
            self._action_pc = action_pc.copy()
        if scene_pc is not None:
            self._scene_pc = np.concatenate(scene_pc, axis=1)
        if cam_ext is not None:
            self._cam_ext = cam_ext.copy()
        if cam_int is not None:
            self._cam_int = cam_int.copy()
    
    @realapi_timer.timer
    def step(
        self, 
        xyz_l: Optional[np.ndarray]=None, xyz_r: Optional[np.ndarray]=None, 
        picker_l: Optional[float]=None, picker_r: Optional[float]=None,
        asynchronous=False, 
    ):
        logger.info(f"RealAPI step: xyz_l={xyz_l}, xyz_r={xyz_r}, picker_l={picker_l}, picker_r={picker_r}, async={asynchronous}")
        tcp_xyz_policy = self._robot_policy.get_tcp_xyz()
        if xyz_l is None:
            xyz_l = tcp_xyz_policy["left"]
        if xyz_r is None:
            xyz_r = tcp_xyz_policy["right"]
        qpos_real = self._call_robot_ik_solver(
            self._robot_real, 
            self._add_offset(xyz_l, picker_l, "left"), 
            self._add_offset(xyz_r, picker_r, "right"),
        )
        qpos_policy = self._call_robot_ik_solver(self._robot_policy, xyz_l, xyz_r)
        self.movej(qpos_real, qpos_policy, asynchronous=asynchronous)
        self.moveg(picker_l, picker_r)
    
    def moveinit(self, lift_height: float = 0.15):
        self.moveg(Picker.OPEN, Picker.OPEN, press_enter_before_move_overwrite=True)
        
        tcp_xyz_policy = self._robot_policy.get_tcp_xyz()
        xyz_l = tcp_xyz_policy["left"]
        xyz_r = tcp_xyz_policy["right"]
        xyz_l[2] = max(xyz_l[2], lift_height)
        xyz_r[2] = max(xyz_r[2], lift_height)
        qpos_real = self._call_robot_ik_solver(self._robot_real, xyz_l + self._cfg.robot_offset_l, xyz_r + self._cfg.robot_offset_r)
        qpos_policy = self._call_robot_ik_solver(self._robot_policy, xyz_l, xyz_r)
        self.movej(qpos_real, qpos_policy, asynchronous=False, press_enter_before_move_overwrite=True)
        self.movej(self._cfg.robot_cfg.init_qpos, self._cfg.robot_cfg.init_qpos, asynchronous=False, press_enter_before_move_overwrite=True)
    
    def sync(self):
        self._api.sync()
        
    @property
    def cfg(self):
        return self._cfg
    
    @property
    def robot_real(self):
        return self._robot_real
    
    @property
    def robot_policy(self):
        return self._robot_policy