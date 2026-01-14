from typing import Optional, Literal, Union
from dataclasses import dataclass, field
import math
import copy
import atexit

import trimesh.transformations as tra
import numpy as np
import torch

import batch_urdf
import urx
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
import pyrealsense2 as rs
from garmentds.real.rm_utils import CommandSender
from garmentds.real.camera_calibration_result import RMBASE_TO_URBASE
import garmentds.common.utils as utils


ROBOTL_BASE = tra.translation_matrix([0., 0., 0.32])
ROBOTR_BASE = ROBOTL_BASE @ RMBASE_TO_URBASE


@dataclass
class RobotlCfg:
    use: bool = True
    urdf_path: str = "asset/ur5/ur5.urdf"
    mesh_dir: str = "asset/ur5/meshes"
    robot_ip: str = "192.168.50.114"
    robot_joints_name: tuple[str] = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
    robot_joints_init_val: tuple[float] = (0., -1.0, 1.0, -1.0, -1.0, -1.0) # no usage, only to compute ik
    ee_str: str = "tcp"
    robot_tcp_init_mat_to_base: np.ndarray = field(default_factory=lambda: (
        tra.translation_matrix([0.5, 0.2, 0.35]) @ 
        tra.euler_matrix(math.pi, 0., 0.)
    ))
    robot_tcp_init_mat: np.ndarray = field(init=False)
    robot_base_mat: np.ndarray = field(init=False)
    use_gripper: bool = True
    default_acc: float = 0.2
    default_vel: float = 0.2

    def __post_init__(self):
        self.robot_base_mat = ROBOTL_BASE.copy()
        self.robot_tcp_init_mat = self.robot_base_mat @ self.robot_tcp_init_mat_to_base
        self.urdf_path = utils.get_path_handler()(self.urdf_path)
        self.mesh_dir = utils.get_path_handler()(self.mesh_dir)


@dataclass
class RobotrCfg:
    use: bool = True
    urdf_path: str = "asset/rm_65_b_description/urdf/urdf.urdf"
    mesh_dir: str = "asset/rm_65_b_description/meshes"
    robot_joints_name: tuple[str] = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
    robot_joints_init_val: tuple[float] = (0.5, 0.5, 1.0, 0.0, 1.5, -0.0) # no usage, only to compute ik
    ee_str: str = "tcp"
    robot_tcp_init_mat_to_base: np.ndarray = field(default_factory=lambda: (
        tra.translation_matrix([+0.1, -0.3, 0.15]) @ 
        tra.euler_matrix(math.pi, 0., 0.)
    ))
    robot_tcp_init_mat: np.ndarray = field(init=False)
    robot_base_mat: np.ndarray = field(init=False)
    use_gripper: bool = True
    default_speed: float = 0.05

    def __post_init__(self):
        self.robot_base_mat = ROBOTR_BASE.copy()
        self.robot_tcp_init_mat = self.robot_base_mat @ self.robot_tcp_init_mat_to_base
        self.urdf_path = utils.get_path_handler()(self.urdf_path)
        self.mesh_dir = utils.get_path_handler()(self.mesh_dir)


@dataclass
class CameraCfg:
    use: bool = True
    height: int = 480
    width: int = 640
    frequency: int = 30


@dataclass
class APICfg:
    ik_kwargs: dict = field(default_factory=lambda: {"max_iter": 128})


class RealAPIDesktop:
    GRIPPER_ACTION = Literal["open", "close", "none"]
    def __init__(
        self,
        api_cfg: Optional[APICfg]=None,
        robotl_cfg: Optional[RobotlCfg]=None,
        robotr_cfg: Optional[RobotrCfg]=None,
        camera_cfg: Optional[CameraCfg]=None,
    ) -> None:
        self._closed = False
        self._urdf_factory_kwargs = dict(dtype=torch.float32, device=torch.device("cpu"))

        print("[INFO] RealAPIDesktop initializing ...")

        self._api_cfg = copy.deepcopy(api_cfg) if api_cfg is not None else APICfg()
        self._init_robotl(robotl_cfg)
        self._init_robotr(robotr_cfg)
        self._init_camera(camera_cfg)

        atexit.register(self._close)

        print("[INFO] RealAPIDesktop initialized ...")
    
    def _init_robotl(self, robotl_cfg: Optional[RobotlCfg]=None):
        if robotl_cfg is None:
            robotl_cfg = RobotlCfg()
        self._robotl_cfg = copy.deepcopy(robotl_cfg)

        if robotl_cfg.use:
            self._robotl = urx.Robot(robotl_cfg.robot_ip)
            if robotl_cfg.use_gripper:
                print("[INFO] test robot left gripper ...")
                self._gripperl = Robotiq_Two_Finger_Gripper(self._robotl)
                self._gripperl_action("open", skip_enter=True)
                self._gripperl_action("close", skip_enter=True)
                self._gripperl_action("open", skip_enter=True)

        self._urdfl = batch_urdf.URDF(batch_size=1, urdf_path=robotl_cfg.urdf_path, mesh_dir=robotl_cfg.mesh_dir)
        self._urdfl.update_cfg(self._robotl_qpos_ndarray_to_dict(np.array(self._robotl_cfg.robot_joints_init_val)))
        self._urdfl.update_base_link_transformation(
            list(self._urdfl.base_link_map.keys())[0], 
            torch.tensor(self._robotl_cfg.robot_base_mat.copy()[None, ...], **self._urdf_factory_kwargs)
        )

        print("[INFO] move robot left to init pose ...")
        self._robotl_movel(robotl_cfg.robot_tcp_init_mat, acc=None, vel=None, skip_enter=False)
    
    def _init_robotr(self, robotr_cfg: Optional[RobotrCfg]=None):
        if robotr_cfg is None:
            robotr_cfg = RobotrCfg()
        self._robotr_cfg = copy.deepcopy(robotr_cfg)
        
        if robotr_cfg.use:
            if robotr_cfg.use_gripper:
                print("[INFO] test robot right gripper ...")
            self._robotr = CommandSender(use_gripper=robotr_cfg.use_gripper)

        self._urdfr = batch_urdf.URDF(batch_size=1, urdf_path=robotr_cfg.urdf_path, mesh_dir=robotr_cfg.mesh_dir)
        self._urdfr.update_cfg(self._robotr_qpos_ndarray_to_dict(np.array(self._robotr_cfg.robot_joints_init_val)))
        self._urdfr.update_base_link_transformation(
            list(self._urdfr.base_link_map.keys())[0], 
            torch.tensor(self._robotr_cfg.robot_base_mat.copy()[None, ...], **self._urdf_factory_kwargs)
        )

        print("[INFO] move robot right to init pose ...")
        # self._robotr_movej(self._robotr.get_qpos(), speed=None, skip_enter=False)
        self._robotr_movel(robotr_cfg.robot_tcp_init_mat, speed=None, skip_enter=False)
        
    def _init_camera(self, camera_cfg: Optional[CameraCfg]=None):
        if camera_cfg is None:
            camera_cfg = CameraCfg()
        self._camera_cfg = copy.deepcopy(camera_cfg)
        if not camera_cfg.use:
            return

        self._pipeline = pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, camera_cfg.width, camera_cfg.height, rs.format.rgb8, camera_cfg.frequency)
        config.enable_stream(rs.stream.depth, camera_cfg.width, camera_cfg.height, rs.format.z16, camera_cfg.frequency)
        profile = pipeline.start(config)
        self._align = align = rs.align(rs.stream.color)
    
    def _robotl_qpos_dict_to_ndarray(self, qpos_dict: dict[str, torch.Tensor]) -> np.ndarray:
        return np.array([float(qpos_dict[k]) for k in self._robotl_cfg.robot_joints_name])
    
    def _robotl_qpos_ndarray_to_dict(self, qpos_ndarray: np.ndarray) -> dict[str, torch.Tensor]:
        return {k: torch.tensor([v], **self._urdf_factory_kwargs) for k, v in zip(self._robotl_cfg.robot_joints_name, qpos_ndarray)}
    
    def _robotr_qpos_dict_to_ndarray(self, qpos_dict: dict[str, torch.Tensor]) -> np.ndarray:
        return np.array([float(qpos_dict[k]) for k in self._robotr_cfg.robot_joints_name])
    
    def _robotr_qpos_ndarray_to_dict(self, qpos_ndarray: np.ndarray) -> dict[str, torch.Tensor]:
        return {k: torch.tensor([v], **self._urdf_factory_kwargs) for k, v in zip(self._robotr_cfg.robot_joints_name, qpos_ndarray)}
    
    def _robotl_movej(self, qpos: np.ndarray, acc: Optional[float], vel: Optional[float], skip_enter: bool):
        print(f"[INFO] move robot left to {qpos} ...")
        assert isinstance(qpos, np.ndarray), type(qpos)
        assert qpos.shape == (len(self._robotl_cfg.robot_joints_name),), qpos.shape

        if not skip_enter and self._robotl_cfg.use:
            input("press continue to move robot left ...")
        
        self._urdfl.update_cfg(self._robotl_qpos_ndarray_to_dict(qpos))
        if acc is None:
            acc = self._robotl_cfg.default_acc
        if vel is None:
            vel = self._robotl_cfg.default_vel
        if self._robotl_cfg.use:
            self._robotl.movej(qpos, acc=acc, vel=vel)
    
    def _robotl_movel(self, mat: np.ndarray, acc: Optional[float], vel: Optional[float], skip_enter: bool):
        assert isinstance(mat, np.ndarray), type(mat)
        assert mat.shape == (4, 4), mat.shape

        cfg, info = self._urdfl.inverse_kinematics(
            self._robotl_cfg.ee_str, 
            torch.tensor(np.array([mat]), **self._urdf_factory_kwargs),
            **self._api_cfg.ik_kwargs
        )
        print(f"[INFO] ik {info}")

        self._robotl_movej(self._robotl_qpos_dict_to_ndarray(cfg), acc, vel, skip_enter)

    def _robotr_movej(self, qpos: np.ndarray, speed: Optional[float], skip_enter: bool):
        print(f"[INFO] move robot right to {qpos} ...")
        assert isinstance(qpos, np.ndarray), type(qpos)
        assert qpos.shape == (len(self._robotr_cfg.robot_joints_name),), qpos.shape

        if not skip_enter and self._robotr_cfg.use:
            input("press continue to move robot right ...")
        
        self._urdfr.update_cfg(self._robotr_qpos_ndarray_to_dict(qpos))
        if speed is None:
            speed = self._robotr_cfg.default_speed
        if self._robotr_cfg.use:
            self._robotr.move_arm(qpos, speed=speed)
    
    def _robotr_movel(self, mat: np.ndarray, speed: Optional[float], skip_enter: bool):
        assert isinstance(mat, np.ndarray), type(mat)
        assert mat.shape == (4, 4), mat.shape

        cfg, info = self._urdfr.inverse_kinematics(
            self._robotr_cfg.ee_str, 
            torch.tensor(np.array([mat]), **self._urdf_factory_kwargs),
            **self._api_cfg.ik_kwargs
        )
        print(f"[INFO] ik {info}")

        self._robotr_movej(self._robotr_qpos_dict_to_ndarray(cfg), speed, skip_enter)

    def _gripperl_action(self, action: GRIPPER_ACTION, skip_enter: bool):
        print(f"[INFO] {action} robot left gripper ...")
        if action == "none":
            return
        if not skip_enter:
            input(f"press continue to {action} robot left gripper ...")

        if action == "open":
            self._gripperl.open_gripper()
        elif action == "close":
            self._gripperl.close_gripper()
        else:
            raise ValueError(f"unknown gripper action: {action}")
    
    def _gripperr_action(self, action: GRIPPER_ACTION, skip_enter: bool):
        print(f"[INFO] {action} robot right gripper ...")
        if action == "none":
            return
        if not skip_enter:
            input(f"press continue to {action} robot right gripper ...")

        if action == "open":
            self._robotr.open_gripper()
        elif action == "close":
            self._robotr.close_gripper()
        else:
            raise ValueError(f"unknown gripper action: {action}")

    def _take_picture(self) -> dict[Literal["color", "depth"], np.ndarray]:
        frames = self._align.process(self._pipeline.wait_for_frames())
        color = np.asanyarray(frames.get_color_frame().get_data()) # [480, 640, 3], uint8
        depth = np.asanyarray(frames.get_depth_frame().get_data()) / 1000. # [480, 640], float

        return dict(color = color, depth = depth)
    
    def _close(self):
        if not self._closed:
            if self._robotl_cfg.use:
                if hasattr(self, "_robotl"):
                    self._robotl.close()
            if hasattr(self, "_camera_cfg") and self._camera_cfg.use:
                self._pipeline.stop()
            self._closed = True
            print("[INFO] RealAPIDesktop all closed ...")
    
    @property
    def urdf_factory_kwargs(self):
        return self._urdf_factory_kwargs
    
    @property
    def robotl_cfg(self):
        return self._robotl_cfg
    
    @property
    def robotr_cfg(self):
        return self._robotr_cfg
    
    @property
    def camera_cfg(self):
        return self._camera_cfg
    
    @property
    def api_cfg(self):
        return self._api_cfg
    
    @property
    def urdfl(self):
        return self._urdfl
    
    @property
    def urdfr(self):
        return self._urdfr
    
    def robotl_qpos_dict_to_ndarray(self, qpos_dict: dict[str, torch.Tensor]) -> np.ndarray:
        return self._robotl_qpos_dict_to_ndarray(qpos_dict)
    
    def robotl_qpos_ndarray_to_dict(self, qpos_ndarray: np.ndarray) -> dict[str, torch.Tensor]:
        return self._robotl_qpos_ndarray_to_dict(qpos_ndarray)

    def robotr_qpos_dict_to_ndarray(self, qpos_dict: dict[str, torch.Tensor]) -> np.ndarray:
        return self._robotr_qpos_dict_to_ndarray(qpos_dict)
    
    def robotr_qpos_ndarray_to_dict(self, qpos_ndarray: np.ndarray) -> dict[str, torch.Tensor]:
        return self._robotr_qpos_ndarray_to_dict(qpos_ndarray)

    def robotl_movej(self, qpos: np.ndarray, acc=None, vel=None, skip_enter=False):
        self._robotl_movej(qpos, acc, vel, skip_enter)
    
    def robotl_movel(self, mat: np.ndarray, acc=None, vel=None, skip_enter=False):
        self._robotl_movel(mat, acc, vel, skip_enter)
    
    def robotr_movej(self, qpos: np.ndarray, speed=None, skip_enter=False):
        self._robotr_movej(qpos, speed, skip_enter)
    
    def robotr_movel(self, mat: np.ndarray, speed=None, skip_enter=False):
        self._robotr_movel(mat, speed, skip_enter)
    
    def gripperl_action(self, action: GRIPPER_ACTION, skip_enter=False):
        self._gripperl_action(action, skip_enter)
    
    def gripperr_action(self, action: GRIPPER_ACTION, skip_enter=False):
        self._gripperr_action(action, skip_enter)
    
    def gripper_action(self, action_l: GRIPPER_ACTION, action_r: GRIPPER_ACTION, skip_enter=False):
        self._gripperl_action(action_l, skip_enter)
        self._gripperr_action(action_r, skip_enter)
    
    def get_robotl_ee_pose(self) -> np.ndarray:
        return utils.torch_to_numpy(self._urdfl.link_transform_map[self._robotl_cfg.ee_str])[0, ...]

    def get_robotr_ee_pose(self) -> np.ndarray:
        return utils.torch_to_numpy(self._urdfr.link_transform_map[self._robotr_cfg.ee_str])[0, ...]
    
    def take_picture(self):
        """
        color # [480, 640, 3], uint8
        depth # [480, 640], float in meter
        """
        return self._take_picture()
    
    def close(self):
        self._close()


if __name__ == "__main__":
    real_api_desktop = RealAPIDesktop(
        robotl_cfg = RobotlCfg(use=False),
        robotr_cfg = RobotrCfg(
            use_gripper=False,
            robot_joints_init_val=(0.5, 0.5, 2.0, -0.5, -1.0, -0.0),
            robot_tcp_init_mat_to_base=tra.translation_matrix([-0.4, -0.3, 0.25]) @ tra.euler_matrix(0., np.pi / 2, -np.pi * 3 / 4),
        ),
        camera_cfg = CameraCfg(use=False),
    )
    print(real_api_desktop.get_robotr_ee_pose())
    print(tra.euler_from_matrix(real_api_desktop.get_robotr_ee_pose()), tra.translation_from_matrix(real_api_desktop.get_robotr_ee_pose()))