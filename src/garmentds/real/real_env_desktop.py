from dataclasses import dataclass, field
from typing import Optional, Union, Literal
import copy
import math
import atexit

import sapien.core as sapien
from sapien.utils.viewer import Viewer

import numpy as np
import torch
from PIL import Image, ImageDraw
import trimesh.transformations as tra

import batch_urdf
from garmentds.real.real_api_desktop import (
    RealAPIDesktop, RobotlCfg, RobotrCfg, CameraCfg, APICfg, 
    ROBOTL_BASE, ROBOTR_BASE
)
from garmentds.real.camera_calibration_result import CAMERA_TO_UREE, INTRINSICS
import garmentds.real.utils as real_utils
import garmentds.real.human_aabb_app as human_aabb_app
import garmentds.common.utils as utils
from garmentds.real.sam_utils import GroundedSAM, SAMCfg


@dataclass
class SapienViewerCfg:
    rt_samples_per_pixel: int = 4
    rt_use_denoiser: bool = False
    timestep: float = 0.01
    resolution: tuple[float, float] = (1280, 960)
    default_step_n: int = 20
    urdfl_path: str = "asset/ur5/ur5.urdf"
    gripperl_path: str = "asset/robotiq_arg85_description/robots/robotiq_arg85_description.urdf"
    urdfr_path: str = "asset/rm_65_b_description/urdf/urdf.urdf"
    gripperr_path: str = "asset/EG.obj"
    device: str = "cuda:0"

    def __post_init__(self):
        self.urdfl_path = utils.get_path_handler()(self.urdfl_path)
        self.gripperl_path = utils.get_path_handler()(self.gripperl_path)
        self.urdfr_path = utils.get_path_handler()(self.urdfr_path)
        self.gripperr_path = utils.get_path_handler()(self.gripperr_path)


class SapienVisualizer:
    def __init__(self, cfg: Optional[SapienViewerCfg] = None) -> None:
        if cfg is None:
            cfg = SapienViewerCfg()
        self._cfg = copy.deepcopy(cfg)

        self._engine = sapien.Engine()
        sapien.render_config.viewer_shader_dir = "rt"
        sapien.render_config.rt_samples_per_pixel = int(cfg.rt_samples_per_pixel)
        sapien.render_config.rt_use_denoiser = bool(cfg.rt_use_denoiser)
        self._renderer = sapien.SapienRenderer(device=cfg.device)
        self._engine.set_renderer(self._renderer)

        scene_config = sapien.SceneConfig()
        self._timestep = float(cfg.timestep)
        self._scene = self._engine.create_scene(scene_config)
        self._scene.set_timestep(self._timestep)
        self._default_step_n = cfg.default_step_n

        # ground
        material = self._renderer.create_material()
        material.base_color = [0., 0., 0., 1.]
        self._scene.add_ground(-0.5, render_material=material)

        # table
        builder = self._scene.create_actor_builder()
        material = self._renderer.create_material()
        material.base_color = [0., 0.5, 0., 1.]
        builder.add_box_visual(half_size=[0.5, 0.5, 0.2], material=material)
        self._table = builder.build_static(name="table")
        self._table.set_pose(pose=sapien.Pose(p=[0.5, 0.5, -0.2]))

        # background
        builder = self._scene.create_actor_builder()
        material = self._renderer.create_material()
        material.base_color = [0., 0., 0., 1.]
        builder.add_box_visual(half_size=[10., 1., 10.], material=material)
        self._background = builder.build_static(name="background")
        self._background.set_pose(pose=sapien.Pose(p=[0., 2., 0.]))

        # light
        self._scene.set_ambient_light([0.5, 0.5, 0.5])
        self._scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

        # viewer
        self._viewer = Viewer(self._renderer, resolutions=cfg.resolution)
        self._viewer.set_scene(self._scene)
        self._viewer.set_camera_xyz(x=0.5, y=-1.5, z=1.0)
        self._viewer.set_camera_rpy(r=0., p=-0.4, y=-np.pi / 2)
        self._viewer.set_fovy(1.0)

        # robotl
        loader: sapien.URDFLoader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self._robotl = loader.load_kinematic(cfg.urdfl_path)
        self._robotl_sapien_notfixed_jnames: list[str] = []
        for j in self._robotl.get_joints():
            if j.type != "fixed":
                self._robotl_sapien_notfixed_jnames.append(j.name)
        self._robotl.set_root_pose(sapien.Pose.from_transformation_matrix(ROBOTL_BASE))
        self._robotl_curr_qpos = self._robotl.get_qpos()

        # gripperl
        loader: sapien.URDFLoader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self._gripperl = loader.load_kinematic(cfg.gripperl_path)

        # robotr
        loader: sapien.URDFLoader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self._robotr = loader.load_kinematic(cfg.urdfr_path)
        self._robotr_sapien_notfixed_jnames: list[str] = []
        for j in self._robotr.get_joints():
            if j.type != "fixed":
                self._robotr_sapien_notfixed_jnames.append(j.name)
        self._robotr.set_root_pose(sapien.Pose.from_transformation_matrix(ROBOTR_BASE))
        self._robotr_curr_qpos = self._robotr.get_qpos()

        # gripperr
        builder = self._scene.create_actor_builder()
        material = self._renderer.create_material()
        material.base_color = [0.2, 0.2, 0.2, 1.]
        builder.add_visual_from_file(cfg.gripperr_path, material=material)
        self._gripperr = builder.build_kinematic(name="gripperr")
    
    def _update_robot_qpos(self, qpos_l: Optional[np.ndarray], qpos_r: Optional[np.ndarray]):
        if qpos_l is not None:
            self._robotl.set_qpos(qpos_l)
        if qpos_r is not None:
            self._robotr.set_qpos(qpos_r)
        self._scene.step()
        for l in self._robotl.get_links():
            if l.name == "tool0":
                self._gripperl.set_root_pose(l.get_pose())
        for l in self._robotr.get_links():
            if l.name == "ee":
                self._gripperr.set_pose(l.get_pose())

    def render(self):
        self._scene.update_render()
        self._viewer.render()
    
    def robot_setj(self, joints_l: Optional[np.ndarray]=None, joints_r: Optional[np.ndarray]=None):
        self._update_robot_qpos(joints_l, joints_r)
        self._scene.step()
        self._robotl_curr_qpos = self._robotl.get_qpos()
        self._robotr_curr_qpos = self._robotr.get_qpos()

    def robot_movej(self, joints_l: Optional[np.ndarray]=None, joints_r: Optional[np.ndarray]=None, step_n: int=None):
        if step_n is None:
            step_n = self._default_step_n
        
        prev_qpos_l = self._robotl_curr_qpos.copy()
        prev_qpos_r = self._robotr_curr_qpos.copy()

        if joints_l is None:
            joints_l = prev_qpos_l.copy()
        if joints_r is None:
            joints_r = prev_qpos_r.copy()

        self._robotl_curr_qpos = robotl_curr_qpos = joints_l.copy()
        self._robotr_curr_qpos = robotr_curr_qpos = joints_r.copy()

        for i in np.linspace(0., 1., step_n):
            qpos_l = robotl_curr_qpos * i + prev_qpos_l * (1. - i)
            qpos_r = robotr_curr_qpos * i + prev_qpos_r * (1. - i)
            self._update_robot_qpos(qpos_l, qpos_r)
            self._scene.step()
            self.render()


@dataclass
class SafetyCheckerLeft:
    rmin: float = 0.35
    rmax: float = 0.90
    zmin: float = 0.0
    zmax: float = 1.50

    def check_and_clip(self, xyz: np.ndarray, print_warn: bool = True) -> np.ndarray:
        x, y, z = xyz
        x0, y0 = ROBOTL_BASE[:2, 3]
        r = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5
        c = np.arctan2(y - y0, x - x0)
        if not (
            self.rmin <= r <= self.rmax and 
            self.zmin <= z <= self.zmax
        ):
            r = np.clip(r, self.rmin, self.rmax)
            x, y = x0 + r * np.cos(c), y0 + r * np.sin(c)
            z = np.clip(z, self.zmin, self.zmax)
            if print_warn:
                print(f"[WARN]: out of safety range: {xyz} {self}, clip to {np.array([x, y, z])}")
        
        return np.array([x, y, z])


@dataclass
class SafetyCheckerRight:
    rmin: float = 0.15
    rmax: float = 0.40
    zmin: float = 0.0
    zmax: float = 1.00

    def check_and_clip(self, xyz: np.ndarray, print_warn: bool = True) -> np.ndarray:
        x, y, z = xyz
        x0, y0 = ROBOTR_BASE[:2, 3]
        r = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5
        c = np.arctan2(y - y0, x - x0)
        if not (
            self.rmin <= r <= self.rmax and 
            self.zmin <= z <= self.zmax
        ):
            r = np.clip(r, self.rmin, self.rmax)
            x, y = x0 + r * np.cos(c), y0 + r * np.sin(c)
            z = np.clip(z, self.zmin, self.zmax)
            if print_warn:
                print(f"[WARN]: out of safety range: {xyz} {self}, clip to {np.array([x, y, z])}")
        
        return np.array([x, y, z])


@dataclass
class MoveXYZ:
    xyz: np.ndarray
    fix_rot: bool = True
    rot_mat: np.ndarray = field(default_factory=lambda: tra.euler_matrix(np.pi, 0., 0.)[:3, :3])

    @staticmethod
    def from_pose(pose: np.ndarray) -> "MoveXYZ":
        return MoveXYZ(pose[:3, 3], fix_rot=True, rot_mat=pose[:3, :3])


@dataclass
class RealEnvDesktopCfg:
    # misc
    skip_enter: bool = False
    use_viewer: bool = True
    use_sam: bool = True

    # robot
    robotl_acc: float = 1.0
    robotl_vel: float = 2.0
    robotr_speed: float = 0.3

    # cfg
    api_cfg: Optional[APICfg] = None
    api_robotl_cfg: Optional[RobotlCfg] = None
    api_robotr_cfg: Optional[RobotrCfg] = None
    api_camera_cfg: Optional[CameraCfg] = None
    viewer_cfg: Optional[SapienViewerCfg] = None
    sam_cfg: Optional[SAMCfg] = None


class RealEnvDesktop:
    def __init__(self, cfg: Optional[RealEnvDesktopCfg] = None):
        self._closed = False
        if cfg is None:
            cfg = RealEnvDesktopCfg()
        self._cfg = copy.deepcopy(cfg)

        self._api = RealAPIDesktop(
            api_cfg=cfg.api_cfg,
            robotl_cfg=cfg.api_robotl_cfg, 
            robotr_cfg=cfg.api_robotr_cfg,
            camera_cfg=cfg.api_camera_cfg,
        )
        if cfg.use_viewer:
            self._viewer = SapienVisualizer(cfg.viewer_cfg)
            self._viewer.robot_setj(
                joints_l=self._api.robotl_qpos_dict_to_ndarray(self._api.urdfl.cfg),
                joints_r=self._api.robotr_qpos_dict_to_ndarray(self._api.urdfr.cfg),
            )
            self._viewer.robot_movej()
        if cfg.use_sam:
            self._gs = GroundedSAM()
        
        self._safety_checker_l = SafetyCheckerLeft()
        self._safety_checker_r = SafetyCheckerRight()

        atexit.register(self._close)
    
    def _move_qpos(self, qpos_l: Optional[np.ndarray] = None, qpos_r: Optional[np.ndarray] = None):
        # move robotl first then robotr
        self._viewer.robot_movej(qpos_l, qpos_r)
        if qpos_l is not None:
            self._api.robotl_movej(qpos_l, acc=self._cfg.robotl_acc, vel=self._cfg.robotl_vel, skip_enter=self._cfg.skip_enter)
        if qpos_r is not None:
            self._api.robotr_movej(qpos_r, speed=self._cfg.robotr_speed, skip_enter=self._cfg.skip_enter)
        
    def _move_xyz_compute_ik(
        self, 
        xyz: MoveXYZ, 
        left_or_right: Literal["left", "right"],
        check_safety: bool,
        use_init_qpos: bool = False,
    ):
        urdf = dict(left=self._api.urdfl, right=self._api.urdfr)[left_or_right]
        robot_cfg = dict(left=self._api.robotl_cfg, right=self._api.robotr_cfg)[left_or_right]
        safety_checker = dict(left=self._safety_checker_l, right=self._safety_checker_r)[left_or_right]
        qpos_ndarray_to_dict = dict(left=self._api.robotl_qpos_ndarray_to_dict, right=self._api.robotr_qpos_ndarray_to_dict)[left_or_right]
        qpos_dict_to_ndarray = dict(left=self._api.robotl_qpos_dict_to_ndarray, right=self._api.robotr_qpos_dict_to_ndarray)[left_or_right]
        
        xyz = copy.deepcopy(xyz)
        if check_safety:
            xyz.xyz = safety_checker.check_and_clip(xyz.xyz)
        
        pose = np.eye(4)
        pose[:3, 3] = xyz.xyz
        if xyz.fix_rot:
            pose[:3, :3] = xyz.rot_mat
            mask = torch.ones(16, **self._api.urdf_factory_kwargs)
        else:
            mask = torch.tensor([0., 0., 0., 1.] * 2 + [1., 0., 0., 1.] + [0., 0., 0., 1.], **self._api.urdf_factory_kwargs) # (e'_x)_z = 0
        pose = torch.tensor(pose, **self._api.urdf_factory_kwargs) # [4, 4]

        def err_func(link_transform_map: dict[str, torch.Tensor]):
            curr_mat4 = link_transform_map[robot_cfg.ee_str].view(1, 16, 16) # [B, 16, 16]
            err_mat = curr_mat4[:, torch.arange(16), torch.arange(16)] - pose.view(1, 16) # [B, 16]
            return err_mat * mask
        def loss_func(link_transform_map: dict[str, torch.Tensor]):
            curr_mat4 = link_transform_map[robot_cfg.ee_str]
            err_mat = curr_mat4 - pose # [B, 16]
            return torch.sum(torch.square(err_mat).view(1, 16) * mask, dim=1) # [B, ]

        # inverse kinematics
        cfg, info = urdf.inverse_kinematics_optimize(
            err_func=err_func, loss_func=loss_func,
            init_cfg=qpos_ndarray_to_dict(np.array(robot_cfg.robot_joints_init_val)) if use_init_qpos else None, # use init qpos to compute ik 
            **self._api.api_cfg.ik_kwargs, 
        )
        print(f"[INFO] ik {info}")

        return qpos_dict_to_ndarray(cfg)

    def _move_to_xyz(self, xyz_l: Optional[MoveXYZ] = None, xyz_r: Optional[MoveXYZ] = None, check_safety: bool = True, use_init_qpos: bool = False):
        if xyz_l is not None:
            qpos_l = self._move_xyz_compute_ik(xyz_l, "left", check_safety, use_init_qpos=use_init_qpos)
        else:
            qpos_l = None
        
        if xyz_r is not None:
            qpos_r = self._move_xyz_compute_ik(xyz_r, "right", check_safety, use_init_qpos=use_init_qpos)
        else:
            qpos_r = None
        self._move_qpos(qpos_l, qpos_r)
    
    def _movel_robot_to_init(self):
        self._move_to_xyz(
            MoveXYZ.from_pose(self._api.robotl_cfg.robot_tcp_init_mat.copy()),
            MoveXYZ.from_pose(self._api.robotr_cfg.robot_tcp_init_mat.copy()),
            use_init_qpos=True
        )
    
    def _select_pick_hand(self, pick_xyz: np.ndarray):
        px, py, pz = pick_xyz
        if (px + py) < 0.9:
            return "left"
        else:
            return "right"
    
    def _get_obs(self):
        return self._api.take_picture()
    
    def _move_camera(self, camera_pose: np.ndarray, check_safety: bool = False):
        uree_pose = camera_pose @ np.linalg.inv(CAMERA_TO_UREE)
        self._move_to_xyz(xyz_l=MoveXYZ.from_pose(uree_pose), check_safety=check_safety)
    
    def _select_smallest_mask(self, masks: np.ndarray):
        # select the smallest mask
        return masks[np.sum(masks, axis=(1, 2)).argmin(), :, :]
    
    def _get_camera_pose_to_init_clothes(self, rot_z: float = 0.) -> np.ndarray:
        # camera_pose = tra.translation_matrix(np.array([0.46, 0.38, 1.10])) @ tra.euler_matrix(math.pi, 0., rot_z) # d435
        camera_pose = tra.translation_matrix(np.array([0.45, 0.40, 0.90])) @ tra.euler_matrix(math.pi, 0., rot_z) # d436
        return camera_pose
    
    def _close(self):
        if not self._closed:
            if hasattr(self, "_api"):
                self._api.close()
            self._closed = True
    
    def get_obs(self):
        return self._get_obs()
    
    def move_camera(self, camera_pose: np.ndarray):
        self._move_camera(camera_pose)

    def pick_and_place(self, pick_xyz: np.ndarray, displacement_xyz: np.ndarray, specify_hand: Literal["left", "right", "none"]=None):
        px, py, pz = pick_xyz
        dx, dy, dz = displacement_xyz
        hand = self._select_pick_hand(pick_xyz)
        if specify_hand == "left" or (hand == "left" and specify_hand == "none"):
            p1l, p1r = MoveXYZ(np.array([px, py, pz])), None
            p2l, p2r = MoveXYZ(np.array([px, py, pz + dz])), None
            p3l, p3r = MoveXYZ(np.array([px + dx, py + dy, pz + dz])), None
            p4l, p4r = MoveXYZ(np.array([px + dx, py + dy, pz])), None
            action_open = "open", "none"
            action_close = "close", "none"
        elif specify_hand == "right" or (hand == "right" and specify_hand == "none"):
            p1l, p1r = None, MoveXYZ(np.array([px, py, pz]), fix_rot=False)
            p2l, p2r = None, MoveXYZ(np.array([px, py, pz + dz]), fix_rot=False)
            p3l, p3r = None, MoveXYZ(np.array([px + dx, py + dy, pz + dz]), fix_rot=False)
            p4l, p4r = None, MoveXYZ(np.array([px + dx, py + dy, pz]), fix_rot=False)
            action_open = "none", "open"
            action_close = "none", "close"
        else:
            raise ValueError(f"Invalid hand: {hand} {specify_hand}")

        # pick and place
        self._movel_robot_to_init()
        self._move_to_xyz(xyz_l=p2l, xyz_r=p2r)
        self._move_to_xyz(xyz_l=p1l, xyz_r=p1r)
        self._api.gripper_action(*action_close)
        self._move_to_xyz(xyz_l=p2l, xyz_r=p2r)
        self._move_to_xyz(xyz_l=p3l, xyz_r=p3r)
        self._move_to_xyz(xyz_l=p4l, xyz_r=p4r)
        self._api.gripper_action(*action_open)
        self._move_to_xyz(xyz_l=p3l, xyz_r=p3r)
        self._movel_robot_to_init()
    
    def get_random_init_clothes_xyz(
        self, 
        camera_pose: np.ndarray, 
        disp_min_radius: float = 0.05, disp_max_radius: float = 0.10, 
        disp_min_height: float = 0.05, disp_max_height: float = 0.10, 
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        if move_camera_first is True, the camera will be moved to the initial position first.
        
        otherwise you need to call move_camera_to_init_clothes() manually.
        """
        obs_height = camera_pose[2, 3]
        obs = self._get_obs()

        app = human_aabb_app.App(sam=self._gs, img=Image.fromarray(obs["color"]), cfg=human_aabb_app.AppCfg())
        mask = app.run()

        # random select pick point
        is_positive = np.argwhere(mask > 0.5)
        while True:
            i1, j1 = is_positive[np.random.choice(len(is_positive))]
            i2, j2 = is_positive[np.random.choice(len(is_positive))]
            p1 = camera_pose[:3, :3] @ real_utils.pixel_to_xyz(i1, j1, obs_height - self._safety_checker_l.zmin, INTRINSICS) + camera_pose[:3, 3]
            p2 = camera_pose[:3, :3] @ real_utils.pixel_to_xyz(i2, j2, obs_height - self._safety_checker_l.zmin, INTRINSICS) + camera_pose[:3, 3]
            p2[2] = p1[2] + np.random.uniform(disp_min_height, disp_max_height)
            xyz_displacement = p2 - p1
            if (
                disp_min_radius < np.linalg.norm(xyz_displacement[:2]) < disp_max_radius and
                np.allclose(self._safety_checker_l.check_and_clip(p1, print_warn=False), p1) and
                np.allclose(self._safety_checker_l.check_and_clip(p2, print_warn=False), p2)
            ):
                # xyz_world_frame is safe and within the workspace
                break
        
        # visualize result
        export_img = obs["color"].copy()
        export_img[np.where(mask > 0.5)] = [255, 255, 255]
        export_img = Image.fromarray(export_img)
        draw = ImageDraw.Draw(export_img)
        radius = 5
        j, i = real_utils.xyz_to_pixel(camera_pose[:3, :3].T @ (p1 - camera_pose[:3, 3]), INTRINSICS)
        draw.ellipse((j - radius, i - radius, j + radius, i + radius), fill="red")
        j, i = real_utils.xyz_to_pixel(camera_pose[:3, :3].T @ (p2 - camera_pose[:3, 3]), INTRINSICS)
        draw.ellipse((j - radius, i - radius, j + radius, i + radius), fill="pink")
        real_utils.vis.show(np.array(export_img), "clothes_mask")

        return p1, p2, "left"
    
    def move_camera_to_init_clothes(self, rot_z: float = 0.) -> np.ndarray:
        camera_pose = self._get_camera_pose_to_init_clothes(rot_z=rot_z)
        self._move_camera(camera_pose)
        return camera_pose
    
    def get_camera_pose_to_take_picture(
        self, 
        rot_x: Optional[float] = None, 
        rot_y: float = 0., 
        rot_z: Optional[float] = None,
        rot_x_range: tuple[float, float] = (0.15, 0.25), 
        rot_z_range: tuple[float, float] = (0., math.pi / 3),
    ) -> np.ndarray:
        if rot_x is None:
            rot_x = np.random.uniform(*rot_x_range)
        rot_x = np.clip(rot_x, *rot_x_range)
        if rot_z is None:
            rot_z = np.random.uniform(*rot_z_range)
        rot_z = np.clip(rot_z, *rot_z_range)
        print(f"[INFO] random camera pose: rot_x={rot_x}, rot_y={rot_y}, rot_z={rot_z}")
        rotation = tra.euler_matrix(math.pi + rot_x * 0.9, rot_y, -rot_z)
        # translation = np.array([
        #     0.35 - np.sin(rot_z) * np.sin(rot_x) * 1.2,
        #     0.45 - np.cos(rot_z) * np.sin(rot_x) * 1.2,
        #     np.cos(rot_x) * 1.2
        # ]) # d435
        translation = np.array([
            0.40 - np.sin(rot_z) * np.sin(rot_x) * 0.9,
            0.45 - np.cos(rot_z) * np.sin(rot_x) * 0.9,
            np.cos(rot_x) * 0.9
        ]) # d436
        return tra.translation_matrix(translation) @ rotation
    
    def move_robot_to_init(self):
        self._movel_robot_to_init()

    def close(self):
        self._close()


if __name__ == "__main__":
    env = RealEnvDesktop(RealEnvDesktopCfg(
        api_robotl_cfg=RobotlCfg(use_gripper=False, default_acc=0.5, default_vel=0.5),
        api_robotr_cfg=RobotrCfg(use=False),
        use_sam=False,
        # api_robotr_cfg=RobotrCfg(use_gripper=False, default_speed=0.3),
        # api_robotl_cfg=RobotlCfg(use_gripper=False),
        # api_robotr_cfg=RobotrCfg(use_gripper=False),
    ))
    camera_pose = env.get_random_camera_pose_to_take_picture()
    env.move_camera(camera_pose)
    env.close()