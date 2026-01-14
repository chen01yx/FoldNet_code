from dataclasses import dataclass
from typing import Optional, Union
import copy
import numpy as np

import sapien.core as sapien
from sapien.utils.viewer import Viewer


import garmentds.common.utils as utils
from garmentds.foldenv.fold_env import RobotCfg

@dataclass
class SapienViewerCfg:
    rt_samples_per_pixel: int = 4
    rt_use_denoiser: bool = True
    timestep: float = 0.01
    resolution: tuple[float, float] = (1280, 960)
    default_step_n: int = 20
    device: str = "cuda:0"


class SapienVisualizer:
    def __init__(self, cfg: Optional[SapienViewerCfg] = None, robot_cfg: Optional[RobotCfg] = None) -> None:
        if cfg is None:
            cfg = SapienViewerCfg()
        self._cfg = copy.deepcopy(cfg)
        if robot_cfg is None:
            robot_cfg = RobotCfg()
        self._robot_cfg = copy.deepcopy(robot_cfg)

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
        builder.add_box_visual(half_size=[0.5, 0.4, 0.2], material=material)
        self._table = builder.build_static(name="table")
        self._table.set_pose(pose=sapien.Pose(p=[0., 0., -0.2]))

        # light
        self._scene.set_ambient_light([0.5, 0.5, 0.5])
        self._scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

        # robot
        loader: sapien.URDFLoader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self._robot = loader.load_kinematic(robot_cfg.urdf_path)
        self._robot_sapien_notfixed_jnames: list[str] = []
        for j in self._robot.get_joints():
            if j.type != "fixed":
                self._robot_sapien_notfixed_jnames.append(j.name)
        self._robot.set_root_pose(sapien.Pose(p=robot_cfg.base_pos[0:3], q=robot_cfg.base_pos[3:7]))
        self._robot_setj(robot_cfg.init_qpos)
        self._robot_curr_qpos = self._robot.get_qpos()
        
        # viewer
        self._viewer = Viewer(self._renderer, resolutions=cfg.resolution)
        self._viewer.set_scene(self._scene)
        self._viewer.set_camera_xyz(x=0.0, y=+1.5, z=1.0)
        self._viewer.set_camera_rpy(r=0., p=-0.5, y=np.pi * 0.5)
        self._viewer.set_fovy(1.0)
        self._viewer.update_coordinate_axes_scale(0.3)
        
        print(f"test render urdf: {robot_cfg.urdf_path}")
        self.render()
        print("test render done")
    
    def _init_action_pc(self, action_pc: np.ndarray):
        self._action_point: list[sapien.ActorStatic] = []
        for idx, (x, y, z, r, g, b) in enumerate(action_pc):
            builder = self._scene.create_actor_builder()
            material = self._renderer.create_material()
            material.base_color = [1., 1., 1., 1.]
            builder.add_sphere_visual(radius=0.01, material=material)
            action_point = builder.build_static(name="action_point")
            action_point.set_pose(pose=sapien.Pose(p=(x, y, z)))
            self._action_point.append(action_point)
    
    def set_action_pc_to_visualize(self, action_pc: np.ndarray):
        if hasattr(self, "_action_point"):
            for action_point, (x, y, z, r, g, b) in zip(self._action_point, action_pc):
                action_point.set_pose(pose=sapien.Pose(p=(x, y, z)))
        else:
            self._init_action_pc(action_pc)
    
    def _qpos_dict_to_array(self, qpos: dict[str, float]):
        return np.array([qpos[j] for j in self._robot_sapien_notfixed_jnames], dtype=np.float32)

    def render(self):
        self._scene.update_render()
        self._viewer.render()
    
    def _robot_setj(self, qpos: Union[np.ndarray, dict[str, float]]):
        if isinstance(qpos, dict):
            qpos = self._qpos_dict_to_array(qpos)
        
        self._robot.set_qpos(qpos)
        self._scene.step()
        self._robot_curr_qpos = self._robot.get_qpos()

    def robot_movej(self, qpos: Optional[Union[np.ndarray, dict[str, float]]]=None, step_n: int=None):
        if step_n is None:
            step_n = self._default_step_n
        
        if isinstance(qpos, dict):
            qpos = self._qpos_dict_to_array(qpos)
        prev_qpos = self._robot_curr_qpos.copy()
        if qpos is None:
            qpos = prev_qpos.copy()

        self._robot_curr_qpos = robot_curr_qpos = qpos.copy()

        for i in np.linspace(0., 1., step_n):
            qpos = robot_curr_qpos * i + prev_qpos * (1. - i)
            self._robot.set_qpos(qpos)
            self._scene.step()
            self.render()