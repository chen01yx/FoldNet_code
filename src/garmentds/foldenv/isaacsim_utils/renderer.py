from isaacsim.core.api.materials import VisualMaterial
from isaacsim.core.prims import SingleRigidPrim, SingleGeometryPrim, SingleXFormPrim
import isaacsim.core.utils.prims as isaacsim_prims
import isaacsim.core.utils.stage as isaacsim_stage
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.api.world.world import World
from isaacsim.sensors.camera import Camera
import omni.kit.commands
import omni.usd
from pxr import UsdShade, Sdf, Gf, Tf, Usd, UsdGeom
import Semantics
import carb
import isaacsim.core.utils.numpy.rotations as isaacsim_rotation

from src.garmentds.foldenv.isaacsim_utils.cuboids import FixedCuboid

import os
import copy
import random
from dataclasses import dataclass
from typing import Literal, Optional
import json
import asyncio
import math

import numpy as np
import trimesh.transformations as tra
import trimesh


@dataclass
class MaterialInfo:
    path: str
    name: str


@dataclass
class TableGroundCfg:
    scale: np.ndarray
    rotation_euler: np.ndarray
    location: np.ndarray
    table_material: MaterialInfo
    ground_material: MaterialInfo

    def __post_init__(self):
        self.scale = np.array(self.scale)
        self.rotation_euler = np.array(self.rotation_euler)
        self.location = np.array(self.location)


@dataclass
class ClothCfg:
    cloth_obj_path: str


@dataclass
class CameraCfg:
    cameras_name: list[Literal["head", "side"]]
    camera_rand_int_dict: dict[str, np.ndarray]
    camera_rand_ext_dict: dict[str, np.ndarray]


@dataclass
class LightCfg:
    position: np.ndarray
    orientation: np.ndarray
    spacing_x: float
    spacing_y: float
    num_x: int
    num_y: int
    radius: float
    length: float
    intensity: float
    color_temperature: float


@dataclass
class RoomCfg:
    input_folder: str
    output_dir: str
    step_idx: int


class SceneGenerator:
    def __init__(
        self,
        asset_base_dir: str,
        split_group: Literal["train", "test"],
        cameras_name: list[Literal["head", "side"]],
        cloth_obj_path: str, 
    ):
        self.asset_base_dir = asset_base_dir
        material_split = np.load("data/asset/isaacsim/data/vMaterials_2/filtered_material_split.npy", allow_pickle=True).item()
        self.table_materials_infos = material_split[split_group]['table']
        self.ground_materials_infos = material_split[split_group]['ground']

        print(f"{split_group} mode, found {len(self.table_materials_infos)} materials for table and {len(self.ground_materials_infos)} materials for ground")

        self.cameras_name = cameras_name
        self.randomize_camera = True

        self.cloth_obj_path = cloth_obj_path
    
    def _generate_material_info(self, infos: list[dict[str, str]]):
        info = random.choice(infos)
        return MaterialInfo(path=os.path.join(self.asset_base_dir, info["path"]), name=info["name"])
    
    def _generate_table_ground_cfg(self):
        uni_f = random.uniform
        rotation_euler, scale = [
            ((0., 0., uni_f(-0.1, 0.1), ), (uni_f(0.5, 0.8), uni_f(0.3, 0.5), uni_f(0.02, 0.04))),
            ((0., 0., uni_f(-0.1, 0.1) + math.pi, ), (uni_f(0.5, 0.8), uni_f(0.3, 0.5), uni_f(0.02, 0.04))),
            ((0., 0., uni_f(-0.1, 0.1) + math.pi / 2, ), (uni_f(0.3, 0.5), uni_f(0.5, 0.8), uni_f(0.02, 0.04))),
            ((0., 0., uni_f(-0.1, 0.1) - math.pi / 2, ), (uni_f(0.3, 0.5), uni_f(0.5, 0.8), uni_f(0.02, 0.04))),
        ][random.randint(0, 3)]
        location = np.random.uniform(-0.02, +0.02, 2).tolist() + [-scale[2]]
        return TableGroundCfg(
            scale, rotation_euler, location,
            self._generate_material_info(self.table_materials_infos), 
            self._generate_material_info(self.ground_materials_infos),
        )

    def _generate_cloth_cfg(self):
        return ClothCfg(self.cloth_obj_path)
    
    def _generate_camera_cfg(self):
        camera_rand_ext_dict, camera_rand_int_dict = dict(), dict()
        for name in self.cameras_name:
            if self.randomize_camera:
                vec = np.random.randn(3)
                vec /= max(np.linalg.norm(vec), 1e-6)
                camera_rand_ext_dict[name] = tra.translation_matrix(np.random.uniform(-0.02, 0.02, 3)) @ tra.rotation_matrix(np.random.uniform(0.0, 0.05), vec)
                camera_rand_int_dict[name] = np.array([
                    np.random.uniform(-5., +5.), 0., np.random.uniform(-5., +5.),
                    0., np.random.uniform(-5., +5.), np.random.uniform(-5., +5.),
                    0., 0., 0.
                ]).reshape(3, 3)
            else:
                camera_rand_ext_dict[name] = np.eye(4)
                camera_rand_int_dict[name] = np.zeros((3, 3))
        
        return CameraCfg(self.cameras_name, camera_rand_int_dict, camera_rand_ext_dict)
    
    def _generate_light_cfg(self):
        light_num = (2, 3)
        light_color_temperature = np.random.uniform(4000, 8000)
        light_intensity = np.random.uniform(4e4, 1.2e5)
        light_radius = np.random.uniform(0.05, 0.1)
        light_length = np.random.uniform(0.5, 2.0)
        light_spacing = np.random.uniform((1.0, 2.5 * light_length), (1.0, 2.5 * light_length))
        light_position = np.random.uniform((-1.0, -1.0, 2.0), (1.0, 1.0, 4.0))
        light_orientation = isaacsim_rotation.euler_angles_to_quats(np.array([0.0, 0.0, np.random.uniform(-0.5 * np.pi, 0.5 * np.pi)]), degrees=False)

        return LightCfg(
            light_position, light_orientation, light_spacing[0], light_spacing[1], light_num[0], light_num[1],
            light_radius, light_length, light_intensity, light_color_temperature
        )

    def generate_init_args(self) -> dict:
        table_ground_cfg = self._generate_table_ground_cfg()
        cloth_cfg = self._generate_cloth_cfg()
        camera_cfg = self._generate_camera_cfg()
        light_cfg = self._generate_light_cfg()
        return dict(
            table_ground_cfg=table_ground_cfg, 
            cloth_cfg=cloth_cfg, 
            camera_cfg=camera_cfg,
            light_cfg=light_cfg,
        )


async def convert_asset_to_usd(input_obj: str, output_usd: str):
    import omni.kit.asset_converter

    def progress_callback(progress, total_steps):
        pass

    converter_context = omni.kit.asset_converter.AssetConverterContext()
    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(input_obj, output_usd, progress_callback, converter_context)
    success = await task.wait_until_finished()
    if not success:
        carb.log_error(task.get_status(), task.get_detailed_error())
    print("converting done")


class IsaacRoom:
    def __init__(
        self, 
        room_prim_path: str,
        position: list,
        world: World,
    ):
        room = SingleXFormPrim(
            prim_path=room_prim_path,
            translation=position,
            orientation=[1, 0, 0, 0],
        )
        workspace_path = f"{room_prim_path}/workspace"
        workspace = SingleXFormPrim(prim_path=workspace_path)

        self.world = world
        self.room = room
        self.workspace = workspace
        self.room_prim_path = room_prim_path
        self.workspace_prim_path = workspace_path
    
    def init_room(
        self, 
        table_ground_cfg: TableGroundCfg,
        cloth_cfg: ClothCfg,
        camera_cfg: CameraCfg,
        light_cfg: LightCfg,
        active_materials: list[str], 
        active_shaders: list[UsdShade.Shader],
    ):
        self._init_table_ground(table_ground_cfg, active_materials, active_shaders)
        self._init_robot()
        self._init_cloth(cloth_cfg)
        self._init_cameras(camera_cfg)
        self._init_lights(light_cfg)
        return
    
    def update_state(self, state: RoomCfg):
        with open(os.path.join(state.input_folder, "cfg.json"), "r") as f:
            robot_cfg = json.load(f)
        
        self.robot.set_joint_positions([robot_cfg["qpos"][name] for name in self.robot.dof_names])
        self.robot.set_local_pose(np.array(robot_cfg["base_pose"])[:3, 3], tra.quaternion_from_matrix(np.array(robot_cfg["base_pose"])))

        cloth_v_sim: np.ndarray = np.load(os.path.join(state.input_folder, "cloth_v_sim.npy"))
        assert cloth_v_sim.shape == self.tm_mesh_sim.vertices.shape, f"cloth_obj_file might be wrong: {cloth_v_sim.shape} {self.tm_mesh_sim.vertices.shape}"
        self.tm_mesh_sim.vertices = cloth_v_sim
        v_np = self.tm_mesh_sim.vertices[self.vert_ren_to_sim]
        vn_np = self.tm_mesh_sim.vertex_normals[self.vert_ren_to_sim]
        self.cloth_visual_mesh.GetPointsAttr().Set(v_np)
        self.cloth_visual_mesh.GetNormalsAttr().Set(vn_np[self.tm_mesh_sim.faces.flatten()])

        for camera, name in zip(self.cameras, self.cameras_name):
            if name == "side":
                tf = tra.translation_matrix([1.2, 1.2, 1.]) @ tra.euler_matrix(1.1, 0., np.pi / 4 * 3) @ self.camera_rand_ext_dict[name]
                camera.set_local_pose(tf[:3, 3], tra.quaternion_from_matrix(tf), camera_axes="usd")
            elif name == "head":
                tf = tra.euler_matrix(np.pi / 2, 0., -np.pi / 2) @ self.camera_rand_ext_dict[name]
                camera.set_local_pose(tf[:3, 3], tra.quaternion_from_matrix(tf), camera_axes="usd")
            else:
                raise ValueError(name)

    def get_render_products(self) -> list[dict[str, np.ndarray]]:
        render_products = []
        for camera in self.cameras:
            sem_seg_raw = camera._custom_annotators["semantic_segmentation"].get_data()
            label_idx = list(sem_seg_raw["info"]["idToLabels"].values()).index({"class":"cloth"})
            label: tuple[float] = eval(list(sem_seg_raw["info"]["idToLabels"].keys())[label_idx])
            sem_seg = np.all(sem_seg_raw["data"] == label, axis=-1)
            render_products.append(dict(
                rgba=camera.get_rgba().reshape((480, 640, 4)),
                semantic_segmentation=sem_seg,
            ))
        return render_products
    
    def __create_material(self, material_info: MaterialInfo, active_materials: list[str], active_shaders: list[UsdShade.Shader]):
        success, result = omni.kit.commands.execute(
            "CreateMdlMaterialPrimCommand",
            mtl_url=material_info.path,
            mtl_name=material_info.name,
            mtl_path=f"/Looks/{material_info.name}",
        )
        if not success:
            raise RuntimeError(f"failed to create material {material_info.name} at /Looks/{material_info.name}")
        
        usd_stage = omni.usd.get_context().get_stage()
        material = UsdShade.Material(usd_stage.GetPrimAtPath(f"/Looks/{material_info.name}"))
        shader = UsdShade.Shader(omni.usd.get_shader_from_material(material, get_prim=True))

        active_materials.append(material_info.name)
        active_shaders.append(shader)
    
    def __create_omnipbr(self, prim_path: str):
        success, result = omni.kit.commands.execute(
            "CreateMdlMaterialPrimCommand",
            mtl_url="OmniPBR.mdl",
            mtl_name="OmniPBR",
            mtl_path=prim_path,
        )
        if not success:
            raise RuntimeError(f"failed to create OmniPBR at {prim_path}")

    def __apply_visual_material(self, prim: FixedCuboid, material_name: str):
        stage = omni.usd.get_context().get_stage()
        material = UsdShade.Material(stage.GetPrimAtPath(f"/Looks/{material_name}"))
        shader = UsdShade.Shader(omni.usd.get_shader_from_material(material, get_prim=True))
        visual_material = VisualMaterial(
            prim_path=f"/Looks/{material_name}",
            prim=stage.GetPrimAtPath(f"/Looks/{material_name}"),
            shaders_list=[shader],
            material=material,
            name=material_name,
        )
        prim.apply_visual_material(visual_material)
    
    def _init_table_ground(
        self, 
        table_ground_cfg: TableGroundCfg, 
        active_materials: list[str], 
        active_shaders: list[UsdShade.Shader],
    ):
        room_prim_path = self.room_prim_path
        workspace_prim_path = self.workspace_prim_path
        ground_material = table_ground_cfg.ground_material
        table_material = table_ground_cfg.table_material

        # ground
        ground_edge = 10.0
        ground = FixedCuboid(
            prim_path=f"{workspace_prim_path}/Ground",
            scale=np.array([ground_edge, ground_edge, 0.01]),
        )
        ground.set_local_pose(np.array([0, 0, -0.455]), np.array([1, 0, 0, 0]))
        self.__create_omnipbr(f"{room_prim_path}/Looks/Ground")
        if ground_material.name not in active_materials:
            self.__create_material(ground_material, active_materials, active_shaders)
        self.__apply_visual_material(ground, ground_material.name)

        self.ground = ground

        # table
        table = FixedCuboid(
            prim_path=f"{workspace_prim_path}/Table",
            scale=table_ground_cfg.scale * 2,
        )
        table.set_local_pose(table_ground_cfg.location, tra.quaternion_from_euler(*table_ground_cfg.rotation_euler))
        self.__create_omnipbr(f"{room_prim_path}/Looks/Table")
        if table_material.name not in active_materials:
            self.__create_material(table_material, active_materials, active_shaders)
        self.__apply_visual_material(table, table_material.name)

        self.table = table
        print("init table ground done.")
    
    def _init_robot(self):
        workspace_prim_path = self.workspace_prim_path
        isaacsim_stage.add_reference_to_stage(
            usd_path="asset/galbot_one_charlie/urdf.usd",
            prim_path=f"{workspace_prim_path}/Robot",
        )
        self.robot = Robot(prim_path=f"{workspace_prim_path}/Robot")
        self.robot.initialize()
        print("init robot done.")

    def _init_cloth(self, cloth_cfg: ClothCfg):
        workspace_prim_path = self.workspace_prim_path
        cloth_obj_path = cloth_cfg.cloth_obj_path

        self.tm_mesh_raw: trimesh.Trimesh = trimesh.load(cloth_obj_path, process=False)
        self.tm_mesh_sim = trimesh.Trimesh(vertices=self.tm_mesh_raw.vertices, faces=self.tm_mesh_raw.faces)
        vert_xyz_to_idx = {}
        for i, v in enumerate(self.tm_mesh_sim.vertices):
            vert_xyz_to_idx[tuple(v)] = i
        vert_ren_to_sim = []
        for i, v in enumerate(self.tm_mesh_raw.vertices):
            vert_ren_to_sim.append(vert_xyz_to_idx[tuple(v)])
        self.vert_ren_to_sim = np.array(vert_ren_to_sim)
        """[NV_R] -> [0, NV_S)"""

        usd_file = cloth_obj_path.replace(".obj", "_usd/cloth.usd")
        if not os.path.exists(usd_file):
            asyncio.get_event_loop().run_until_complete(convert_asset_to_usd(cloth_obj_path, usd_file))

        stage = omni.usd.get_context().get_stage()
        cloth_visual_prim = isaacsim_stage.add_reference_to_stage(
            usd_path=usd_file,
            prim_path=f"{workspace_prim_path}/Cloth/cloth_visual",
        )
        self.cloth_visual_mesh: UsdGeom.Mesh = UsdGeom.Mesh.Get(stage, f"{workspace_prim_path}/Cloth/cloth_visual/mesh/mesh")

        material_path = f"{workspace_prim_path}/Cloth/cloth_visual/Looks/material_0/material_0"
        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        texture_path = os.path.join(os.path.dirname(usd_file), "textures/material_0.png")
        material.GetInput("diffuse_texture").Set(Sdf.AssetPath(texture_path))

        semantic_type, semantic_value = "class", "cloth"
        instance_name = Tf.MakeValidIdentifier(f"{semantic_type}_{semantic_value}")
        sem = Semantics.SemanticsAPI.Apply(cloth_visual_prim, instance_name)
        sem.CreateSemanticTypeAttr()
        sem.CreateSemanticDataAttr()
        sem.GetSemanticTypeAttr().Set(semantic_type)
        sem.GetSemanticDataAttr().Set(semantic_value)
        print("init cloth done.")
    
    def _init_cameras(
        self, 
        camera_cfg: CameraCfg,
    ):
        workspace_prim_path = self.workspace_prim_path
        cameras_name = camera_cfg.cameras_name
        camera_rand_ext_dict = camera_cfg.camera_rand_ext_dict
        camera_rand_int_dict = camera_cfg.camera_rand_int_dict

        fx = 388.
        fy = 388.
        focal_length = 2.
        resolution_x = 640
        resolution_y = 480

        cameras: list[Camera] = []
        self.camera_rand_ext_dict = camera_rand_ext_dict
        self.camera_rand_int_dict = camera_rand_int_dict
        for name in cameras_name:
            if name == "head":
                camera = Camera(
                    prim_path=f"{workspace_prim_path}/Robot/head_camera_sim_view_frame/Camera",
                    resolution=(resolution_x, resolution_y),
                )
            elif name == "side":
                camera = Camera(
                    prim_path=f"{workspace_prim_path}/Camera",
                    resolution=(resolution_x, resolution_y),
                )
            else:
                raise ValueError(name)
            
            camera.initialize()
            camera.add_semantic_segmentation_to_frame(dict(colorize=True))

            horizontal_aperture = focal_length * resolution_x / (fx + self.camera_rand_int_dict[name][0, 0])
            vertical_aperture = focal_length * resolution_y / (fy + self.camera_rand_int_dict[name][1, 1])
            camera.set_clipping_range(0.1, 10.0)
            camera.set_focal_length(focal_length)
            camera.set_horizontal_aperture(horizontal_aperture)
            camera.set_vertical_aperture(vertical_aperture)
            cameras.append(camera)
            
        self.cameras = cameras
        self.cameras_name = copy.deepcopy(cameras_name)

        print("init cameras done.")
    
    def _init_lights(self, light_cfg: LightCfg):
        workspace_prim_path = self.workspace_prim_path

        light_center = SingleXFormPrim(f"{workspace_prim_path}/LightCenter")
        lights: list[tuple[Usd.Prim, int, int]] = []
        for row in range(light_cfg.num_y):
            for col in range(light_cfg.num_x):
                prim_path = f"{workspace_prim_path}/LightCenter/Light{row}_{col}"
                light = isaacsim_prims.create_prim(prim_path, "CylinderLight")
                lights.append([light, row, col])
        
        light_center.set_local_pose(light_cfg.position, light_cfg.orientation)
        for light in lights:
            light_prim = light[0]
            row = light[1]
            col = light[2]
            light_prim.GetAttribute("xformOp:translate").Set((
                light_cfg.spacing_x * (col - (light_cfg.num_x - 1) / 2), 
                light_cfg.spacing_y * (row - (light_cfg.num_y - 1) / 2), 
                0.
            ))
            light_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(1., 0., 0., 0.))
            light_prim.GetAttribute("inputs:radius").Set(light_cfg.radius)
            light_prim.GetAttribute("inputs:length").Set(light_cfg.length)
            light_prim.GetAttribute("inputs:intensity").Set(light_cfg.intensity)
            light_prim.GetAttribute("inputs:enableColorTemperature").Set(True)
            light_prim.GetAttribute("inputs:colorTemperature").Set(light_cfg.color_temperature)
        print("init lights done.")


class IsaacRenderer:
    def __init__(
        self,
        num_rooms: int, 
        render_hz: int,
        render_mode: str, 
    ):
        self.active_tableground_materials: list[str] = []
        self.active_tableground_shaders: list[UsdShade.Shader] = []
        self.render_hz = render_hz

        self.set_render_mode(render_mode)
        self.world = World(physics_dt=1 / render_hz, rendering_dt=1 / render_hz, backend="numpy")
        self.world.get_physics_context().set_gravity(0.0)
        self.world.reset()
        self.rooms: list[IsaacRoom] = []
        house_edge_length = int(np.sqrt(num_rooms)) + 1
        for room_id in range(num_rooms):
            room_prim_path = f"/World/rooms/room_{room_id}"
            room_position = [10.0*(room_id%house_edge_length), 10.0*(room_id//house_edge_length), 0.0]
            room = IsaacRoom(room_prim_path, room_position, self.world)
            self.rooms.append(room)
        
        self.randomize_camera = True
        self.camera_rand_ext_dict = dict()
        self.camera_rand_int_dict = dict()
    
    def set_render_mode(self, render_mode: str):
        carb_settings = carb.settings.get_settings()
        if render_mode == "rtx_default":
            carb_settings.set("/rtx/directLighting/sampledLighting/enable", True)
        elif render_mode == "rtx_lowest":
            carb_settings.set("/rtx/post/dlss/execMode", 0)
            carb_settings.set("/rtx/newDenoiser/enabled", False)
            carb_settings.set("/rtx/shadows/enabled", False)
            carb_settings.set("/rtx/directLighting/sampledLighting/enabled", False)
            carb_settings.set("/rtx/directLighting/sampledLighting/autoEnable", False)
            carb_settings.set("/rtx/directLighting/domeLight/enabled", False)
            carb_settings.set("/rtx/indirectDiffuse/enabled", False)
            carb_settings.set("/rtx/ambientOcclusion/enabled", False)
            carb_settings.set("/rtx/reflections/enabled", False)
            carb_settings.set("/rtx/translucency/enabled", False)
            carb_settings.set("/rtx/sceneDb/ambientLightIntensity", 0.5)
        elif render_mode == "rtx_low":
            carb_settings.set("/rtx/post/dlss/execMode", 0)
            carb_settings.set("/rtx/newDenoiser/enabled", False)
            carb_settings.set("/rtx/shadows/enabled", True)
            carb_settings.set("/rtx/directLighting/sampledLighting/enabled", True)
            carb_settings.set("/rtx/directLighting/sampledLighting/autoEnable", False)
            carb_settings.set("/rtx/indirectDiffuse/enabled", False)
            carb_settings.set("/rtx/ambientOcclusion/enabled", False)
            carb_settings.set("/rtx/reflections/enabled", False)
            carb_settings.set("/rtx/translucency/enabled", False)
            carb_settings.set("/rtx/sceneDb/ambientLightIntensity", 0.5)
        elif render_mode == "path_tracing":
            carb_settings.set("/rtx/pathtracing/spp", 64)
            carb_settings.set("/rtx/pathtracing/totalSpp", 64)
        else:
            raise ValueError(render_mode)
    
    def init_single_room(
        self, 
        room_id: int, 
        table_ground_cfg: TableGroundCfg, 
        cloth_cfg: ClothCfg, 
        camera_cfg: CameraCfg, 
        light_cfg: LightCfg,
    ):
        self.rooms[room_id].init_room(
            table_ground_cfg, cloth_cfg, camera_cfg, light_cfg, 
            self.active_tableground_materials, self.active_tableground_shaders
        )
    
    def update_states(self, states: list[Optional[RoomCfg]]):
        for room, room_state in zip(self.rooms, states):
            if room_state is not None:
                room.update_state(room_state)
    
    def get_cam_param(self, room_idx: int, name: Literal["head", "side"]):
        room = self.rooms[room_idx]
        camera = room.cameras[room.cameras_name.index(name)]
        tw, qw = camera.get_world_pose(camera_axes="usd")
        tr, qr = room.room.get_world_pose()
        e = np.linalg.inv(tra.translation_matrix(tr) @ tra.quaternion_matrix(qr)) @ (tra.translation_matrix(tw) @ tra.quaternion_matrix(qw))
        i = camera.get_intrinsics_matrix()
        return dict(camera_extrinsics=e.tolist(), camera_intrinsics=i.tolist())