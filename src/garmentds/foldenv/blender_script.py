import bpy # type: ignore
import bmesh # type: ignore

import json
import os
import psutil
import signal
from dataclasses import dataclass
from typing import Optional
import tempfile
import pprint
import logging
import argparse
import sys
import random
from collections import defaultdict
import math

import numpy as np
import trimesh.transformations as tra
import batch_urdf
import torch
import trimesh
import bpycv
import cv2
from PIL import Image, ImageEnhance

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".cache", "blender"))
HDRI_CACHE_DIR = os.path.join(CACHE_DIR, "hdri")
TEXTURE_CACHE_DIR = os.path.join(CACHE_DIR, "texture")


USE_HDRI_AND_REMOVE_LIGHT = True
USE_POLYHAVEN_TEXTURES = True
RANDOMIZE_CAMERA = True
SKIP_ROBOT_FOR_QUICK_DEBUG = False


@dataclass
class Cfg:
    input_folder: str
    output_png: str
    mode: str
    picker_xyz: list[Optional[list[float]]]
    save_npy_img: bool
    save_blend_file: bool
    target_pid: int
    send_sigusr1: bool = True


def set_camera(
    camera_extrinsics: np.ndarray, 
    camera_intrinsics: np.ndarray, 
    width: int, 
    height: int, 
    sensor_width_mm: float=36.
):
    fx, fy, cx, cy = camera_intrinsics[0, 0], camera_intrinsics[1, 1], camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height

    camera = bpy.data.objects["Camera"]
    camera.rotation_euler = tra.euler_from_matrix(camera_extrinsics)
    camera.location = tra.translation_from_matrix(camera_extrinsics)
    camera.data.sensor_width = sensor_width_mm
    camera.data.lens = fx * sensor_width_mm / width
    camera.data.shift_x = (width / 2 - cx) / width
    camera.data.shift_y = (cy - height / 2) / height
    camera.data.sensor_fit = 'HORIZONTAL'
    fx_fy = fx / fy
    if fx_fy > 1:
        bpy.context.scene.render.pixel_aspect_x = 1.
        bpy.context.scene.render.pixel_aspect_y = fx_fy
    else:
        bpy.context.scene.render.pixel_aspect_x = 1. / fx_fy
        bpy.context.scene.render.pixel_aspect_y = 1.
    
    return camera_extrinsics.copy(), camera_intrinsics.copy()


import_export_kwargs = dict(forward_axis='Y', up_axis='Z')


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().copy()


def seed_all(seed: int):
    """seed all random number generators except taichi"""
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


def randomize_texture(obj_name):
    obj = bpy.data.objects.get(obj_name)
    assert obj and obj.active_material
    
    mat = obj.active_material
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    bsdf_node = None
    for node in nodes:
        if node.type == 'BSDF_PRINCIPLED':
            bsdf_node = node
            break
    assert bsdf_node

    base_color_image = None
    for link in links:
        if link.to_node == bsdf_node and link.to_socket.name == "Base Color":
            from_node = link.from_node
            if from_node.type == 'TEX_IMAGE' and from_node.image:
                base_color_image = from_node.image
                break
    assert base_color_image
    
    img_path = bpy.path.abspath(base_color_image.filepath)
    img = Image.open(img_path).convert("RGB")
    img = ImageEnhance.Brightness(img).enhance(2 ** np.random.uniform(-1., +1.))
    img = ImageEnhance.Contrast(img).enhance(2 ** np.random.uniform(-1., +1.))
    img_hsv = np.array(img.convert("HSV"))
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + np.random.randint(-20, +21)) % 256
    img = Image.fromarray(img_hsv, mode="HSV").convert("RGB")
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        print(f"save img to {f.name}")
        img.save(f.name, "PNG")
        from_node.image = bpy.data.images.load(f.name)
        bpy.ops.file.pack_all()

    for i in range(len(obj.data.loops)):
        obj.data.uv_layers.active.data[i].uv = np.array(obj.data.uv_layers.active.data[i].uv) + np.random.uniform(-0.5, 0.5, 2)


class RenderFunction:
    @staticmethod
    def robot_link_vis_name(link: str, vis: batch_urdf.Visual):
        assert isinstance(link, str), f"{type(link)}"
        assert isinstance(vis, batch_urdf.Visual), f"{type(vis)}"
        return f"robot_{link}_{os.path.basename(vis.geometry.mesh.filename)[:-4]}"
    
    def __init__(self, init_message: dict, hdri_manager: bpycv.HdriManager, texture_manager: bpycv.TextureManager):
        self.curr_obj_idx = 2 # leave '1' for cloth
        logger.info(f"pid:{os.getpid()} ppid:{psutil.Process(os.getpid()).parent().pid} pppid:{psutil.Process(os.getpid()).parent().parent().pid}")

        # setup
        self.camera_size_level = init_message["camera_size_level"]
        if init_message["engine"] == "cycles":
            bpy.data.scenes[0].render.engine = "CYCLES"
            bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
            bpy.context.scene.cycles.device = "GPU"
            bpy.context.scene.cycles.samples = 256 // self.camera_size_level 
            bpy.context.preferences.addons["cycles"].preferences.get_devices()
            for idx, d in enumerate(bpy.context.preferences.addons["cycles"].preferences.devices):
                if d["name"].startswith("NVIDIA"): # only use GPU, not CPU
                    d["use"] = 1
                print(d["name"], d["use"])
        elif init_message["engine"] == "eevee":
            bpy.context.scene.render.engine = "BLENDER_EEVEE"
        else:
            raise ValueError(f"Unsupported engine: {init_message['engine']}")
        
        bpy.context.scene.render.use_persistent_data = True
        bpy.context.scene.render.image_settings.file_format = "PNG"
        seed_all(init_message["seed"])
        self.camera_type = init_message["camera_type"]
        if RANDOMIZE_CAMERA:
            vec = np.random.randn(3)
            vec /= max(np.linalg.norm(vec), 1e-6)
            self.camera_ext_0 = tra.translation_matrix(np.random.uniform(-0.02, 0.02, 3)) @ tra.rotation_matrix(np.random.uniform(0.0, 0.05), vec)
            self.delta_camera_int = np.array([
                np.random.uniform(-5., +5.), 0., np.random.uniform(-5., +5.),
                0., np.random.uniform(-5., +5.), np.random.uniform(-5., +5.),
                0., 0., 0.
            ]).reshape(3, 3)
        else:
            self.camera_ext_0 = np.eye(4)
            self.delta_camera_int = np.zeros((3, 3))
        self.camera_parameters = dict(
            d435 = dict(
                camera_intrinsics = (np.array([[455, 0, 320], [0, 455, 180], [0, 0, self.camera_size_level]], dtype=np.float32) + self.delta_camera_int) / self.camera_size_level ,
                height = 360 // self.camera_size_level, width = 640 // self.camera_size_level,
            ),
            d436 = dict(
               camera_intrinsics = (np.array([[388, 0, 320], [0, 388, 240], [0, 0, self.camera_size_level]], dtype=np.float32) + self.delta_camera_int) / self.camera_size_level,
               height = 480 // self.camera_size_level, width = 640 // self.camera_size_level,
            ),
        )
        logger.info(f"camera_ext_0:\n{self.camera_ext_0}\ndelta_camera_int:\n{self.delta_camera_int}\n")

        # init table
        table = bpy.data.objects["table"]
        if USE_POLYHAVEN_TEXTURES:
            table.data.materials[0] = texture_manager.load_texture(texture_manager.sample())
        uni_f = random.uniform
        rotation_euler, scale = [
            ((0., 0., uni_f(-0.1, 0.1), ), (uni_f(0.5, 0.8), uni_f(0.3, 0.5), uni_f(0.02, 0.04))),
            ((0., 0., uni_f(-0.1, 0.1) + math.pi, ), (uni_f(0.5, 0.8), uni_f(0.3, 0.5), uni_f(0.02, 0.04))),
            ((0., 0., uni_f(-0.1, 0.1) + math.pi / 2, ), (uni_f(0.3, 0.5), uni_f(0.5, 0.8), uni_f(0.02, 0.04))),
            ((0., 0., uni_f(-0.1, 0.1) - math.pi / 2, ), (uni_f(0.3, 0.5), uni_f(0.5, 0.8), uni_f(0.02, 0.04))),
        ][random.randint(0, 3)]
        table.rotation_euler = rotation_euler
        table.scale = scale
        table.location = np.random.uniform(-0.02, +0.02, 2).tolist() + [-scale[2]]
        self.set_obj_inst_id(table)
        if init_message["render_set"] == "train":
            pass
        elif init_message["render_set"] == "valid":
            randomize_texture("table")
        else:
            raise ValueError(init_message["render_set"])
        
        # init pickers  
        self.picker_num = init_message["picker_num"]
        self.picker_names: list[str] = []
        for i in range(self.picker_num):
            bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=init_message["picker_radius"], location=(0, 0, 0))
            obj = bpy.context.object
            obj.name = f"picker{i}"
            self.picker_names.append(obj.name)
            self.set_obj_inst_id(obj)
        self.hide_picker = bool(init_message["hide_picker"])
        
        # init robot
        self.urdf = urdf = batch_urdf.URDF(
            batch_size=1, urdf_path=init_message["urdf_path"], dtype=torch.float32,
            device="cpu", mesh_dir=init_message["mesh_dir"],
        )
        self.robot_visual_name_all: list[str] = []
        self.robot_visual_to_obj: dict[str, list[str]] = defaultdict(list)

        def import_obj(filepath: str, vis_tf: np.ndarray, link: str, vis: batch_urdf.Visual):
            bpy.ops.wm.obj_import(filepath=filepath, use_split_groups=True, **import_export_kwargs) # split all, avoid potential bugs
            selected_obj_names = [obj.name for obj in bpy.context.selected_objects]
            for obj_name in selected_obj_names:
                obj = bpy.data.objects[obj_name]
                if len(obj.material_slots) >= 2:
                    bpy.ops.object.select_all(action='DESELECT')
                    obj.select_set(True)
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        tmp_path = os.path.join(tmp_dir, "multiple_material.obj")
                        # recursively call import_obj to handle multiple materials
                        # group_by_material=True can seperate materials into different groups
                        bpy.ops.wm.obj_export(filepath=tmp_path, export_selected_objects=True, export_material_groups=True, **import_export_kwargs)
                        bpy.data.objects.remove(obj)
                        import_obj(tmp_path, vis_tf, link, vis)
                        bpy.ops.file.pack_all()
                else:
                    obj.rotation_euler = tra.euler_from_matrix(vis_tf)
                    obj.location = tra.translation_from_matrix(vis_tf)
                    self.robot_visual_name_all.append(obj.name)
                    self.robot_visual_to_obj[self.robot_link_vis_name(link, vis)].append(obj.name)
                    self.set_obj_inst_id(obj)
        
        if not SKIP_ROBOT_FOR_QUICK_DEBUG:
            for link in urdf.link_map.keys():
                link_tf = torch_to_numpy(urdf.link_transform_map[link])[0, ...]
                for vis in urdf.link_map[link].visuals:
                    filepath = os.path.join(os.path.dirname(init_message["urdf_path"]), vis.geometry.mesh.filename)
                    vis_tf = link_tf @ torch_to_numpy(vis.origin)[0, ...]
                    import_obj(filepath, vis_tf, link, vis)
        
        ### magic modify material
        def create_random_material():
            material = bpy.data.materials.new(name="RandomMaterial")
            material.use_nodes = True
            nodes = material.node_tree.nodes
            diffuse_node = nodes["Principled BSDF"]
            gray_val = random.uniform(0.0, 1.0)
            random_color = (gray_val, gray_val, gray_val, 1)  # RGBA
            diffuse_node.inputs['Base Color'].default_value = random_color
            return material
        random_material = create_random_material()

        for obj_name, obj in bpy.data.objects.items():
            if obj_name.startswith("Gripper_Base_Link"):
                target_material = obj.data.materials[0]
        
        for obj_name, obj in bpy.data.objects.items():
            if (
                obj_name.startswith("new_gripper_l1") or 
                obj_name.startswith("new_gripper_l2") or
                obj_name.startswith("new_gripper_l3") or
                obj_name.startswith("new_gripper_r1") or 
                obj_name.startswith("new_gripper_r2") or
                obj_name.startswith("new_gripper_r3")
            ):
                obj.data.materials.append(target_material)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.modifier_add(type='NORMAL_EDIT')
            elif (
                obj_name.startswith("new_gripper_l4") or 
                obj_name.startswith("new_gripper_l5") or
                obj_name.startswith("new_gripper_r4") or 
                obj_name.startswith("new_gripper_r5")
            ):
                obj.data.materials.append(random_material)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.modifier_add(type='NORMAL_EDIT')
        
        # init cloth
        bpy.ops.wm.obj_import(filepath=init_message["cloth_obj_path"], **import_export_kwargs)
        assert len(bpy.context.selected_objects) == 1
        obj = bpy.context.selected_objects[0]
        obj.name = "cloth"
        obj["inst_id"] = 1

        self.tm_mesh_raw: trimesh.Trimesh = trimesh.load(init_message["cloth_obj_path"], process=False)
        self.tm_mesh_sim = trimesh.Trimesh(vertices=self.tm_mesh_raw.vertices, faces=self.tm_mesh_raw.faces)
        vert_xyz_to_idx = {}
        for i, v in enumerate(self.tm_mesh_sim.vertices):
            vert_xyz_to_idx[tuple(v)] = i
        vert_ren_to_sim = []
        for i, v in enumerate(self.tm_mesh_raw.vertices):
            vert_ren_to_sim.append(vert_xyz_to_idx[tuple(v)])
        self.vert_ren_to_sim = np.array(vert_ren_to_sim)
        """[NV_R] -> [0, NV_S)"""

        # init background
        if USE_HDRI_AND_REMOVE_LIGHT:
            hdri_path = hdri_manager.sample()
            bpycv.load_hdri_world(hdri_path, random_rotate_z=True)
            for obj_name in ["Light1", "Light2", "Light3", "Light4"]:
                bpy.data.objects[obj_name].hide_render = True
            
            if init_message["render_set"] == "train":
                pass
            elif init_message["render_set"] == "valid":
                world = bpy.data.worlds[0]
                node_tree = world.node_tree
                node_name = "Background"
                node = node_tree.nodes.get(node_name)
                node.inputs["Strength"].default_value = 2. ** np.random.uniform(-1, 1)
            else:
                raise ValueError(init_message["render_set"])
        
        # clear unused data
        for obj_name, obj in bpy.data.objects.items():
            if obj_name not in (
                ["Camera", "Light1", "Light2", "Light3", "Light4", "table", "cloth"] + 
                self.picker_names + self.robot_visual_name_all
            ):
                logger.info(f"remove {obj_name}")
                bpy.data.objects.remove(obj)
        
        # pack all data, avoid potential bugs
        bpy.ops.file.pack_all()
    
    def set_obj_inst_id(self, obj):
        obj["inst_id"] = self.curr_obj_idx
        print(self.curr_obj_idx, obj.name)
        self.curr_obj_idx += 1
    
    def render(self, args: dict):
        print(f"rendering\n{pprint.pformat(args)}")
        logger.info(f"\n\nstart rendering")
        cfg = Cfg(**args)

        logger.info(f"start clean")

        logger.info(f"set pickers")
        for picker_name, xyz in zip(self.picker_names, cfg.picker_xyz):
            obj = bpy.data.objects[picker_name]
            if xyz is None or self.hide_picker:
                obj.hide_render = True
            else:
                obj.hide_render = False
                obj.location = xyz
        
        logger.info(f"set robot")
        with open(os.path.join(cfg.input_folder, "cfg.json"), "r") as f:
            robot_cfg = json.load(f)
        urdf = self.urdf
        urdf.update_base_link_transformation(robot_cfg["base_link"], urdf.tensor([robot_cfg["base_pose"]]))
        urdf.update_cfg(urdf.cfg_f2t(robot_cfg["qpos"]))

        if not SKIP_ROBOT_FOR_QUICK_DEBUG:
            for link in urdf.link_map.keys():
                link_tf = torch_to_numpy(urdf.link_transform_map[link])[0, ...]
                for vis in urdf.link_map[link].visuals:
                    vis_tf = link_tf @ torch_to_numpy(vis.origin)[0, ...]
                    for obj_name in self.robot_visual_to_obj[self.robot_link_vis_name(link, vis)]:
                        obj = bpy.data.objects[obj_name]
                        obj.rotation_euler = tra.euler_from_matrix(vis_tf)
                        obj.location = tra.translation_from_matrix(vis_tf)

        logger.info(f"set cloth")
        self.tm_mesh_sim.vertices = np.load(os.path.join(cfg.input_folder, "cloth_v_sim.npy"))
        v_np = self.tm_mesh_sim.vertices[self.vert_ren_to_sim]
        vn_np = self.tm_mesh_sim.vertex_normals[self.vert_ren_to_sim]

        obj = bpy.data.objects["cloth"]
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        for i, v in enumerate(obj.data.vertices):
            v.co = v_np[i]
        obj.data.normals_split_custom_set_from_vertices(vn_np)
        bpy.ops.object.shade_smooth = True

        logger.info(f"set camera")
        if cfg.mode == "head":
            camera_frame_tf = torch_to_numpy(self.urdf.link_transform_map["head_camera_sim_view_frame"])[0, ...]
            cam_ext, cam_int = set_camera(
                camera_extrinsics=camera_frame_tf @ tra.euler_matrix(np.pi / 2, 0., -np.pi / 2) @ self.camera_ext_0, 
                **self.camera_parameters[self.camera_type], 
            )
        elif cfg.mode == "side":
            cam_ext, cam_int = set_camera(
                camera_extrinsics=tra.translation_matrix([1.2, 1.2, 1.]) @ tra.euler_matrix(1.1, 0., np.pi / 4 * 3) @ self.camera_ext_0,
                **self.camera_parameters["d435"], 
            )
        else:
            raise NotImplementedError(f"mode {cfg.mode} not supported")

        logger.info(f"start render")
        result = bpycv.render_data()
        assert np.all(result["inst"] < 256), "max inst id is 255"

        logger.info(f"save result")
        os.makedirs(os.path.dirname(cfg.output_png), exist_ok=True)
        cv2.imwrite(cfg.output_png, result["image"][..., ::-1])

        cam_param_path = os.path.join(os.path.dirname(cfg.output_png) + "_cam_param", os.path.basename(cfg.output_png)[:-4] + ".json")
        os.makedirs(os.path.dirname(cam_param_path), exist_ok=True)
        with open(cam_param_path, "w") as f:
            json.dump(dict(
                camera_extrinsics=cam_ext.tolist(),
                camera_intrinsics=cam_int.tolist(),
            ), f, indent=4)

        if cfg.save_npy_img:
            npy_path = os.path.join(os.path.dirname(cfg.output_png) + "_rgb_mask", os.path.basename(cfg.output_png)[:-4] + ".npy")
            os.makedirs(os.path.dirname(npy_path), exist_ok=True)
            np.save(npy_path, np.concatenate([result["image"].transpose(2, 0, 1), result["inst"][None, :, :]], axis=0).astype(np.uint8))
        if cfg.save_blend_file:
            blend_path = os.path.join(os.path.dirname(cfg.output_png) + "_blend", os.path.basename(cfg.output_png)[:-4] + ".blend")
            if os.path.exists(blend_path):
                os.remove(blend_path)
            os.makedirs(os.path.dirname(blend_path), exist_ok=True)
            bpy.ops.file.pack_all()
            bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        
        args_path = os.path.join(os.path.dirname(cfg.output_png) + "_render_args", os.path.basename(cfg.output_png)[:-4] + ".json")
        os.makedirs(os.path.dirname(args_path), exist_ok=True)
        with open(args_path, "w") as f:
            json.dump(args, f, indent=4)
        
        logger.info(f"done\n\n")
        if cfg.send_sigusr1:
            os.kill(cfg.target_pid, signal.SIGUSR1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_init", action="store_true")
    parser.add_argument("--run_test", action="store_true")
    parser.add_argument("--use_file_instead_of_pipe", action="store_true")
    parser.add_argument("--init_message_file", type=str, default="init_message.json")
    parser.add_argument("--render_args_file", type=str, default="render_args.json")
    parser.add_argument("--output_png", type=str, default="tmp_render_output/png/output.png")
    argv = sys.argv[sys.argv.index("--") + 1:]
    command_args = parser.parse_args(argv)
    print(command_args)

    if command_args.run_test:
        return

    if command_args.run_init:
        fail_flag = True
        while fail_flag:
            fail_flag = False
            try:
                hdri_manager = bpycv.HdriManager(hdri_dir=HDRI_CACHE_DIR, category="indoor", resolution="1k", download=True, debug=True)
                texture_manager = bpycv.TextureManager(tex_dir=TEXTURE_CACHE_DIR, category="wood,clean", resolution="1k", download=True, debug=True)
            except Exception as e:
                fail_flag = True
                print(f"failed to init, {e}")
            for p in texture_manager.tex_paths:
                try:
                    table = bpy.data.objects["table"]
                    table.data.materials[0] = texture_manager.load_texture(p)
                except Exception as e:
                    fail_flag = True
                    print(f"failed to load {p}, {e}")
                    os.remove(p)
                # print(f"successfully loaded {p}")
            for p in hdri_manager.hdr_paths:
                try:
                    bpycv.load_hdri_world(p, random_rotate_z=True)
                except Exception as e:
                    fail_flag = True
                    print(f"failed to load {p}, {e}")
                    os.remove(p)
                # print(f"successfully loaded {p}")
        return
    hdri_manager = bpycv.HdriManager(hdri_dir=HDRI_CACHE_DIR, category="indoor", resolution="1k")
    texture_manager = bpycv.TextureManager(tex_dir=TEXTURE_CACHE_DIR, category="wood,clean", resolution="1k")

    if command_args.use_file_instead_of_pipe:
        with open(command_args.init_message_file, "r") as f:
            init_message = json.load(f)
    else:
        dec = json.JSONDecoder()
        input_str = input()
        init_message = dec.decode(input_str)
    
    rf = RenderFunction(init_message, hdri_manager, texture_manager)

    if command_args.use_file_instead_of_pipe:
        with open(command_args.render_args_file, "r") as f:
            render_args = json.load(f)
        render_args["output_png"] = command_args.output_png
        render_args["send_sigusr1"] = False
        rf.render(render_args)
    else:
        while True:
            try:
                input_str = input()
                message = dec.decode(input_str)
                print(f"get action: {message['action']}")
                if message["action"] == "join":
                    break
                elif message["action"] == "message":
                    rf.render(message["message"])
            except EOFError:
                break
    
    # clean up
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj)
    bpy.ops.outliner.orphans_purge()

if __name__ == '__main__':
    main()