import os
import sys

import json
import argparse

import math
import numpy as np
import trimesh.transformations as tra

import bpy
import bpycv

blender_version: tuple[int, int, int] = bpy.app.version
print(f"blender_version:{blender_version}")

class RscManager:
    def __init__(self, base_dir:str):
        self.base_dir = base_dir

        # find material_0.png
        if os.path.exists(os.path.join(self.base_dir, "material_0.json")):
            with open(os.path.join(self.base_dir, "material_0.json"), "r") as f:
                data = json.load(f)
            self.texture_path = data["texture_path"]
        elif os.path.exists(os.path.join(self.base_dir, "material_0.png")):
            self.texture_path = os.path.join(self.base_dir, "material_0.png")
        elif os.path.exists(os.path.join(self.base_dir, "texture_kd.png")):
            # text2tex
            self.texture_path = {
                "texture_kd" : os.path.join(self.base_dir, "texture_kd.png"),
                "texture_ks" : os.path.join(self.base_dir, "texture_ks.png"),
                "texture_n" : os.path.join(self.base_dir, "texture_n.png"),
            }
        else:
            raise ValueError("Cannot find material_0.png.")

        # find mesh.obj/mesh_deformed.obj
        if os.path.exists(os.path.join(self.base_dir, "mesh.obj")):
            self.mesh_path = os.path.join(self.base_dir, "mesh.obj")
        elif os.path.exists(os.path.join(self.base_dir, "mesh_deformed.obj")):
            self.mesh_path = os.path.join(self.base_dir, "mesh_deformed.obj")
        elif os.path.exists(os.path.join(self.base_dir, "..", "..", "mesh.obj")):
            self.mesh_path = os.path.join(self.base_dir, "..", "..", "mesh.obj")
        else:
            raise ValueError("Cannot find mesh.obj/mesh_deformed.obj.")

    def get_base_dir(self):
        return self.base_dir

    def get_texture_path(self):
        return self.texture_path
    
    def get_mesh_path(self):
        return self.mesh_path
    
    def get_keypoints_3D_path(self):
        path = os.path.join(self.base_dir, "keypoints_3D.npy")
        if os.path.exists(path):
            return path
        else:
            raise ValueError(f"Cannot find keypoints_3D.npy.")

class BlenderEnv:
    def __init__(self, rsc_manager: RscManager, 
                 hdri_manager:bpycv.hdri_manager,
                 texture_manager:bpycv.texture_manager):
        self.rsc_manager = rsc_manager
        self.hdri_manager = hdri_manager
        self.texture_manager = texture_manager
        self.config = self.get_default_config()
        self.lights = None

    def update_config(self, config: dict):
        for k, v in config.items():
            if k in self.config.keys():
                self.config[k].update(v)
            else:
                self.config[k] = v

    def get_default_config(self) -> dict:
        default_config = {}
        default_config.update(
            mesh=dict(
                location=(0.0, 0.0, 0.0),
                #rotation_euler=(-np.pi/2., np.pi, 0.),
                rotation_euler=(0., 0., 0.),
                use_polyhaven=False,
            ),
            camera=dict(
                location=(0.0, 0.0, 2.5),
                rotation_euler=(0.0, 0.0, 0.0),
                sensor_width=32,
                focal_length=50,
            ),
            render=dict(
                engine="cycles",
                samples = 512,
                use_gpu=True,
                resolution_x = 640,
                resolution_y = 480,
            ),
            light=dict(
                number=4,
                type="POINT",
                energy=30,
                color=(1, 1, 1),
                random_disable = True,
            ),
            background=dict(
                color=(0.2, 0.4, 0.8, 1),
                strength=0.0,
                use_hdri=True,
            )
        )
        return default_config

    def set_scene(self):
        # Clear existing objects in the scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        # Set environment
        self.__set_mesh__(self.config["mesh"])
        self.__set_light__(self.config["light"])
        self.__set_camera__(self.config["camera"])
        self.__set_render__(self.config["render"])
        self.__set_background__(self.config["background"])

    def __set_light__(self, light_cfg):
        """
        Light are placed in a loop as a fixed height.
        """
        
        light_num, light_type, light_energy, light_color, \
            light_random_disable = light_cfg.values()
        
        force_on = np.random.choice(range(light_num))
        for i in range(light_num):
            light = bpy.data.lights.new(name=f"Light{i}",type=light_type)
            light.color, light.energy = light_color, light_energy
            light_obj = bpy.data.objects.new(name=f"Light{i}", object_data=light)
            
            # place light in a loop at y=0.5
            x = np.cos(2*math.pi*(i+0.5)/light_num)
            z = np.sin(2*math.pi*(i+0.5)/light_num)
            light_obj.location = (x, 0.5, z)
            bpy.context.collection.objects.link(light_obj)
            
            # randomly disable some lights
            if light_random_disable and np.random.rand() < 0.5 and not i == force_on:
                light_obj.hide_render = True

            # register light in case of later use
            if self.lights is None:
                self.lights = []
            self.lights.append(f"Light{i}")

    def __set_camera__(self, camera_cfg):
        location, rotation_euler, sensor_width, focal_length = camera_cfg.values()

        # Create a new camera object
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        camera.location = location
        camera.rotation_euler = rotation_euler
        camera.matrix_world = camera.matrix_basis
        camera.data.sensor_width = sensor_width
        camera.data.lens = focal_length
        camera.data.sensor_fit = "HORIZONTAL"

        # Set this camera as the active camera in the scene
        bpy.context.scene.camera = camera

    def __set_mesh__(self, mesh_cfg):
        location, rotation_euler, use_polyhaven = mesh_cfg.values()

        # Import mesh object
        mesh_path = self.rsc_manager.get_mesh_path()
        #bpy.ops.wm.obj_import(filepath=mesh_path, use_split_groups=True)
        if blender_version[0] < 4:
            bpy.ops.import_scene.obj(filepath=mesh_path)
        else:
            bpy.ops.wm.obj_import(filepath=mesh_path)
        cloth_obj = bpy.context.selected_objects[-1]
        cloth_obj.name = "Cloth"
        cloth_obj.location = location
        cloth_obj.rotation_euler = rotation_euler

        # Apply texture
        texture_path = self.rsc_manager.get_texture_path()
        if not use_polyhaven:
            # Generate new material and apply texture
            material = bpy.data.materials.new(name="finalmtl")
            material.use_nodes = True

            # Get the material's nodes
            nodes = material.node_tree.nodes

            # Clear default nodes
            for node in nodes:
                nodes.remove(node)

            # Create a Principled BSDF shader
            nod_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
            # nod_bsdf.location = (0, 0)

            # Create material output
            nod_material_output = nodes.new(type="ShaderNodeOutputMaterial")

            if isinstance(texture_path, str):
                # synthetic texture, only base color
                txt_base_color = nodes.new(type='ShaderNodeTexImage')
                txt_base_color.image = bpy.data.images.load(texture_path)

                # Link the nodes
                links = material.node_tree.links
                links.new(nod_material_output.inputs["Surface"], nod_bsdf.outputs["BSDF"])
                links.new(nod_bsdf.inputs["Base Color"], txt_base_color.outputs["Color"])
            elif isinstance(texture_path, dict):
                # text2tex, base color, specularity, normal map
                txt_base_color = nodes.new(type='ShaderNodeTexImage')
                txt_spec_color = nodes.new(type='ShaderNodeTexImage')
                txt_norm_color = nodes.new(type='ShaderNodeTexImage')

                # Load image files
                txt_base_color.image = bpy.data.images.load(texture_path["texture_kd"])
                txt_spec_color.image = bpy.data.images.load(texture_path["texture_ks"])
                txt_norm_color.image = bpy.data.images.load(texture_path["texture_n"])

                # Link the nodes
                links = material.node_tree.links
                links.new(nod_material_output.inputs["Surface"], nod_bsdf.outputs["BSDF"])
                links.new(nod_bsdf.inputs["Base Color"], txt_base_color.outputs["Color"])
                links.new(nod_bsdf.inputs['Specular'], txt_spec_color.outputs['Color'])

                nod_normal_map = nodes.new(type='ShaderNodeNormalMap')
                links.new(nod_normal_map.inputs['Color'], txt_norm_color.outputs['Color'])
                links.new(nod_bsdf.inputs['Normal'], nod_normal_map.outputs['Normal'])
        else:
            polyhaven_texture_path = texture_path
            print(f"[ INFO ] Using polyhaven texture: {polyhaven_texture_path}")
            material = self.texture_manager.load_texture(polyhaven_texture_path)

        # Assign the material to the imported object
        if cloth_obj.data.materials:
            cloth_obj.data.materials[0] = material
        else:
            cloth_obj.data.materials.append(material)
        cloth_obj.active_material = material
    
    def __set_background__(self, background_cfg):
        """
        Set background color and strength.
        If use_hdri is True, set background to an HDRI image.
        """
        
        color, strength, use_hdri = background_cfg.values()

        # Set background color and strength
        bpy.context.scene.world = bpy.data.worlds["World"] # Assign the World to the scene
        bpy.context.scene.world.use_nodes = True
        nodes = bpy.context.scene.world.node_tree.nodes

        background_node = nodes.get("Background")
        if not background_node:
            background_node = nodes.new(type="ShaderNodeBackground")

        background_node.inputs["Color"].default_value = color
        background_node.inputs["Strength"].default_value = strength

        # use hdri 
        if use_hdri:
            hdri_path = self.hdri_manager.sample()
            bpycv.load_hdri_world(hdri_path, random_rotate_z=True)
            for light in self.lights:
                bpy.data.objects[light].hide_render = True

    def __set_render__(self, render_cfg):
        engine, samples, use_gpu, resolution_x, resolution_y = render_cfg.values()

        if engine == "cycles":
            bpy.context.scene.render.engine = "CYCLES"
            bpy.context.scene.cycles.samples = samples
            if use_gpu:
                bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
                bpy.context.scene.cycles.device = "GPU"
                bpy.context.preferences.addons["cycles"].preferences.get_devices()
                for idx, d in enumerate(bpy.context.preferences.addons["cycles"].preferences.devices):
                    if d["name"].startswith("NVIDIA"): # only use GPU, not CPU
                        d["use"] = 1
        elif engine == "eevee":
            bpy.context.scene.render.engine = "BLENDER_EEVEE"
        else:
            raise ValueError(f"Unsupported engine: {engine}")

        render = bpy.context.scene.render
        render.image_settings.file_format = 'PNG'
        render.image_settings.color_mode = 'RGB'
        render.resolution_x = resolution_x
        render.resolution_y = resolution_y

    def render(self, need_mask=True):
        """
        Get rgb and mask images from blender rendering.
        """
        bpy.context.view_layer.use_pass_object_index = True # Enable Object Index pass
        cloth_obj = bpy.data.objects["Cloth"]
        cloth_obj.pass_index = 1  # Unique ID for segmentation

        # Ensure the compositor is used
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        tree.nodes.clear()

        # Add render layers node
        render_layers = tree.nodes.new(type="CompositorNodeRLayers")

        # Add output node for RGB
        file_output_rgb = tree.nodes.new(type="CompositorNodeOutputFile")
        file_output_rgb.base_path = self.rsc_manager.get_base_dir()
        file_output_rgb.file_slots[0].path = "mesh_rendered"

        # Connect render image (RGB) to output
        tree.links.new(render_layers.outputs["Image"], file_output_rgb.inputs[0])

        if need_mask:
            # Add output node for Object Index pass (Mask)
            file_output_mask = tree.nodes.new(type="CompositorNodeOutputFile")
            file_output_mask.base_path = self.rsc_manager.get_base_dir()
            file_output_mask.file_slots[0].path = "mask"

            # Connect Object Index pass to mask output
            tree.links.new(render_layers.outputs["IndexOB"], file_output_mask.inputs[0])

        # Render the scene
        base_dir = self.rsc_manager.get_base_dir()
        bpy.ops.render.render(write_still=True)
        os.rename(os.path.join(base_dir, "mesh_rendered0001.png"), 
                  os.path.join(base_dir, "mesh_rendered.png"))
        if need_mask:
            os.rename(os.path.join(base_dir, "mask0001.png"), 
                      os.path.join(base_dir, "mask.png"))

    def get_keypoints_2D(self):
        """
        Map keypoints_3D to keypoints_2D.
        """
        keypoints_3D_path = self.rsc_manager.get_keypoints_3D_path()
        keypoints_np = np.load(keypoints_3D_path)
        mesh_rotation_matrix = tra.euler_matrix(*self.config["mesh"]["rotation_euler"])
        keypoints_3D_homo = np.hstack([keypoints_np[:, :3], np.ones((keypoints_np.shape[0], 1))])
        keypoints_3D = (keypoints_3D_homo @ mesh_rotation_matrix.T)[:, :3]

        # Get the scene's resolution and scale
        render = bpy.context.scene.render
        resolution_x = render.resolution_x * render.resolution_percentage / 100.0
        resolution_y = render.resolution_y * render.resolution_percentage / 100.0

        # Prepare camera matrices
        camera = bpy.context.scene.camera
        camera_matrix_world_inv = np.array(camera.matrix_world.inverted())

        lens = camera.data.lens
        sensor_width = camera.data.sensor_width if camera.data.sensor_fit != 'VERTICAL' else camera.data.sensor_height
        sensor_height = sensor_width * resolution_y / resolution_x

        camera_type = camera.data.type
        if camera_type not in ['PERSP', 'ORTHO']:
            raise ValueError(f"Unsupported camera type: {camera_type}")

        # Function to map multiple points to pixel coordinates
        def world_to_pixel_batch(camera_matrix_world_inv, points_world_np):
            # Convert world coordinates to camera space
            points_world_homogeneous = np.hstack([points_world_np, np.ones((points_world_np.shape[0], 1))])  # Make points homogeneous
            points_camera = points_world_homogeneous @ camera_matrix_world_inv.T  # Transform to camera space

            if camera_type == 'PERSP':
                # Perspective projection
                x_ndc = (points_camera[:, 0] / -points_camera[:, 2]) / (sensor_width / (2.0 * lens))
                y_ndc = (points_camera[:, 1] / -points_camera[:, 2]) / (sensor_height / (2.0 * lens))
            elif camera_type == 'ORTHO':
                # Orthographic projection
                scale = camera.data.ortho_scale
                x_ndc = points_camera[:, 0] / scale
                y_ndc = points_camera[:, 1] / scale

            # Map NDC to pixel coordinates
            x_pixel = (x_ndc + 1) * 0.5 * resolution_x
            y_pixel = (1 - y_ndc) * 0.5 * resolution_y  # Flip y-axis for Blender's coordinate system

            # Combine valid results and their corresponding indices
            pixel_coords = np.vstack([x_pixel, y_pixel]).T
            return pixel_coords

        # Compute pixel coordinates for all points
        pixel_coords = world_to_pixel_batch(camera_matrix_world_inv, keypoints_3D)
        pixel_coords_unobstructed = np.hstack([pixel_coords, keypoints_np[:, -1].reshape(-1, 1)])
        for i in range(pixel_coords_unobstructed.shape[0]):
            x, y = pixel_coords_unobstructed[i, :2]
            if x < 0 or x > resolution_x or y < 0 or y > resolution_y:
                pixel_coords_unobstructed[i, -1] = 0  # Set invalid keypoints to 0
        np.save(os.path.join(self.rsc_manager.get_base_dir(), "keypoints_2D.npy"), 
                pixel_coords_unobstructed)


def main():
    CACHE_DIR = os.path.join(os.path.dirname(__file__), \
                "..", "..", "..", "..", ".cache", "blender")
    HDRI_CACHE_DIR = os.path.join(CACHE_DIR, "hdri")
    TEXTURE_CACHE_DIR = os.path.join(CACHE_DIR, "texture")

    args = sys.argv[sys.argv.index("--") + 1:]

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--base_dir", type=str, required=True, 
                           help="where texture.png or mesh_deformed.obj is located")
    argparser.add_argument("--hdri_cache_dir", type=str, default=HDRI_CACHE_DIR, 
                           help="where hdri files are located")
    argparser.add_argument("--hdri_download", action="store_true", 
                           help="whether to download hdri from polyhaven or not")
    argparser.add_argument("--texture_cache_dir", type=str, default=TEXTURE_CACHE_DIR, 
                           help="where polyhaven texture files are located")
    argparser.add_argument("--texture_download", action="store_true", 
                           help="whether to download polyhaven textures or not")    
    argparser.add_argument("--need_mask", action="store_true", 
                           help="whether to render mask or not")
    argparser.add_argument("--need_keypoints_2D", action="store_true",
                           help="whether to compute keypoints_2D or not")
    argparser.add_argument("--cloth_rotation_euler", type=str, default="[0,0,0]",
                           help="how should cloth be rotated to face +y direction," 
                           "and align its head to camera's viewport.")
    argparser.add_argument("--cloth_use_polyhaven_textures", action="store_true",
                           help="Use polyhaven texture instead of generated one.")
    args = argparser.parse_args(args)
    
    # Set up hdri manager and texture manager
    hdri_manager = bpycv.HdriManager(hdri_dir=HDRI_CACHE_DIR, category="indoor", \
                                     resolution="1k", download=args.hdri_download, debug=True)
    text_manager = bpycv.TextureManager(tex_dir=TEXTURE_CACHE_DIR, resolution="1k",
                                        download=args.texture_download, debug=True)

    # Set up rsc manager
    rsc_manager = RscManager(base_dir=args.base_dir)

    env = BlenderEnv(rsc_manager=rsc_manager, hdri_manager=hdri_manager, texture_manager=text_manager)
    cloth_rotation_euler = json.loads(args.cloth_rotation_euler)
    env.update_config({
        "mesh": {
            "rotation_euler": cloth_rotation_euler, 
            "use_polyhaven": args.cloth_use_polyhaven_textures
        }})
    
    env.set_scene()
    env.render(need_mask = args.need_mask)
    if args.need_keypoints_2D:
        env.get_keypoints_2D()

if __name__ == "__main__":
    main()