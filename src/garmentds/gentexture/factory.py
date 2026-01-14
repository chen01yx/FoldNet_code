import os
import shutil

import threading
from time import sleep

import json
import omegaconf
import subprocess

from sklearn import base
import numpy as np

from garmentds.gentexture.paint import Painter
from garmentds.gentexture.rating import Judge

class Factory:
    def __init__(self, cloth_output_dir, name):
        self.cloth_output_dir = cloth_output_dir
        self.cloth_output_idx = 0
        os.makedirs(self.cloth_output_dir, exist_ok=True)

    def set_texture_paths(self):
        """ Setup the texture paths """
        pass

    def set_cloth_output_idx(self, idx):
        """ 
        Set the output dirname under cloth_output_dir, \
        idx presents which cloth is being made.    
        """
        self.cloth_output_idx = idx
        self.output_dir = os.path.join(self.cloth_output_dir, f"{self.cloth_output_idx}")
        os.makedirs(self.output_dir, exist_ok=True)

    def get_mesh(self, mesh_data_dir):
        """ Get mesh data files from mesh_data_dir """
        pass

    def paint_cloth(self, categoroy:str):
        """ apply texture to cloth and save the render result """
        pass
    
    def export_cfg(self):
        """ export the configurations for future check """
        pass

class SyntheticFactory(Factory):
    def __init__(
        self, cloth_output_dir:str, 
        paint_cfg:omegaconf.DictConfig, 
        rating_cfg:omegaconf.DictConfig
    ):
        super().__init__(cloth_output_dir, "synthetic")        
        self.paint_cfg = paint_cfg.synthetic
        self.rating_cfg = rating_cfg
        self.painter = Painter(**self.paint_cfg)
        self.judge = Judge(**rating_cfg)

    def set_texture_paths(self):
        self.texture_ready_to_use = []
        if self.paint_cfg["ready_to_use_texture_dir"] is not None:
            self.texture_ready_to_use = get_nested_file(self.paint_cfg["ready_to_use_texture_dir"])
            print(f"[ INFO ] found {len(self.texture_ready_to_use)} ready-to-use "
                  f"textures in {self.paint_cfg['ready_to_use_texture_dir']}")

    def get_mesh(self, mesh_data_dir):
        mesh_path = os.path.join(mesh_data_dir, "mesh.obj")
        mtrl_path = os.path.join(mesh_data_dir, "material.mtl")
        info_path = os.path.join(mesh_data_dir, "mesh_info.json")

        shutil.copy(mesh_path, os.path.join(self.output_dir, "mesh.obj"))
        shutil.copy(mtrl_path, os.path.join(self.output_dir, "material.mtl"))
        shutil.copy(info_path, os.path.join(self.output_dir, "mesh_info.json"))

    def paint_cloth(self, category):
        """
            generate num_to_generate*3 textures, remain 1/3 of 
            them according to which got the best score
        """

        group_size = self.paint_cfg["group_size"]

        # generate 'group_size' images as a group for comparison
        for i in range(group_size):
            print(f"[ INFO ] Generating candidate cloth {i+1} of {group_size}...")
            texture_dir = os.path.join(self.output_dir, "textured", f"{i}")
            os.makedirs(texture_dir, exist_ok=True)
            if len(self.texture_ready_to_use) != 0:
                texture_path = np.random.choice(self.texture_ready_to_use)
                self.painter.paint_cloth(category, texture_dir, texture_path)
            else:
                self.painter.paint_cloth(category, texture_dir)

        print(f"[ INFO ] Rating candidate clothes...")
        self.judge.choose_best_texture(self.output_dir)
        print(f"[ INFO ] Rating done...")

    def export_cfg(self):
        return dict(
            paint_cfg=dict(
                self.painter.export_cfg().items(), 
                ready_to_use_texture_dir=self.paint_cfg["ready_to_use_texture_dir"]
            ),
            rating_cfg=self.judge.export_cfg(),     
        )

class Text2TexFactory(Factory):
    def __init__(self, cloth_output_dir, paint_cfg:omegaconf.DictConfig):
        super().__init__(cloth_output_dir, "text2tex")
        self.paint_cfg = paint_cfg.text2tex
        self.painter = Painter(**self.paint_cfg)
        
        # get path to blender_script
        base_dir = os.environ["GARMENTDS_BASE_DIR"]
        script = os.path.join(base_dir, "src/garmentds/gentexture/utils/blender_script.py")
        self.blender_script = script

        # get model_path
        self.model_path = os.path.join(base_dir, "external/Paint-it/paint_it.py")

    def set_texture_paths(self):
        pass

    def get_mesh(self, mesh_data_dir):
        mesh_path = os.path.join(mesh_data_dir, "mesh.obj")
        info_path = os.path.join(mesh_data_dir, "mesh_info.json")

        shutil.copy(mesh_path, os.path.join(self.output_dir, "mesh.obj"))
        shutil.copy(info_path, os.path.join(self.output_dir, "mesh_info.json"))

    def paint_cloth(self, category):
        cmd = [self.paint_cfg["python"], self.model_path]
        cmd += ["--identity", self.paint_cfg["identity"]]
        cmd += ["--obj_path", os.path.join(self.output_dir, "mesh.obj")]
        cmd += ["--output_dir", self.output_dir]
        subprocess.call(cmd)
        self.painter.render(self.blender_script, self.output_dir,
                            need_mask=False, need_keypoints_2D=False,
                            cloth_use_polyhaven_textures=False)

    def export_cfg(self):
        return dict(paint_cfg=dict(self.paint_cfg))

class PolyHavenFactory(Factory):
    def __init__(self, cloth_output_dir, paint_cfg:omegaconf.DictConfig):
        super().__init__(cloth_output_dir, "polyhaven")
        self.paint_cfg = paint_cfg.polyhaven
        self.painter = Painter(**self.paint_cfg)

        # get path to blender_script
        base_dir = os.environ["GARMENTDS_BASE_DIR"]
        script = os.path.join(base_dir, "src/garmentds/gentexture/utils/blender_script.py")
        self.blender_script = script

    def set_texture_paths(self):
        self.texture_ready_to_use = []
        self.texture_ready_to_use = get_nested_file(self.paint_cfg["cache_dir"], ext_name=".blend")
        print(f"[ INFO ] found {len(self.texture_ready_to_use)} ready-to-use "
              f"textures in {self.paint_cfg['cache_dir']}")

    def get_mesh(self, mesh_data_dir):
        mesh_path = os.path.join(mesh_data_dir, "mesh.obj")
        info_path = os.path.join(mesh_data_dir, "mesh_info.json")
                
        shutil.copy(mesh_path, os.path.join(self.output_dir, "mesh.obj"))
        shutil.copy(info_path, os.path.join(self.output_dir, "mesh_info.json"))

    def paint_cloth(self, category):
        texture_path = np.random.choice(self.texture_ready_to_use)
        with open(os.path.join(self.output_dir, "material_0.json"), "w") as f:
            json.dump({"texture_path": texture_path}, f)
        self.painter.render(self.blender_script, self.output_dir,
                            need_mask=False, need_keypoints_2D=False,
                            cloth_use_polyhaven_textures=True)
        
    def export_cfg(self):
        return dict(paint_cfg=dict(self.paint_cfg))

def make_cloth(**cfg):
    category, start_idx, num_to_generate, mesh_input_dir, \
        cloth_output_dir, strategy, paint_cfg, rating_cfg = cfg["garment"].values()
        
    # get mesh
    all_mesh_paths = os.listdir(mesh_input_dir)
    print(f"[ INFO ] found {len(all_mesh_paths)} meshes in {mesh_input_dir}")
    
    # create factory
    if strategy == "synthetic":
        factory = SyntheticFactory(cloth_output_dir, paint_cfg, rating_cfg)
    elif strategy == "polyhaven":
        factory = PolyHavenFactory(cloth_output_dir, paint_cfg)
    elif strategy == "text2tex":
        factory = Text2TexFactory(cloth_output_dir, paint_cfg)
    else:
        raise ValueError(f"No such factory: {strategy}")
    factory.set_texture_paths()
    
    with open(os.path.join(cloth_output_dir, "cloth_cfg.json"), "w") as f:
        json.dump(dict(
            category=category, start_idx=start_idx, num_to_generate=num_to_generate,
            mesh_input_dir=mesh_input_dir, strategy=strategy,
            factory_cfg=factory.export_cfg(),
        ), f, indent=4)

    for i in range(num_to_generate):
        # randomly select a mesh
        random_mesh_idx = np.random.choice(all_mesh_paths)

        factory.set_cloth_output_idx(i+start_idx)
        factory.get_mesh(os.path.join(mesh_input_dir, random_mesh_idx))
        factory.paint_cloth(category)

    print("[ Info ] cloth maked successfully! exiting...")

def get_nested_file(path:str, ext_name:str = None) -> list[str]:
    if os.path.isfile(path):
        if ext_name is None or os.path.splitext(path)[1] == ext_name:
            return [path,]
    elif os.path.isdir(path):
        ret = []
        for f in os.listdir(path):
            ret += get_nested_file(os.path.join(path, f), ext_name)
        return ret
    return []
    