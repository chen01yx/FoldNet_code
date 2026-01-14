import os
import shutil
import subprocess

import torch
import numpy as np 

import cv2
from PIL import Image

from diffusers import DiffusionPipeline

from garmentds.gentexture.template.clothes import * 

class Painter:
    def __init__(
        self, 
        client:str = None,
        pipeline:str = "stabilityai/stable-diffusion-3.5-large",
        use_same_front_back: bool = False,
        use_symmetric_texture: bool = False,
        use_polyhaven_textures: bool = False,
        clear_cache: bool = False, 
        **kwargs
    ):
        self.pipeline = None
        self.pipeline_name = pipeline
        self.use_polyhaven_textures = use_polyhaven_textures
        self.cloth_handler: dict[str, Base] = {
            "tshirt": TShirtSim(client, use_same_front_back, use_symmetric_texture),
            "tshirt_sp": TShirtSPSim(client, use_same_front_back, use_symmetric_texture),
            "trousers": TrousersSim(client, use_same_front_back, use_symmetric_texture),
            "vest": VestCloseSPSim(client, use_same_front_back, use_symmetric_texture),
            "hooded": HoodedCloseSim(client, use_same_front_back, use_symmetric_texture),     
        }
        
        self.client = client
        self.clear_cache = bool(clear_cache)

    def paint_cloth(self, category:str=None, output_dir:str=None, texture_ready_to_use:str=None):
        """
        First, generate texture image under 'output_dir', 
        Second, apply texture to the cloth object and render it, \
                the outcome image is places under 'output_dir'.
        """
        
        self.generate_texture_images(category, output_dir, texture_ready_to_use)

        base_dir = os.environ["GARMENTDS_BASE_DIR"]
        script = os.path.join(base_dir, "src/garmentds/gentexture/utils/blender_script.py")
        self.render(script, output_dir, need_mask=False, need_keypoints_2D=False)

    def generate_texture_images(self, category, output_dir, texture_ready_to_use=None):
        """
            generate texture images according to category

            parameters:
                category: str, category of cloth, e.g. "tshirt_sp"
                output_dir: str, directory to place texture images
                texture_ready_to_use: str, path to texture image, if provided, will not generate texture image
        """
        if texture_ready_to_use is None:
            images = []
            prompts = self.cloth_handler[category].generate_prompts()

            if self.pipeline is None: # lazy initiate
                self.pipeline = DiffusionPipeline.from_pretrained(
                    self.pipeline_name, torch_dtype=torch.float16, 
                    device_map="balanced", local_files_only=True
                )

            for prompt in prompts:
                images.append(self.pipeline(prompt, num_inference_steps=20).images[0])

            img = self.cloth_handler[category].union_images(images)
            Image.fromarray(np.uint8(img)).save(os.path.join(output_dir, "material_0.png"))
        
        else:
            shutil.copy(texture_ready_to_use, os.path.join(output_dir, "material_0.png"))

    def render(self, blender_script: str, output_dir: str, 
               need_mask: bool, need_keypoints_2D: bool,
               cloth_use_polyhaven_textures: bool = False):
        """
            render 'mesh.obj' using 'material_0.png', 
            the outcome image is called 'mesh_rendered.png', 
            'material_0.png' should be placed under 'output_dir'.

            file_structure:
                tshirt_sp/0/
                |---- mesh.obj               <--- we get mesh.obj here using relpath
                |---- textures/
                    |---- 0/                 <--- here is "output_dir"
                        |---- material_0.png <--- texture image
        """
        command_line = ['blender', '-noaudio', '--background', '--python', blender_script, 
                         '--', f'--base_dir={output_dir}', f'--cloth_rotation_euler=[0.0,0.0,0.0]']
        if need_mask:
            command_line.append("--need_mask")
        if need_keypoints_2D:
            command_line.append("--need_keypoints_2D")
        if cloth_use_polyhaven_textures:
            command_line.append("--cloth_use_polyhaven_textures")
        
        if self.clear_cache and self.pipeline is not None:
            import gc
            del self.pipeline
            gc.collect()
            torch.cuda.empty_cache()
            self.pipeline = None

        subprocess.call(command_line)

    def export_cfg(self):
        return {
            "client": self.client,
            "pipeline": self.pipeline_name,
            "client_prompt_template": Base().get_client_prompt_template()
        }

if __name__ == "__main__":
    """
        Use the following script to generate ready-to-use texture images.
    """

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="tshirt_sp")
    parser.add_argument("--pipeline", type=str, default=None)
    parser.add_argument("--client", type=str, default="openai")
    parser.add_argument("--use_same_front_back", action="store_true")
    parser.add_argument("--use_symmetric_texture", action="store_true")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_to_generate", type=int, default=1)
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()

    os.environ["GARMENTDS_BASE_DIR"] = os.path.abspath(__file__).split("src")[0][:-1]
    painter = Painter(client=args.client, pipeline=args.pipeline, 
                    use_same_front_back=args.use_same_front_back,
                    use_symmetric_texture=args.use_symmetric_texture)

    for i in range(args.start_idx, args.start_idx + args.num_to_generate):
        output_dir = os.path.join(args.output_dir, str(i))
        os.makedirs(output_dir, exist_ok=True)
        painter.generate_texture_images(category=args.category, output_dir=output_dir, texture_ready_to_use=None)
