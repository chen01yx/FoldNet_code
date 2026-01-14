import os
import shutil

import base64
import json
import numpy as np
import PIL.Image as Image

import requests
from openai import OpenAI

import torch
from torchmetrics.functional.multimodal import clip_score
from functools import partial

from garmentds.gentexture.utils.clients import *

class CLIP_Client:
    def __init__(self):
        self.clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

    def infer_score(self, prompt, images):
        scores = []
        for image in images:
            image = np.array(Image.open(image))
            clip_score = self.clip_score_fn(torch.from_numpy(image).permute(2, 0, 1), prompt).detach()
            scores.append(clip_score.item())
        return np.array(scores)

class Judge:
    def __init__(self, backend:str):
        self.backend = backend
        if backend == "openai":
            self.client = OpenAI_Client()
        elif backend == "dmiapi":
            self.client = DMIAPI_Client()
        elif backend == "clip_score":
            self.client = CLIP_Client()
        else:
            print(f"[ INFO ] No rating backend, using random selection...")
            self.client = None

        self.gpt_prompt_template = \
            "describe these {num_textures} images, and score them " \
            "from 0 to 100 according to which garment in the image has more realistic textures. " \
            "The output should only contain {num_textures} scores, seperated by spaces."
        self.clip_prompt_template = "A garment with realistic textures, leaving alone the background."

    def choose_best_texture(self, output_dir, prompt=None):
        """
            score generated textures
        """

        num_textures = len(os.listdir(os.path.join(output_dir, "textured")))

        if self.client is None:
            best_idx = np.random.randint(num_textures)
        else:
            if prompt is None:
                if self.backend != "clip_score":
                    prompt = self.gpt_prompt_template.format(num_textures=num_textures)
                else:
                    prompt = self.clip_prompt_template

            images = []
            for i in range(num_textures):
                image = os.path.join(output_dir, "textured", f"{i}", "mesh_rendered.png")
                images.append(image)

            while True:
                scores = self.client.infer_score(prompt, images)
                if isinstance(scores, np.ndarray) and len(scores) == len(images):
                    break
                else:
                    print(f"[ WARN ] API returned invalid scores: {scores}, retrying...")

            print(f"[ INFO ] Rating Outcome for {output_dir}:")
            print(f"Prompt: {prompt}")
            print(f"Scores: {scores}")

            best_idx = np.argmax(scores)

        best_dir = os.path.join(output_dir, "textured", f"{best_idx}")
        # copy all files under best_dir to output_dir
        for file in os.listdir(best_dir):
            shutil.copy(os.path.join(best_dir, file), os.path.join(output_dir, file))

    def export_cfg(self):
        return {
            "backend": self.backend,
            "gpt_prompt_template": self.gpt_prompt_template,
            "clip_prompt_template": self.clip_prompt_template,
        }

if __name__ == "__main__":
    """
        Use this script to rate generated textures manually.
    """

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="openai", help="backend for rating, can be openai, dmiapi, or clip_score")
    parser.add_argument("--category", type=str, default="vest", help="category of garment")
    parser.add_argument("--start_idx", type=int, default=0, help="start index of dir to rate")
    parser.add_argument("--num_to_rate", type=int, default=1, help="number of dir to rate")
    args = parser.parse_args()

    judge = Judge(args.backend)
    category = args.category

    base_dir = ""
    for i in range(args.start_idx, args.start_idx + args.num_to_rate):
        output_dir = os.path.join(base_dir, f"{i}")
        judge.choose_best_texture(output_dir=output_dir,) 
