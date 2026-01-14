import os
import subprocess
import json
import shutil
import pprint
from PIL import Image, ImageDraw
from garmentds.foldenv.fold_env import FoldEnv
import garmentds.common.utils as utils


def append_render_dir(env: FoldEnv, render_dir_list: list[str]):
    for mode in env.get_render_mode():
        if mode in ["head", "side"]:
            render_dir_list.append(os.path.abspath(os.path.join(env.get_render_output(), mode)))


def generate_video_old(fps: int, render_dir_list: list[str]):
    cwd = os.getcwd()
    for d in render_dir_list:
        if os.path.exists(d):
            os.chdir(d)
            cmd = f"ti video -f {int(fps)} -o 0.mp4"
            print(f"{d} run cmd: {cmd}")
            subprocess.run(cmd, shell=True)
    os.chdir(cwd)


def generate_video(fps: int, render_dir_list: list[str]):
    cwd = os.getcwd()
    for d in render_dir_list:
        if os.path.exists(d):
            os.chdir(d)
            text_d = os.path.join(d, "text")
            os.mkdir("text")
            for img_path in os.listdir(d):
                if img_path.endswith(".png"):
                    original_img = Image.open(os.path.join(d, img_path))
                    width, height = original_img.size
                    black_img = Image.new("RGB", (width, height), color="black")
                    img = Image.new("RGB", (width * 2, height))
                    img.paste(black_img, (0, 0))
                    img.paste(original_img, (width, 0))

                    draw = ImageDraw.Draw(img)
                    step_idx = int(img_path.split('.')[0])
                    output_path = os.path.join(text_d, img_path)
                    action_path = os.path.join(d, "..", "action", f"{step_idx}.json")
                    state_path = os.path.join(d, "..", "state", f"{step_idx}.json")
                    draw.text((10, 10), f"Step {step_idx}")
                    if os.path.exists(action_path):
                        action = utils.load_json(action_path)
                        draw.text((10, 30), f"is correct action: {action['is_correct_action']}")
                    if os.path.exists(state_path):
                        state = utils.load_json(state_path)
                        draw.text((10, 50), f"state: {pprint.pformat(state)}")
                    img.save(output_path)
            os.chdir("text")
            cmd = f"ti video -f {int(fps)} -o ../0.mp4"
            print(f"{d} run cmd: {cmd}")
            subprocess.run(cmd, shell=True)
            os.chdir("..")
            shutil.rmtree("text")
    os.chdir(cwd)


def export_meta_info_and_clear(traj_dir: str, meta_info: dict, filename: str):
    meta_info_path = os.path.join(traj_dir, filename)
    os.makedirs(os.path.dirname(meta_info_path), exist_ok=True)
    with open(meta_info_path, "w") as f:
        json.dump(meta_info, f, indent=4)
    meta_info.clear()


__all__ = [
    "append_render_dir", 
    "generate_video", 
    "export_meta_info_and_clear", 
]