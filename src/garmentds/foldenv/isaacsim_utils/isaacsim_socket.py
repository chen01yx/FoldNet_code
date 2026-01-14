import socket
import sys
import argparse
import struct
import json
import random
import os

import numpy as np
from PIL import Image
import tqdm
import trimesh.transformations as tra
import numpy as np
import torch

from src.garmentds.foldenv.socket_utils import sendall, recvall

uid, gid = None, None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str)
    parser.add_argument("--port", type=int)
    parser.add_argument("--roomnum", type=int)
    parser.add_argument("--uid", type=int)
    parser.add_argument("--gid", type=int)
    args = parser.parse_args()
    return args


def init_socket(args):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((args.host, args.port))
    return client_socket


def close_socket(client_socket):
    print(f"close {client_socket.getsockname()}")
    client_socket.close()


def main():
    args = get_args()
    global uid, gid
    uid, gid = args.uid, args.gid
    client_socket = init_socket(args)
    # main_test(client_socket)
    main_isaac(client_socket, args.roomnum)
    close_socket(client_socket)


def main_test(client_socket: socket.socket):
    from dataclasses import dataclass
    @dataclass
    class RoomCfg:
        input_folder: str
        output_png: str
    print("this is test mode.")
    while True:
        data = json.loads(recvall(client_socket))
        if data == "exit":
            sendall(client_socket, "exit")
            break
        else:
            states = [RoomCfg(**state) for state in data]
            print(states)
            sendall(client_socket, "render complete")


def seed_all(seed: int):
    """seed all random number generators except taichi"""
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


def makedir_755(p: str):
    global uid, gid
    os.makedirs(p, exist_ok=True, mode=0o755)
    os.chown(p, uid, gid)


def main_isaac(client_socket: socket.socket, roomnum: int):
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({
        "headless": True,
        "renderer": "RayTracedLighting",
        "multi_gpu": False,
    })

    import omni.replicator.core as rep
    import omni.usd
    import omni.kit.commands
    from src.garmentds.foldenv.isaacsim_utils.renderer import IsaacRenderer, SceneGenerator, RoomCfg, ClothCfg

    import numpy as np
    from PIL import Image
    import tqdm
    import trimesh.transformations as tra

    class Main:
        def __init__(
            self,
            render_hz = 10,
            num_rooms = 2, 
        ):
            self.num_rooms = num_rooms

            import carb.settings
            # Disable capture on play and async rendering
            carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)
            carb.settings.get_settings().set("/omni/replicator/asyncRendering", False)
            carb.settings.get_settings().set("/app/asyncRendering", False)
            carb.settings.get_settings().set("/rtx/raytracing/fractionalCutoutOpacity", True)
            
            self.renderer = IsaacRenderer(
                num_rooms=num_rooms,
                render_hz=render_hz,
                render_mode="rtx_default"
            )
        
        def init(self):
            init_data = json.loads(recvall(client_socket))
            seed_all(init_data["seed"])

            scene_generator = SceneGenerator(
                os.path.abspath("data/asset/isaacsim"), "train", 
                init_data["cameras_name"], init_data["cloth_obj_path"]
            )
            scene_init_args = scene_generator.generate_init_args()

            for i in range(self.num_rooms):
                self.renderer.init_single_room(
                    i, **scene_init_args,
                )

            # rep.orchestrator.set_capture_on_play(False) # Data will be captured manually using step

            # num_subframes = 16 # avoid artifacts in the first frame 
            # self.renderer.world.step(render=False) # allow robot reset to initial pose
            # rep.orchestrator.step(rt_subframes=num_subframes, pause_timeline=False) # BUG: extra render to avoid empty frame

            # self.renderer.world.render()
            # rep.orchestrator.step(rt_subframes=num_subframes, pause_timeline=False) # BUG: extra render to avoid empty frame
            print("init main done.")
            sendall(client_socket, "init complete")

        def run(self):
            while True:
                data = json.loads(recvall(client_socket))
                if data == "exit":
                    sendall(client_socket, "exit")
                    break
                
                states = [RoomCfg(**state) if state is not None else None for state in data]
                if states.count(None) != len(states):
                    self.renderer.update_states(states)

                    self.renderer.world.render()
                    rep.orchestrator.step(rt_subframes=16, pause_timeline=False)

                    for room_idx, (room, state) in enumerate(zip(self.renderer.rooms, states)):
                        if state is not None:
                            result = room.get_render_products()
                            for cam_idx, cam_name in enumerate(room.cameras_name):
                                png_path = os.path.join(state.output_dir, cam_name, str(state.step_idx).zfill(4) + ".png")
                                makedir_755(os.path.dirname(png_path))
                                Image.fromarray(result[cam_idx]["rgba"]).save(png_path)

                                npy_path = os.path.join(state.output_dir, cam_name + "_rgb_mask", str(state.step_idx).zfill(4) + ".npy")
                                makedir_755(os.path.dirname(npy_path))
                                np.save(npy_path, np.concatenate([
                                    result[cam_idx]["rgba"][:, :, :3].transpose(2, 0, 1),
                                    result[cam_idx]["semantic_segmentation"][None, :, :]
                                ], axis=0).astype(np.uint8))

                                cam_param_path = os.path.join(state.output_dir, cam_name + "_cam_param", str(state.step_idx).zfill(4) + ".json")
                                makedir_755(os.path.dirname(cam_param_path))
                                with open(cam_param_path, "w") as f:
                                    json.dump(self.renderer.get_cam_param(room_idx, cam_name), f, indent=4)
                
                sendall(client_socket, "render complete")
                
        def shutdown(self):
            rep.orchestrator.wait_until_complete()
    
    main_inst = Main(num_rooms=roomnum)
    main_inst.init()
    main_inst.run()
    main_inst.shutdown()

    simulation_app.close()


if __name__ == "__main__":
    main()
