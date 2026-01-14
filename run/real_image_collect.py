import pathlib
import os
import math

import numpy as np
from PIL import Image
import omegaconf
import hydra

from garmentds.real.real_env_desktop import RealEnvDesktop, RealEnvDesktopCfg, RobotlCfg, RobotrCfg
import garmentds.real.utils as real_utils

from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear() # clear the global hydra instance to avoid issues with multiple runs

@hydra.main(config_path="../config/run", config_name=pathlib.Path(__file__).stem, version_base='1.3')
def main(cfg: omegaconf.DictConfig):
    env = RealEnvDesktop(RealEnvDesktopCfg(
        api_robotl_cfg=RobotlCfg(default_acc=0.5, default_vel=0.5),
        api_robotr_cfg=RobotrCfg(use=False),
    ))
        
    class IterFunc:
        def __init__(self, clothes_name="."):
            self.curr_img_idx = 0
            self.clothes_name = clothes_name
            os.makedirs(self.clothes_name, exist_ok=True)

        def __call__(self):
            while True:
                prompt = (
                    f"current clothes name: {self.clothes_name} {self.curr_img_idx}\n"
                    "what to do next?\n"
                    "c to continue\n"
                    "s to skip clothes initialization\n"
                    "r to rename clothes\n"
                    "q to quit\n"
                    "enter: "
                )   
                ans = input(prompt)
                if ans not in ["c", "s", "r", "q"]:
                    print("invalid input, please try again")
                else:
                    break
            
            if ans == "c":
                print("[INFO] get random pick and place to initialize clothes")
                if cfg.category in ["tshirt", "vest_close"]:
                    rot_z = 0.
                elif cfg.category in ["trousers"]:
                    rot_z = -np.pi / 6
                elif cfg.category in ["hooded_close"]:
                    rot_z = 0.
                else:
                    raise NotImplementedError(cfg.category)
                camera_pose = env.move_camera_to_init_clothes(rot_z = rot_z)
                while True:
                    p1, p2, hand = env.get_random_init_clothes_xyz(camera_pose)
                    ans = input("press r to retry, any other key to continue: ")
                    if ans != "r":
                        break
                
                print(f"[INFO] pick and place clothes at {p1}, {p2}, {hand}")
                env.pick_and_place(p1, p2 - p1, hand)
            elif ans == "s":
                pass
            elif ans == "r":
                new_name = input("enter new clothes name: ")
                self.clothes_name = new_name
                os.makedirs(self.clothes_name, exist_ok=True)
                self.curr_img_idx = 0
                return 0
            elif ans == "q":
                return 1
            else:
                raise NotImplementedError(ans)
            
            for _ in range(2): # take two pictures
                print(f"[INFO] collecting real images, current image index: {self.curr_img_idx}")
                camera_pose = env.get_camera_pose_to_take_picture(
                    rot_z=np.radians(np.array([15, 45]) + np.random.uniform(low=-5, high=5, size=2))[self.curr_img_idx % 2],
                    rot_y=-0.1, 
                )
                env.move_camera(camera_pose)

                obs = env.get_obs()
                np.save(os.path.join(self.clothes_name, f"{str(self.curr_img_idx).zfill(3)}_depth.npy"), obs["depth"])
                Image.fromarray(obs["color"]).save(os.path.join(self.clothes_name, f"{str(self.curr_img_idx).zfill(3)}_color.png"))
                self.curr_img_idx += 1
                
            env.move_robot_to_init()
            return 0
    
    # start the main loop
    iter_func = IterFunc(cfg.clothes_name)
    while True:
        try:
            ret = iter_func()
            if ret != 0:
                break
        except real_utils.SIGUSR1Exception:
            print("catch SIGUSR1 Exception!")
            input()
        except Exception as e:
            raise e
    
    env.close()


if __name__ == '__main__':
    main()