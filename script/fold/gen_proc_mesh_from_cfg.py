import subprocess
import os
import multiprocessing
import shutil
import random
import json
from dataclasses import dataclass, field
import copy
from typing import Optional, Callable
import datetime
import os
import argparse
from pathlib import Path

import taichi as ti
import numpy as np
import torch
import trimesh

from garmentds.genmesh.template import *
from garmentds.genmesh.cfg import generate_cfg
from garmentds.foldenv.preproc_mesh import modify_cfg
import garmentds.common.utils as utils


mesh_size = "tiny"
device_memory_GB = 4
skip_check_self_intersection = True
regenerate_cfg = False
target_dx_ratio_dict = dict(
    xxtiny = dict(
        tshirt_sp=0.024,
        trousers=0.030,
    ),
    xtiny = dict(
        tshirt_sp=0.014,
        trousers=0.017,
    ),
    tiny = dict(
        tshirt_sp=0.008,
        trousers=0.010,
    ),
    small = dict(
        tshirt_sp=0.006,
        trousers=0.0075,
    ),
    medium = dict(
        tshirt_sp=0.004,
        trousers=0.005,
    ),
    large = dict(
        tshirt_sp=0.0032,
        trousers=0.004,
    )
)[mesh_size]


@dataclass
class Cfg:
    cudas: list[int] = field(default_factory=lambda: list(range(0, 8)))
    num_workers: int = 32
    job_ids: list[int] = field(default_factory=lambda: list(range(1000)))
    category: str = "tshirt_sp"
    input_dir: str = "data/asset/texture/tshirt_sp_20250629"
    out_dir: str = f"data/fold/mesh/gen_proc/tshirt_sp_20250629_{mesh_size}"

    def get_input_obj_dir(self, i: int):
        return f"{self.input_dir}/{i}"
    
    def get_output_sub_dir(self, i: int):
        return f"{i}"
    
    def get_output_obj_dir(self, i: int):
        return f"{self.out_dir}/{self.get_output_sub_dir(i)}"
    
    def __post_init__(self):
        for i in self.job_ids:
            assert Path(self.get_input_obj_dir(i)).exists(), f"{self.get_input_obj_dir(i)} not exists"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subprocess", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--category", type=str)
    args = parser.parse_args()
    return args


def run_job(args):
    utils.seed_all(args.seed)
    ti.init(
        arch=ti.cuda, device_memory_GB=device_memory_GB, debug=False, random_seed=args.seed, 
        advanced_optimization=True, fast_math=False, offline_cache=True, 
    )

    input_dir = args.input_dir
    output_dir = args.output_dir
    garment_name = args.category
    
    with open(os.path.join(input_dir, "mesh_info.json"), "r") as f:
        raw_info = json.load(f)
    mesh = trimesh.load_mesh(os.path.join(input_dir, "mesh.obj"), process=False)

    garment = garment_dict[garment_name](**modify_cfg(mesh, raw_info, target_dx_ratio=target_dx_ratio_dict[garment_name]))
    mesh, info = garment.triangulation(skip_check_self_intersection=skip_check_self_intersection)

    if info["success"]:
        os.makedirs(output_dir, exist_ok=True)
        garment.quick_export(os.path.join(output_dir, "mesh.obj"))

        for f in ["material.mtl", "material_0.png"]:
            shutil.copy(os.path.join(input_dir, f), os.path.join(output_dir, f))
    
    elif regenerate_cfg:
        with open(Path(input_dir).parent / "generate_cfg.json", "r") as f:
            generate_cfg_args = json.load(f)
        assert garment_name == generate_cfg_args["category"], (garment_name, generate_cfg_args["category"])
        while True:
            garment = garment_dict[garment_name](**generate_cfg(
                garment_name, 
                description=generate_cfg_args["description"], 
                **generate_cfg_args["cfg"],
            ).asdict())
            mesh, info = garment.triangulation(skip_check_self_intersection=skip_check_self_intersection)

            garment = garment_dict[garment_name](**modify_cfg(mesh, garment.get_info_to_export(), target_dx_ratio=target_dx_ratio_dict[args.category]))
            mesh, info = garment.triangulation(skip_check_self_intersection=skip_check_self_intersection)

            if info["success"]:
                os.makedirs(output_dir, exist_ok=True)
                garment.quick_export(os.path.join(output_dir, "mesh.obj"))
                break
    
        for f in ["material.mtl", "material_0.png"]:
            shutil.copy(os.path.join(input_dir, f), os.path.join(output_dir, f))


class Main:
    def __init__(self, cfg: Cfg):
        self.cfg = copy.deepcopy(cfg)
        self.job_queue: multiprocessing.Queue[Optional[int]] = multiprocessing.Queue(maxsize=8)
    
    def worker(self, worker_id: int):
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{self.cfg.cudas[worker_id % len(self.cfg.cudas)]}"
        while True:
            job_id = self.job_queue.get()
            if job_id is None:
                break
            seed = job_id
            attemp_cnt = 0
            while True:
                input_obj_dir = self.cfg.get_input_obj_dir(job_id)
                output_obj_dir = self.cfg.get_output_obj_dir(job_id)
                os.makedirs(output_obj_dir, exist_ok=True)
                cmd = (
                    f"python {__file__} --subprocess " + 
                    f"--seed {seed} " + 
                    f"--input_dir {input_obj_dir} " + 
                    f"--output_dir {output_obj_dir} " +
                    f"--category {self.cfg.category}"
                )
                with open(os.path.join(output_obj_dir, f"out_{os.getpid()}_{attemp_cnt}.log"), "w") as f:
                    ret = subprocess.run(cmd, shell=True, stdout=f, stderr=f)
                    f.flush()
                print(f"job_id:{job_id} returncode:{ret.returncode}")
                if ret.returncode == 0 or not regenerate_cfg:
                    break
                else:
                    seed = np.random.randint(0, 2**31)
                attemp_cnt += 1

    def run(self):
        # init mp
        process_list: list[multiprocessing.Process] = []
        for i in range(self.cfg.num_workers):
            p = multiprocessing.Process(target=self.worker, args=(i,), daemon=True)
            process_list.append(p)
            p.start()

        # append jobs
        for i in self.cfg.job_ids:
            self.job_queue.put(i)

        for i in range(self.cfg.num_workers):
            self.job_queue.put(None)

        for p in process_list:
            p.join()
        
        # output
        def generate_success(job_id: int):
            return (Path(self.cfg.get_output_obj_dir(job_id)) / "mesh.obj").exists()
        
        with open(Path(self.cfg.out_dir) / "meta.json", "w") as f:
            success_subdir = [
                self.cfg.get_output_sub_dir(i) 
                for i in self.cfg.job_ids if generate_success(i)
            ]
            num_success = len(success_subdir)
            json.dump(dict(
                num_success = num_success,
                success_subdir = success_subdir,
                mesh_size = mesh_size,
            ), f, indent=4)


def main():
    args = get_args()
    if args.subprocess:
        run_job(args)
    else:
        Main(Cfg()).run()


if __name__ == '__main__':
    main()