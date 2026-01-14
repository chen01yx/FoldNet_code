import json
import argparse
import copy
import os
from collections import defaultdict

import taichi as ti
import trimesh
import numpy as np

from garmentds.genmesh.template import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", type=str, required=True)
    parser.add_argument("--output_path", "-o", type=str, required=True)
    args = parser.parse_args()
    return args


def compute_part_area_and_len(
    mesh: trimesh.Trimesh, 
    vert_info: list[str], 
    boundary_idx: dict[str, list[int]], 
    boundary_dx: float
):
    mesh_area = defaultdict(float)
    boundary_len = defaultdict(float)
    for fid, f in enumerate(mesh.faces):
        part = set([vert_info[v] for v in f])
        if len(part) == 1: # only consider faces containing one part
            mesh_area[list(part)[0]] += mesh.area_faces[fid]
    
    for p, b in boundary_idx.items():
        boundary_len[p] = len(b) * boundary_dx
    return mesh_area, boundary_len


def modify_cfg(mesh: trimesh.Trimesh, info: dict, target_dx_ratio = 0.006):
    target_dx = (mesh.bounds[1] - mesh.bounds[0])[:2].mean() * target_dx_ratio
    cfg = info["cfg"]
    cfg = copy.deepcopy(cfg)
    mesh_area, boundary_len = compute_part_area_and_len(
        mesh, info["triangulation"]["vert_info"], 
        info["triangulation"]["boundary_idx"], info["cfg"]["boundary_dx"]
    )

    area_per_vert = target_dx ** 2 * (3 ** 0.5 / 2) # assume every angle is 60 degree angle
    cfg["boundary_dx"] = target_dx
    for p in mesh_area.keys():
        n0 = int(mesh_area[p] / area_per_vert + 0.5)
        n = max(int((mesh_area[p] - boundary_len[p] / target_dx * area_per_vert / 2) / area_per_vert + 0.5), 0)
        print(f"{p}: {n0} vs {n}")
        cfg["interior_num"][p] = n

    return cfg


def main():
    args = get_args()
    input_path = str(args.input_path)
    output_path = str(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ti.init(
        arch=ti.cuda, device_memory_GB=4, debug=False, 
        advanced_optimization=True, fast_math=False, offline_cache=True
    )

    with open(input_path[:-len(".obj")] + "_info.json", "r") as f:
        info = json.load(f)
        mesh = trimesh.load_mesh(input_path, process=False)
    garment_name = info["meta"]["name"]
    garment = garment_dict[garment_name](**modify_cfg(mesh, info, garment_name))
    garment.triangulation()
    garment.quick_export(output_path)


if __name__ == "__main__":
    main()