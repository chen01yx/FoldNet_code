import os
import json
import trimesh
import numpy as np
import batch_urdf
import trimesh.transformations as tra


def get_mesh(traj_dir: str, frame_idx: int):
    assert "traj" in os.path.basename(traj_dir)
    cfg_file = os.path.join(traj_dir, ".tmp", str(frame_idx).zfill(4), "cfg.json")
    with open(cfg_file, "r") as f:
        cfg = json.load(f)
    init_file = os.path.join(os.path.dirname(traj_dir), ".render_process", "init_message_0.json")
    with open(init_file, "r") as f:
        init_msg = json.load(f)
    cloth_file = os.path.join(traj_dir, ".tmp", str(frame_idx).zfill(4), "cloth_v_sim.npy")
    cloth_v_sim = np.load(cloth_file)

    tm_mesh_raw: trimesh.Trimesh = trimesh.load(init_msg["cloth_obj_path"], process=False)
    tm_mesh_sim = trimesh.Trimesh(vertices=tm_mesh_raw.vertices, faces=tm_mesh_raw.faces)
    vert_xyz_to_idx = {}
    for i, v in enumerate(tm_mesh_sim.vertices):
        vert_xyz_to_idx[tuple(v)] = i
    vert_ren_to_sim = []
    for i, v in enumerate(tm_mesh_raw.vertices):
        vert_ren_to_sim.append(vert_xyz_to_idx[tuple(v)])
    vert_ren_to_sim = np.array(vert_ren_to_sim)
    tm_mesh_sim.vertices = cloth_v_sim

    urdf = batch_urdf.URDF(1, urdf_path = init_msg["urdf_path"])
    urdf.update_base_link_transformation(cfg["base_link"], urdf.tensor([cfg["base_pose"]]))
    urdf.update_cfg(urdf.cfg_f2t(cfg["qpos"]))

    table = trimesh.primitives.Box(
        extents=[1.0, 0.8, 0.1], transform=tra.translation_matrix([0, 0, -0.05])
    )

    concat_mesh: trimesh.Trimesh = trimesh.util.concatenate([
        urdf.get_scene(0, use_collision_instead_of_visual=True),
        tm_mesh_sim, table,
    ])
    return trimesh.Trimesh(vertices=concat_mesh.vertices, faces=concat_mesh.faces)