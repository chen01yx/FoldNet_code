import taichi as ti
import numpy as np
import trimesh

ti.init(arch=ti.gpu) 

v_field = ti.Vector.field(3, dtype=ti.f32, shape=20000)
f_field = ti.Vector.field(3, dtype=ti.i32, shape=20000)
ray_origins_field = ti.Vector.field(3, dtype=ti.f32, shape=6000)
ray_directions_field = ti.Vector.field(3, dtype=ti.f32, shape=6000)
result = ti.field(dtype=ti.i32, shape=6000)

def detect_obstruction(
    mesh_obj_path: str,
    point_idxs: list,
) -> np.ndarray:
    # Load mesh using trimesh
    mesh = trimesh.load_mesh(mesh_obj_path, process=False)
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    num_faces = faces.shape[0]

    # Setting up the rays
    num_rays = len(point_idxs)
    ray_origins = vertices[point_idxs]
    ray_directions = np.array([
        [0.0, 1.0, 0.0] for _ in range(num_rays)
    ], dtype=np.float32)

    # Initialize fields with data
    global v_field, f_field, ray_origins_field, ray_directions_field, result
    v_field.from_numpy(vertices)
    f_field.from_numpy(faces)
    ray_origins_field.from_numpy(ray_origins)
    ray_directions_field.from_numpy(ray_directions)
    result.fill(1)

    check_rays_intersection(num_faces, num_rays)
    return result.to_numpy()[:num_rays]


@ti.func
def ray_triangle_intersect(ray_idx, orig, dir, v0, v1, v2, eps=1e-6):
    M1 = ti.Matrix.cols([dir, v2 - v0, v2 - v1])
    M2 = ti.Matrix.cols([v2 - orig])

    if M1.determinant() != 0:
        M1_inv = M1.inverse()
        params = M1_inv @ M2
        if params[0, 0] > eps and \
           params[1, 0] >= eps and params[1, 0] <= 1 and \
           params[2, 0] >= eps and params[2, 0] <= 1 and \
           params[1, 0] + params[2, 0] <= 1:
            #print(params)
            result[ray_idx] = 0

@ti.kernel
def check_rays_intersection(num_faces: ti.i32, num_rays: ti.i32):
    for ray_idx in range(num_rays):  # Parallel over rays
        ray_orig = ray_origins_field[ray_idx]
        ray_dir = ray_directions_field[ray_idx]
        for i in range(num_faces):  # Iterate over faces
            v0 = v_field[f_field[i][0]]
            v1 = v_field[f_field[i][1]]
            v2 = v_field[f_field[i][2]]
            ray_triangle_intersect(ray_idx, ray_orig, ray_dir, v0, v1, v2)
            if result[ray_idx] <= 1e-6:
                break


if __name__ == '__main__':
    pass