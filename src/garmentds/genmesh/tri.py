from typing import Literal

import trimesh
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.spatial import KDTree

import omegaconf

from CGAL.CGAL_Kernel import Point_2
from CGAL.CGAL_Mesh_2 import Mesh_2_Constrained_Delaunay_triangulation_2, Mesh_2_parameters
from CGAL import CGAL_Mesh_2

from .spatial import SpatialPartition, detect_intersect_kernel, SparseMask


def generate_vert_on_boundary_edge(
    start: tuple[float, float], 
    control: list[tuple[float, float]],
    end: tuple[float, float],
    reverse_control: bool, 
    include: Literal["start", "end", "both", "none"], 
    dx: float,
    dense_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate vertices on a boundary edge. """
    if reverse_control: control = list(reversed(control))
    points_x = np.array([xy[0] for xy in [start] + control + [end]])
    points_y = np.array([xy[1] for xy in [start] + control + [end]])
    points_t = np.linspace(0, 1, len(points_x))

    t_dense = np.linspace(0., 1., dense_n)
    x_dense = CubicSpline(points_t, points_x)(t_dense)
    y_dense = CubicSpline(points_t, points_y)(t_dense)

    segment_length = np.linalg.norm(np.concatenate([
        x_dense[1:, None] - x_dense[:-1, None], 
        y_dense[1:, None] - y_dense[:-1, None]
    ], axis=1), axis=1)
    s_dense = np.pad(segment_length.cumsum(), (1, 0))
    l = s_dense[-1]
    n = int(l / dx) + 2

    idx = np.round(interp1d(s_dense, t_dense)(np.linspace(0., l, n)) * (dense_n - 1)).astype(int)
    idx[0], idx[-1] = 0, dense_n - 1 # force end point
    vert = np.concatenate([x_dense[idx, None], y_dense[idx, None]], axis=1)
    affine_param = t_dense[idx]

    if include == "start":
        s = slice(None, -1)
    elif include == "end":
        s = slice(1, None)
    elif include == "none":
        s = slice(1, -1)
    elif include == "both":
        s = slice(None, None)
    else:
        raise ValueError(include)
    return vert[s], affine_param[s]


def modify_demoninator(a: np.ndarray, eps=1e-7):
    a = a.copy()
    a[np.where(np.abs(a) < eps)] = eps
    return a


def generate_vert_within_boundary(
    polygon_xy: np.ndarray,
    target_num: int,
    batch_size=1024,
    max_iter=1024,
):
    xy = np.array(polygon_xy) # [P, 2]
    curr_good_num = 0
    good_sample = []

    for _ in range(max_iter):
        # generate one batch
        sample = np.random.rand(batch_size, 2) # [S, 2]
        sample = sample * (xy.max(axis=0) - xy.min(axis=0)) + xy.min(axis=0)

        poly_idx_1 = np.arange(xy.shape[0]) # [P]
        poly_idx_2 = np.mod(poly_idx_1 + 1, xy.shape[0]) # [P]

        x, y = sample[:, None, 0], sample[:, None, 1] # [S, 1]
        x1, y1 = xy[None, poly_idx_1, 0], xy[None, poly_idx_1, 1] # [1, P]
        x2, y2 = xy[None, poly_idx_2, 0], xy[None, poly_idx_2, 1] # [1, P]

        intersection_x = np.logical_and(
            np.logical_xor(y1 > y, y2 > y),
            x < (x2 - x1) * (y - y1) / modify_demoninator(y2 - y1) + x1,
        ) # [S, P]
        inside = (np.mod(np.sum(intersection_x, axis=1), 2) == 1) # [S]
        selected_sample = sample[np.where(inside)[0], :] # [S', 2]

        new_good_num = selected_sample.shape[0]
        curr_good_num += new_good_num
        if curr_good_num >= target_num:
            good_sample.append(selected_sample[:target_num + new_good_num - curr_good_num])
            break
        else:
            good_sample.append(selected_sample)
    
    ans = np.concatenate(good_sample, axis=0)
    return ans


def delaunay_in_boundary(vert_boundary: np.ndarray, vert_interior: np.ndarray, optimize_mesh: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Use Constrained_Delaunay_triangulation_2 in CGAL. The 'vert_interior' may change due to optimization process. 

    return
    - vert_interior_new: [I, 2]
    - face: [F, 2]
    """

    # https://github.com/CGAL/cgal-swig-bindings/blob/main/examples/python/polygonal_triangulation.py
    class FaceInfo2(object):
        def __init__(self):
            self.nesting_level = -1

        def in_domain(self):
            return (self.nesting_level % 2) != 1
    
    def mark_domains(ct, start_face, index, edge_border, face_info):
        if face_info[start_face].nesting_level != -1:
            return
        queue = [start_face]
        while queue != []:
            fh = queue[0]  # queue.front
            queue = queue[1:]  # queue.pop_front
            if face_info[fh].nesting_level == -1:
                face_info[fh].nesting_level = index
                for i in range(3):
                    e = (fh, i)
                    n = fh.neighbor(i)
                    if face_info[n].nesting_level == -1:
                        if ct.is_constrained(e):
                            edge_border.append(e)
                        else:
                            queue.append(n)

    def mark_domain(cdt):
        face_info = {}
        for face in cdt.all_faces():
            face_info[face] = FaceInfo2()
        index = 0
        border = []
        mark_domains(cdt, cdt.infinite_face(), index + 1, border, face_info)
        while border != []:
            e = border[0]  # border.front
            border = border[1:]  # border.pop_front
            n = e[0].neighbor(e[1])
            if face_info[n].nesting_level == -1:
                lvl = face_info[e[0]].nesting_level + 1
                mark_domains(cdt, n, lvl, border, face_info)
        return face_info
    
    cdt = Mesh_2_Constrained_Delaunay_triangulation_2()

    # insert polygon
    polygon_cgal = [Point_2(x, y) for x, y in vert_boundary]
    handles = [cdt.insert(p) for p in polygon_cgal]
    n_polygon = len(polygon_cgal)
    for i in range(n_polygon):
        cdt.insert_constraint(handles[i], handles[(i + 1) % n_polygon])
    
    # insert interior
    for x, y in vert_interior:
        cdt.insert(Point_2(x, y))
    
    # optimize
    if optimize_mesh:
        params = Mesh_2_parameters()
        params.set_max_iteration_number(10)
        CGAL_Mesh_2.lloyd_optimize_mesh_2(cdt, params)
    
    # index points after optimize mesh
    def Point_2_to_tuple(p: Point_2): return (p.x(), p.y())
    all_points = [v.point() for v in cdt.finite_vertices()] # vert_boundary + vert_interior_new
    points_index = {Point_2_to_tuple(p): i for i, p in enumerate(all_points)}
    
    # save result
    face_info = mark_domain(cdt)
    all_face = []
    for f, finfo in face_info.items():
        if finfo.in_domain():
            vids = [points_index[Point_2_to_tuple(f.vertex(i).point())] for i in range(3)]
            all_face.append(vids)
    
    vert_interior_new = np.array([Point_2_to_tuple(p) for p in all_points[vert_boundary.shape[0]:]])
    if len(vert_interior) == 0: 
        vert_interior_new = vert_interior_new.reshape((0, 2))
    all_face_np = np.array(all_face)

    # print(f"vert_boundary:{vert_boundary.shape}, vert_interior_new:{vert_interior_new.shape}")
    
    return vert_interior_new, all_face_np


def post_process_face_2d(vert: np.ndarray, face: np.ndarray, norm_up: bool):
    """make sure all faces' norm are up or down"""
    def cross2d(a, b): return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    face = face.copy()
    p0, p1, p2 = vert[face[:, 0]], vert[face[:, 1]], vert[face[:, 2]]

    flip_idx = np.where(np.logical_xor(cross2d(p1 - p0, p2 - p0) > 0., norm_up))[0]
    face[flip_idx] = face[flip_idx][:, [0, 2, 1]]

    return face


def vert_2d_to_3d_new(vert: np.ndarray, vert_boundary_all: list[np.ndarray], vert_connect_all: list[np.ndarray], max_z: float, width: float):
    """
    use the distance from vert to connected edges to compute z.

    args
    - vert: [P, 2], float
    - vert_boundary: [B, 2], float
    - vert_connect: [B, ], int

    return
    - vert3d: [P, 3], float
    """
    v = vert # [P, 2]
    P = vert.shape[0]
    eps = 1e-7
    
    # find all edges which are back-and-front connected within the same list
    a, b = [], []
    for vert_boundary, vert_connect in zip(vert_boundary_all, vert_connect_all):
        B = vert_boundary.shape[0]
        assert vert_connect.shape[0] == B, f"vert_boundary {vert_boundary.shape}, vert_connect {vert_connect.shape}"
        for i in range(B):
            j = (i + 1) % B
            if vert_connect[i] and vert_connect[j]:
                a.append(vert_boundary[i]), b.append(vert_boundary[j])
    assert len(a) > 0 and len(b) > 0, "all vertex is not connected"
    a, b = np.array(a), np.array(b) # [C, 2]

    # compute distance from vert to connected edges
    def proj(x: np.ndarray, y: np.ndarray): 
        return np.sum(x * y, axis=len(x.shape) - 1) / np.clip(np.sum(x * x, axis=len(x.shape) - 1), eps, None)
    t = np.clip(proj((b - a)[None, :, :], v[:, None, :] - a[None, :, :]), 0., 1.) # [P, C]
    d_all = np.linalg.norm(
        v[:, None, :] - (a[None, :, :] * (1. - t)[:, :, None] + b[None, :, :] * t[:, :, None]), axis=2
    )
    d = np.min(d_all, axis=1) # [P]
    c_idx = np.argmin(d_all, axis=1) # [P], [0, ..., C-1]

    def make_ellipse_map(d0: float, z0: float, d: np.ndarray, sample_n: int = 100):
        theta = np.linspace(0., np.pi / 2, sample_n)
        magic_power = 1.0
        pd = d0 * (np.cos(theta) ** magic_power)
        pz = z0 * (np.sin(theta) ** magic_power)
        accumulate_len = np.cumsum(np.linalg.norm(np.diff(np.concatenate([pd[:, None], pz[:, None]], axis=1), axis=0), axis=1))
        accumulate_percentage = np.concatenate([[0], accumulate_len]) / accumulate_len[-1]
        theta_interp = interp1d(accumulate_percentage, theta)(d / d0)

        d_new = d0 * (1. - (np.cos(theta_interp) ** magic_power))
        z_new = z0 * (np.sin(theta_interp) ** magic_power)
        return d_new, z_new

    # given d, compute corresponding z
    d0, z0 = width / 2, max_z / 2 # / 2 to make the generated mesh looks similar to the original method
    d_new, z_new = make_ellipse_map(d0, z0, np.clip(d, 0., d0))

    proj_point = a[c_idx] + (b[c_idx] - a[c_idx]) * t[np.arange(P), c_idx, None]
    proj_vec = (v - proj_point) / np.clip(d, a_min=eps, a_max=None)[:, None]
    vert3d = np.concatenate([v, z_new[:, None]], axis=1)
    move_xy_idx = np.where(d < d0)[0]
    vert3d[move_xy_idx, :2] = proj_point[move_xy_idx] + proj_vec[move_xy_idx] * (d_new)[move_xy_idx, None]

    return vert3d


def vert_2d_to_3d(vert: np.ndarray, vert_boundary_all: list[np.ndarray], vert_connect_all: list[np.ndarray], max_z: float, width: float):
    """
    use the distance from vert to connected edges to compute z.

    args
    - vert: [P, 2], float
    - vert_boundary: [B, 2], float
    - vert_connect: [B, ], int

    return
    - vert3d: [P, 3], float
    """
    # find all edges which are back-and-front connected within the same list
    a, b = [], []
    for vert_boundary, vert_connect in zip(vert_boundary_all, vert_connect_all):
        B = vert_boundary.shape[0]
        assert vert_connect.shape[0] == B, f"vert_boundary {vert_boundary.shape}, vert_connect {vert_connect.shape}"
        for i in range(B):
            j = (i + 1) % B
            if vert_connect[i] and vert_connect[j]:
                a.append(vert_boundary[i]), b.append(vert_boundary[j])
    assert len(a) > 0 and len(b) > 0, "all vertex is not connected"
    a, b = np.array(a), np.array(b) # [C, 2]

    # compute distance from vert to connected edges
    def proj(x: np.ndarray, y: np.ndarray, eps=1e-7): 
        return np.sum(x * y, axis=len(x.shape) - 1) / np.clip(np.sum(x * x, axis=len(x.shape) - 1), eps, None)
    t = np.clip(proj((b - a)[None, :, :], vert[:, None, :] - a[None, :, :]), 0., 1.)
    d = np.min(np.linalg.norm(
        vert[:, None, :] - (a[None, :, :] * (1. - t)[:, :, None] + b[None, :, :] * t[:, :, None]), axis=2
    ), axis=1) # [P]

    # given d, compute corresponding z
    # d_ = np.clip(d / width, 0., 1.)
    # z_ = (np.sqrt(1 + 8 * d_) - 1) / 2 # d_ * (d_ + 1) = 2 * z_
    d_ = np.clip(d / width / 2, 0., 1.)
    z_ = np.sqrt(1. - np.square(1. - d_)) # z_ * z_ + (1 - d_) * (1 - d_) = 1
    z = max_z * z_ # [P]

    vert3d = np.concatenate([vert, z[:, None]], axis=1)
    return vert3d


_spatial_partition = None

def _init_spatial_partition():
    return SpatialPartition(
        spatial_partition_cfg=omegaconf.DictConfig(dict(
            bounds=[[0., 0., 0.], [1., 1., 1.]],
            xyz_size=[128, 128, 16],
            max_spatial_cell_size=1024,
            xyz_block_size=8,
            spatial_cell_chunk_size=16,
            max_bb_occupy_num=4096, # 32 * 32 * 4
        )),
        global_cfg=omegaconf.DictConfig(dict(
            batch_size=1,
            default_float="float32",
            default_int="int32",
            torch_device="cpu",
        ))
    )


def check_self_intersection(
    vert: np.ndarray,
    face: np.ndarray,
    edge: np.ndarray,
    eps = 1e-7,
    int_type=np.int32,
    float_type=np.float32,
):
    """
    args
    - vert: [V, 3], float
    - face: [F, 3], int
    - edge: [E, 2], int
    """

    # norm v
    vert = (vert - vert.min(axis=0)) / (vert.max(axis=0) - vert.min(axis=0) + eps)
    
    global _spatial_partition
    if _spatial_partition is None:  _spatial_partition = _init_spatial_partition()
    _spatial_partition.deactivate_all_kernel()

    mask_face_edge = SparseMask(face.shape[0], edge.shape[0])
    ans_face_edge = SparseMask(face.shape[0], edge.shape[0])

    vert_color = np.zeros_like(vert).astype(float_type)
    ans = detect_intersect_kernel(
        _spatial_partition, mask_face_edge.mask, ans_face_edge.mask, 
        vert.astype(float_type), face.astype(int_type), edge.astype(int_type), vert_color, eps
    )
    if ans: ans_face_edge.print_active()

    return ans, vert_color


def sanity_check(
    mesh: trimesh.Trimesh,
    self_intersection_eps = 1e-7,
    min_vert_dist = 1e-5,
    skip_check_self_intersection = False, 
):
    success = True
    mesh_processed = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    if not skip_check_self_intersection:
        # check self intersection
        retval, vert_color = check_self_intersection(mesh_processed.vertices, mesh_processed.faces, mesh_processed.edges, eps=self_intersection_eps)
        if retval:
            print("[ERROR] check self intersection failed ...")
            success = False
            trimesh.Trimesh(vertices=mesh_processed.vertices, faces=mesh_processed.faces, vertex_colors=vert_color).export("debug.obj")
    
    # check min vert distance
    vert = mesh_processed.vertices
    kdtree = KDTree(vert.copy())
    result = kdtree.query(vert, k = 2)[0][:, 1]
    if result.min() < min_vert_dist:
        print("[ERROR] min vertex distance check failed ...")
        print(np.where(result < min_vert_dist))
        success = False

    return success


def remove_repeat_vertex_and_build_vertex_map(mesh_with_repeat_vert: trimesh.Trimesh):
    """
    remove repeat vertex in 'mesh_with_repeat_vert', and build a map from vertex in 'mesh_with_repeat_vert' to vertex in 'mesh_without_repeat_vert'.

    Usage: mesh_with_repeat_vert.vertices = mesh_without_repeat_vert.vertices[vert_w_to_wo]
    """
    mesh_without_repeat_vert = trimesh.Trimesh(
        vertices=mesh_with_repeat_vert.vertices, 
        faces=mesh_with_repeat_vert.faces, 
        process=True,
    )
    vert_xyz_to_idx = {}
    for i, v in enumerate(mesh_without_repeat_vert.vertices):
        vert_xyz_to_idx[tuple(v.astype(np.float32))] = i
    vert_w_to_wo = []
    for i, v in enumerate(mesh_with_repeat_vert.vertices):
        vert_w_to_wo.append(vert_xyz_to_idx[tuple(v.astype(np.float32))])
    vert_w_to_wo = np.array(vert_w_to_wo)
    """[NV_R] -> [0, NV_S)"""
    return mesh_without_repeat_vert, vert_w_to_wo