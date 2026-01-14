import taichi as ti

import torch
import numpy as np
import omegaconf

import garmentds.common.taichi as taichi_utils


@ti.dataclass
class BoundingBox:
    bounds: ti.types.matrix(2, 3, float)


@ti.func
def point_bounding_box_func(a: ti.math.vec3) -> BoundingBox:
    bounds = ti.Matrix.zero(float, 2, 3)
    bounds[0, :] = a
    bounds[1, :] = a
    return BoundingBox(bounds=bounds)


@ti.func
def line_bounding_box_func(a: ti.math.vec3, b: ti.math.vec3) -> BoundingBox:
    bounds = ti.Matrix.zero(float, 2, 3)
    bounds[0, :] = ti.min(a, b)
    bounds[1, :] = ti.max(a, b)
    return BoundingBox(bounds=bounds)


@ti.func
def triangle_bounding_box_func(a: ti.math.vec3, b: ti.math.vec3, c: ti.math.vec3) -> BoundingBox:
    bounds = ti.Matrix.zero(float, 2, 3)
    bounds[0, :] = ti.min(a, b, c)
    bounds[1, :] = ti.max(a, b, c)
    return BoundingBox(bounds=bounds)


@ti.func
def merge_bounding_box_func(bb1: BoundingBox, bb2: BoundingBox) -> BoundingBox:
    bounds = ti.Matrix.zero(float, 2, 3)
    bounds[0, :] = ti.min(bb1.bounds[0, :], bb2.bounds[0, :])
    bounds[1, :] = ti.max(bb1.bounds[1, :], bb2.bounds[1, :])
    return BoundingBox(bounds=bounds)


@ti.data_oriented
class SpatialPartition:
    def __init__(self, spatial_partition_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig) -> None:
        self._batch_size: int = int(global_cfg.batch_size)
        self._dtype: torch.dtype = getattr(torch, global_cfg.default_float)
        assert isinstance(self._dtype, torch.dtype)
        self._dtype_int: torch.dtype = getattr(torch, global_cfg.default_int)
        assert isinstance(self._dtype_int, torch.dtype)
        self._device: str = str(global_cfg.torch_device)

        self._bounds = taichi_utils.GLOBAL_CREATER.MatrixField(n=2, m=3, dtype=float, shape=())
        """float, [None][2, 3]"""
        self._bounds.from_torch(torch.tensor(spatial_partition_cfg.bounds, dtype=self._dtype, device=self._device))

        self._xyz_size = taichi_utils.GLOBAL_CREATER.VectorField(n=3, dtype=int, shape=())
        """int, [None][3, ]"""
        self._xyz_size.from_torch(torch.tensor(spatial_partition_cfg.xyz_size, dtype=self._dtype_int, device=self._device))
        self._max_spatial_cell_size = int(spatial_partition_cfg.max_spatial_cell_size)

        self._xyz_block_size = int(spatial_partition_cfg.xyz_block_size)
        self._spatial_cell_chunk_size = int(spatial_partition_cfg.spatial_cell_chunk_size)

        self._spatial_cell = ti.field(int)
        """int, [B, SX, SY, SZ, IL]"""
        self._spatial_pointer = ti.root.pointer(ti.ijkl, (
            self._batch_size,
            (spatial_partition_cfg.xyz_size[0] + self._xyz_block_size - 1) // self._xyz_block_size,
            (spatial_partition_cfg.xyz_size[1] + self._xyz_block_size - 1) // self._xyz_block_size,
            (spatial_partition_cfg.xyz_size[2] + self._xyz_block_size - 1) // self._xyz_block_size,
        ))
        self._spatial_pixel = self._spatial_pointer.dense(
            ti.ijkl,
            (1, self._xyz_block_size, self._xyz_block_size, self._xyz_block_size)
        )
        self._spatial_dynamic = self._spatial_pixel.dynamic(ti.axes(4), self._max_spatial_cell_size, chunk_size=self._spatial_cell_chunk_size)
        self._spatial_dynamic.place(self._spatial_cell)

        taichi_utils.GLOBAL_CREATER.LogSparseField(shape=(
            self._batch_size,
            (spatial_partition_cfg.xyz_size[0] + self._xyz_block_size - 1) // self._xyz_block_size * self._xyz_block_size,
            (spatial_partition_cfg.xyz_size[1] + self._xyz_block_size - 1) // self._xyz_block_size * self._xyz_block_size,
            (spatial_partition_cfg.xyz_size[2] + self._xyz_block_size - 1) // self._xyz_block_size * self._xyz_block_size,
            self._max_spatial_cell_size,
        ))

        # for safety, prevent large serialized loop
        self._max_bb_occupy_num = int(spatial_partition_cfg.max_bb_occupy_num)
        self._fatal_flag: ti.ScalarField = taichi_utils.GLOBAL_CREATER.ScalarField(dtype=bool, shape=(self._batch_size, ))
        """bool, [B, ]"""
    
    def deactivate_all_kernel(self) -> torch.Tensor:
        """Clear fatal_flag and Return old fatal_flag"""
        self._spatial_pointer.deactivate_all()
        old_fatal_flag = self._fatal_flag.to_torch()
        self._fatal_flag.fill(0)
        return old_fatal_flag
    
    @ti.func
    def xyz2ijk_func(self, xyz: ti.math.vec3) -> ti.math.ivec3:
        bounds = self._bounds[None]
        xyz_size = self._xyz_size[None]

        return ti.math.clamp(
            ti.cast((xyz - bounds[0, :]) / 
                    (bounds[1, :] - bounds[0, :]) *
                    xyz_size, int),
            0, xyz_size - 1
        )

    @ti.func
    def add_bounding_box_func(self, batch_idx: int, bb: BoundingBox, value: int, v0: ti.math.vec3, v1: ti.math.vec3, v2: ti.math.vec3):
        lower_xyz = ti.min(bb.bounds[0, :], bb.bounds[1, :])
        upper_xyz = ti.max(bb.bounds[0, :], bb.bounds[1, :])
        lower_ijk = self.xyz2ijk_func(lower_xyz)
        upper_ijk = self.xyz2ijk_func(upper_xyz)
        cnt = calculate_loop_size_func(lower_ijk, upper_ijk)
        if cnt <= self._max_bb_occupy_num:
            for i, j, k in ti.ndrange(
                (lower_ijk[0], upper_ijk[0] + 1),
                (lower_ijk[1], upper_ijk[1] + 1),
                (lower_ijk[2], upper_ijk[2] + 1),
            ):
                self._spatial_cell[batch_idx, i, j, k].append(value)
        else:
            old_fatal_flag = ti.atomic_or(self._fatal_flag[batch_idx], True)
            if not old_fatal_flag:
                print(f"[ERROR] batch_idx {batch_idx} spatial loop size: {cnt} is too large. Vert:{v0} {v1} {v2}")

    @ti.func
    def get_cell_length_func(self, batch_idx: int, i: int, j: int, k: int) -> int:
        return self._spatial_cell[batch_idx, i, j, k].length()
    
    @ti.func
    def get_cell_item_func(self, batch_idx: int, i: int, j: int, k: int, l: int) -> int:
        return self._spatial_cell[batch_idx, i, j, k, l]

    def get_fatal_flag(self) -> torch.Tensor:
        return self._fatal_flag.to_torch(self._device)


@ti.func
def calculate_loop_size_func(ijk_lower: ti.math.ivec3, ijk_upper: ti.math.ivec3) -> int:
    return (ijk_upper[0] - ijk_lower[0] + 1) * (ijk_upper[1] - ijk_lower[1] + 1) * (ijk_upper[2] - ijk_lower[2] + 1)


@ti.data_oriented
class SparseMask:
    def __init__(self, n: int, m: int, block_size=None) -> None:
        if block_size is None:
            bs = 1
            while bs * 2 < min(n, m, 32):
                bs *= 2
        else: 
            bs = block_size
        assert bs < min(n, m), (bs, n, m)

        self.mask: ti.ScalarField = ti.field(int)
        self.pointer: ti.SNode = ti.root.pointer(
            ti.ij, (
                (n + bs - 1) // bs,
                (m + bs - 1) // bs,
            )
        )
        self.pointer.bitmasked(ti.ij, (bs, bs)).place(self.mask)
        taichi_utils.GLOBAL_CREATER.LogSparseField(
            shape=(
                (n + bs - 1) // bs * bs,
                (m + bs - 1) // bs * bs
            )
        )
    
    @ti.kernel
    def print_active(self):
        for i, j in self.mask:
            print('field x[{}, {}] = {}'.format(i, j, self.mask[i, j]))


@ti.func
def check_face_edge_pair_intersect_func(
    v0: ti.math.vec3, 
    v1: ti.math.vec3, 
    v2: ti.math.vec3, 
    v3: ti.math.vec3, 
    v4: ti.math.vec3, 
    eps: float
):
    ret = False
    mat = ti.Matrix.zero(ti.f64, 3, 3)
    mat[:, 0] = ti.cast(v3 - v4, ti.f64)
    mat[:, 1] = ti.cast(v1 - v0, ti.f64)
    mat[:, 2] = ti.cast(v2 - v0, ti.f64)
    xyz_scale = ti.abs(mat).sum() / 9
    mat_det = mat.determinant()
    if ti.abs(mat_det) > (xyz_scale ** 2) * eps:
        right = v3 - v0
        left = mat.inverse() @ ti.cast(right, ti.f64)

        a, b, c = 1. - left[1] - left[2], left[1], left[2]
        t = left[0]
        abct = ti.Vector([a, b, c, t], ti.f64)
        zero_f64 = ti.cast(0.0, ti.f64)
        one_f64 = ti.cast(1.0, ti.f64)
        if (zero_f64 < abct).all() and (abct < one_f64).all():
            ret = True
    return ret


@ti.kernel
def detect_intersect_kernel(
    sp: ti.template(),
    mask_face_edge: ti.template(),
    ans_face_edge: ti.template(),
    vert: ti.types.ndarray(dtype=ti.math.vec3),
    face: ti.types.ndarray(dtype=ti.math.ivec3),
    edge: ti.types.ndarray(dtype=ti.math.ivec2),
    vert_color: ti.types.ndarray(),
    eps: float,
) -> bool:
    batch_idx = 0
    ans = False

    # add face
    for fid in range(face.shape[0]):
        v0id = face[fid][0]
        v1id = face[fid][1]
        v2id = face[fid][2]
        v0 = vert[v0id]
        v1 = vert[v1id]
        v2 = vert[v2id]
        sp.add_bounding_box_func(batch_idx, triangle_bounding_box_func(v0, v1, v2), fid, v0, v1, v2)
    
    # query edge
    for eid in range(edge.shape[0]):
        v3id = edge[eid][0]
        v4id = edge[eid][1]
        v3 = vert[v3id]
        v4 = vert[v4id]
        bb = line_bounding_box_func(v3, v4)
        ijk_lower = sp.xyz2ijk_func(bb.bounds[0, :])
        ijk_upper = sp.xyz2ijk_func(bb.bounds[1, :])
        cnt = calculate_loop_size_func(ijk_lower, ijk_upper)
        if cnt <= sp._max_bb_occupy_num:
            ti.loop_config(serialize=True)
            for i, j, k in ti.ndrange(
                (ijk_lower[0], ijk_upper[0] + 1),
                (ijk_lower[1], ijk_upper[1] + 1),
                (ijk_lower[2], ijk_upper[2] + 1),):
                ti.loop_config(serialize=True)
                for l in range(sp.get_cell_length_func(batch_idx, i, j, k)):
                    fid = sp.get_cell_item_func(batch_idx, i, j, k, l)
                    if not ti.atomic_or(mask_face_edge[fid, eid], 1):
                        v0id = face[fid][0]
                        v1id = face[fid][1]
                        v2id = face[fid][2]
                        v0 = vert[v0id]
                        v1 = vert[v1id]
                        v2 = vert[v2id]
                        edge_on_face = (
                            (v3id == v0id or v3id == v1id or v3id == v2id) or
                            (v4id == v0id or v4id == v1id or v4id == v2id)
                        )
                        if not edge_on_face:
                            if check_face_edge_pair_intersect_func(v0, v1, v2, v3, v4, eps):
                                ans = True
                                ans_face_edge[fid, eid] = 1
                                vert_color[v0id, 0] = 1.
                                vert_color[v1id, 0] = 1.
                                vert_color[v2id, 0] = 1.
                                vert_color[v3id, 1] = 1.
                                vert_color[v4id, 1] = 1.
                                
    return ans