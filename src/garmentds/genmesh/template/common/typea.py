from typing import Callable, Literal, get_args, Optional, Any
from dataclasses import asdict, dataclass, field
import os
import json
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw
import trimesh
from shapely.geometry import Polygon

from ...base_cls import GarmentCfgABC, GarmentTemplateABC, GarmentBoundaryEdge, GarmentPart, VertInfo
from ...tri import (
    generate_vert_on_boundary_edge, generate_vert_within_boundary, 
    delaunay_in_boundary, post_process_face_2d, vert_2d_to_3d, vert_2d_to_3d_new, 
    sanity_check, remove_repeat_vertex_and_build_vertex_map
)
from ...make_hood import HoodOptimizer, HoodOptimizerCfg, HoodSmoother

from garmentds.common.misc import default_tex_img_path
import garmentds.common.utils as utils


@dataclass
class GarmentCfgTypeA(GarmentCfgABC):
    edge_max_z: int
    edge_width: int
    boundary_dx: int
    boundary_dense_n: int
    interior_num: dict[str, int]
    hood_opt_cfg: HoodOptimizerCfg = field(default_factory=lambda: HoodOptimizerCfg())


class GarmentTypeA(GarmentTemplateABC):
    _meta = dict(name = "none")
    all_part_name_type = str
    all_part: dict[all_part_name_type, GarmentPart] = {}
    all_edge: dict[all_part_name_type, list[GarmentBoundaryEdge]] = dict()
    reuse_edge_pair_dict: dict[
        tuple[all_part_name_type, all_part_name_type], 
        list[tuple[GarmentBoundaryEdge, GarmentBoundaryEdge]]
    ] = dict()

    def _check_args(self):
        all_part_name_type_as_list = get_args(self.all_part_name_type)
        all_part_name_list = list(self.all_part.keys())
        is_hood_part = []
        for part_name in all_part_name_list:
            assert all_part_name_list.count(part_name) == 1, f"[ERROR] duplicated '{part_name}' in {all_part_name_list}"
            assert all_part_name_type_as_list.count(part_name) == 1, f"[ERROR] '{part_name}' typing error {all_part_name_type_as_list}"
            if self.all_part[part_name].is_hood: is_hood_part.append(self.all_part[part_name])
        
        assert len(is_hood_part) <= 1, "At most 1 hood part"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._check_args()
        self._cfg: GarmentCfgTypeA
        self._mesh_cache: Optional[tuple[trimesh.Trimesh, dict[Literal["success", "keypoint_idx", "boundary_idx", "vert_info"], Any]]]

    def _draw(self, *args, **kwargs): ...

    def _calculate_uv(self, *args, **kwargs): ...

    def _get_mask(
        self, 
        width: int, height: int, 
        xy2ij: Callable[[float, float], tuple[int, int]],
        cfg: GarmentCfgTypeA,
        part_name: all_part_name_type, 
    ):
        def swap(a, b): return b, a

        polygon_xy: list[tuple[int, int]] = []
        for edge in self.all_edge[part_name]:
            polygon_xy.append(swap(*xy2ij(*getattr(cfg, edge.start))))
            if edge.reverse:
                control_list = list(reversed(getattr(cfg, edge.control)))
            else:
                control_list = getattr(cfg, edge.control)
            for control_xy in control_list:
                polygon_xy.append(swap(*xy2ij(*control_xy)))

        img = Image.new(mode='L', size=(width, height), color=0)
        draw = ImageDraw.Draw(img)
        draw.polygon(xy=polygon_xy, fill=1, width=0)
        return np.array(img)

    def _add_annotation_and_draw_mesh(
        self, 
        img_np: np.ndarray, 
        xy2ij: Callable[[float, float], tuple[int, int]],
    ):
        # add annotation
        img = Image.fromarray(img_np)
        draw = ImageDraw.Draw(img)

        font_size, i_offset = 10, 10
        for k, v in self.asdict_keypoints().items():
            i, j = xy2ij(*v)
            j0, i0, j1, i1 = draw.textbbox(xy=(j, i), text=k, font_size=font_size)
            dj, di = j1 - j0, i1 - i0
            draw.text((int(j - dj / 2), int(i - di / 2 + i_offset)), text=k, font_size=font_size, fill="white")
            draw.circle((j, i), radius=5)
        
        # draw mesh
        if self._mesh_cache is not None:
            mesh = self._mesh_cache[0]
            for edge in mesh.edges:
                i0, j0 = xy2ij(*mesh.vertices[edge[0], :2])
                i1, j1 = xy2ij(*mesh.vertices[edge[1], :2])
                draw.line([(j0, i0), (j1, i1)])
        
        return img

    def draw(self, width: int, height: int, xy2ij: Callable[[float, float], tuple[int, int]]):
        return self._draw(width, height, xy2ij, self._cfg)
    
    def _sample_vert_boundary(
        self, cfg: GarmentCfgTypeA, dx: float, dense_n: int
    ):
        """
        args
        - dx: float, control the distance between two samples
        - dense_n: int, control the accuracy when sampling on edges

        return
        - vert_boundary: str -> [B, 2], float
        - vert_connect: str -> [B, ], int
        - vert_vertmap: str -> [B, ], int, if vertmap is the same integer, two vertices should at the same 2D / 3D location (reuse issue)
        - affine_dict: str -> str -> [B, ], float, affine parameter for each vertex
        - success: bool
        """
        success = True

        def call_generate_vert_on_boundary_edge(edge: GarmentBoundaryEdge):
            vert, affine = generate_vert_on_boundary_edge(
                getattr(cfg, edge.start), getattr(cfg, edge.control), 
                getattr(cfg, edge.end), edge.reverse, edge.include, dx, dense_n,
            )
            if edge.connect:
                return vert, np.ones((vert.shape[0], ), dtype=int), affine
            else:
                return vert, np.zeros((vert.shape[0], ), dtype=int), affine
        def find_reuse_edge(boundary_dict: dict[str, dict[GarmentBoundaryEdge, np.ndarray]], target_edge: GarmentBoundaryEdge, target_part_name: str):
            for part_name, boundary in boundary_dict.items():
                for edge in boundary.keys():
                    '''if edge == target_edge:
                        return part_name, edge, slice(None, None, None)'''
                    if (part_name, target_part_name) in self.reuse_edge_pair_dict.keys():
                        for e1, e2 in self.reuse_edge_pair_dict[(part_name, target_part_name)]:
                            if e1 == edge and e2 == target_edge:
                                if e1 == e2: s = slice(None, None, None)
                                else: s = slice(None, None, -1)
                                return part_name, edge, s
            return None
        
        # get all reuse pair
        wait_for_reuse = set()
        for (reuse_part_name, part_name), reuse_list in self.reuse_edge_pair_dict.items():
            for reuse_edge, edge in reuse_list: wait_for_reuse.add((part_name, reuse_part_name, edge, reuse_edge))
        
        # generate on boundary edge
        boundary_dict: dict[self.all_part_name_type, dict[GarmentBoundaryEdge, np.ndarray]] = {part_name: {} for part_name in self.all_part.keys()}
        connect_dict: dict[self.all_part_name_type, dict[GarmentBoundaryEdge, np.ndarray]] = {part_name: {} for part_name in self.all_part.keys()}
        vertmap_dict: dict[self.all_part_name_type, dict[GarmentBoundaryEdge, np.ndarray]] = {part_name: {} for part_name in self.all_part.keys()}
        affine_dict: dict[self.all_part_name_type, dict[GarmentBoundaryEdge, np.ndarray]] = {part_name: {} for part_name in self.all_part.keys()}
        curr_different_vert_num = 0
        for part_name in self.all_part.keys():
            boundary, connect, vertmap, affine = boundary_dict[part_name], connect_dict[part_name], vertmap_dict[part_name], affine_dict[part_name]
            for edge in self.all_edge[part_name]:
                find_result = find_reuse_edge(boundary_dict, edge, part_name)
                if find_result is not None:
                    reuse_part_name, reuse_edge, edge_slice = find_result
                    boundary[edge], connect[edge], vertmap[edge], affine[edge] = (
                        boundary_dict[reuse_part_name][reuse_edge][edge_slice].copy(), 
                        connect_dict[reuse_part_name][reuse_edge][edge_slice].copy(),
                        vertmap_dict[reuse_part_name][reuse_edge][edge_slice].copy(),
                        affine_dict[reuse_part_name][reuse_edge][edge_slice].copy(),
                    ) # Reuse edge and reverse
                    print(f"[INFO] find reuse edge {part_name} {edge} {reuse_part_name} {reuse_edge}")
                    wait_for_reuse.remove((part_name, reuse_part_name, edge, reuse_edge))
                else:
                    boundary[edge], connect[edge], affine[edge] = call_generate_vert_on_boundary_edge(edge)
                    nv = boundary[edge].shape[0]
                    vertmap[edge] = np.arange(curr_different_vert_num, curr_different_vert_num + nv)
                    curr_different_vert_num += nv
        assert len(wait_for_reuse) == 0, f"not all reuse_edge_pair_dict is used: {list(wait_for_reuse)}"
        
        # merge result
        vert_boundary = {
            part_name: np.concatenate([v for v in boundary_dict[part_name].values()], axis=0)
            for part_name in self.all_part.keys()
        }
        vert_connect = {
            part_name: np.concatenate([v for v in connect_dict[part_name].values()], axis=0)
            for part_name in self.all_part.keys()
        }
        vert_vertmap = {
            part_name: np.concatenate([v for v in vertmap_dict[part_name].values()], axis=0)
            for part_name in self.all_part.keys()
        }
        for part_name, boundary_points in vert_boundary.items():
            if not Polygon(boundary_points).is_simple:
                success = False
                print(f"[ERROR] boundary {part_name} is not simple ... ")

        print(f"[INFO] boundary " + " ".join([f"{part_name} {vert_boundary[part_name].shape}" for part_name in self.all_part.keys()]))
        return vert_boundary, vert_connect, vert_vertmap, affine_dict, success

    def _delaunay_and_assign_z(
        self, 
        vert_boundary: dict[all_part_name_type, np.ndarray], 
        vert_interior: dict[all_part_name_type, np.ndarray],
        vert_connect: dict[all_part_name_type, np.ndarray],
        vert_vertmap: dict[all_part_name_type, np.ndarray],
        affine_dict: dict[all_part_name_type, dict[GarmentBoundaryEdge, np.ndarray]],
        max_z: float, width: float
    ):
        """
        args
        - vert_boundary: str -> [B, 2], float
        - vert_interior: str -> [I, 2], float
        - vert_connect: str -> [B, ], int
        - vert_vertmap: str -> [B, ], int
        - affine_dict: str -> edge -> [B, ], float
        """
        success = True
        keypoint_idx = {}
        boundary_idx = {}

        vert_all: list[np.ndarray] = []
        face_all: list[np.ndarray] = []
        uv_all: list[np.ndarray] = []
        vert_info: list[VertInfo] = []

        vertmap_to_3d: dict[int, np.ndarray] = {}
        xy_for_uv_dict: dict[str, np.ndarray] = {}
        curr_total_vert_num = 0
        for part_id, (part_name, part) in enumerate(self.all_part.items()):
            # make 2D vertices
            is_collar, is_back, is_hood = part.is_collar, part.is_back, part.is_hood
            vert_b, vert_i, vert_c, vert_m = vert_boundary[part_name], vert_interior[part_name], vert_connect[part_name], vert_vertmap[part_name]
            vert_i, face = delaunay_in_boundary(vert_b, vert_i, True)
            vert = np.concatenate([vert_b, vert_i], axis=0) # always first vert_b then vert_i
            xy_for_uv_dict[part_name] = vert.copy()

            # 2D vertices to 3D vertices
            power = int(not is_back) + int(is_collar)
            face = post_process_face_2d(vert, face, power == 1)
            vert_3d = vert_2d_to_3d_new(vert, vert_boundary.values(), vert_connect.values(), max_z, width) # use all boundary verts to compute z
            if is_collar:
                # if is_collar, compute the minimum distance to 'is_reuse_vert' (instead of 'vert_c')
                # based on this distance, we compute the z of the collar
                is_reuse_vert = np.array([(vm in vertmap_to_3d.keys()) for vm in vert_m])
                vert_3d_collar = vert_2d_to_3d(vert, [vert_b], [is_reuse_vert], max_z * 0.5, width)
                vert_3d = np.concatenate([vert_3d[:, :2], (vert_3d[:, [2]] + vert_3d_collar[:, [2]])], axis=1)
            if is_back: 
                z_sign = -1
            else: 
                z_sign = +1
            vert_3d[:, 2] *= z_sign
            vert_3d[np.where(vert_c)[0], 2] = 0. # force connected vertex's z to 0
            
            # force reused vertices at same 3D location
            for idx, vm in enumerate(vert_m):
                if vm not in vertmap_to_3d.keys():
                    vertmap_to_3d[vm] = vert_3d[idx]
                    vert_info.append(VertInfo(part, part_name, True, False))
                else:
                    vert_3d[idx] = vertmap_to_3d[vm]
                    vert_info.append(VertInfo(part, part_name, True, True))

            vert_all.append(vert_3d)
            face_all.append(face)
            for _ in range(vert_i.shape[0]):
                vert_info.append(VertInfo(part, part_name, False, False))

            # save keypoint idx and boundary idx
            boundary_idx[part_name] = np.arange(curr_total_vert_num, curr_total_vert_num + len(vert_b))
            for edge, aff in affine_dict[part_name].items():
                if edge.include in ["start", "both"]:
                    if edge.start not in keypoint_idx.keys():
                        keypoint_idx[edge.start] = [curr_total_vert_num]
                    else:
                        keypoint_idx[edge.start].append(curr_total_vert_num)
                if edge.include in ["end", "both"]:
                    if edge.end not in keypoint_idx.keys():
                        keypoint_idx[edge.end] = [curr_total_vert_num + len(aff) - 1]
                    else:
                        keypoint_idx[edge.end].append(curr_total_vert_num + len(aff) - 1)
                curr_total_vert_num += len(aff)
            curr_total_vert_num += len(vert_i)
        
        for xyz, face, (part_name, part) in zip(vert_all, face_all, self.all_part.items()):
            uv_, success_calculate_uv, info = self._calculate_uv(
                part_name=part_name, face=face, 
                xy_dict=xy_for_uv_dict, affine_dict=affine_dict
            )
            assert uv_.shape[0] == xyz.shape[0], f"{uv_.shape}, {xyz.shape}"
            uv_all.append(uv_)
            success = success and success_calculate_uv
        
        vert_merge, face_merge, uv_merge = None, None, None
        for vert, face, uv in zip(vert_all, face_all, uv_all):
            if vert_merge is None:
                vert_merge, face_merge, uv_merge = vert, face, uv
            else:
                face_merge = np.concatenate([face_merge, face + vert_merge.shape[0]], axis=0)
                vert_merge = np.concatenate([vert_merge, vert], axis=0)
                uv_merge = np.concatenate([uv_merge, uv], axis=0)
        
        mesh = trimesh.Trimesh(vertices=vert_merge, faces=face_merge, process=False)
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv_merge, image=Image.open(default_tex_img_path))
        return mesh, keypoint_idx, boundary_idx, vert_info, success
    
    def _refine_hood_inplace(self, mesh: trimesh.Trimesh, vert_info_list: list[VertInfo]):
        success = True

        hood_vert_type = []
        hood_vert_xyz = []
        hood_edge = set()
        vert_old_to_new = dict()
        vert_new_to_old = []
        need_to_refine_hood = False
        for vert_idx, vert_info in enumerate(vert_info_list):
            if vert_info.part.is_hood: # assume at most 1 hood part
                need_to_refine_hood = True
                vert_old_to_new[vert_idx] = len(hood_vert_xyz)
                vert_new_to_old.append(vert_idx)
                hood_vert_xyz.append(mesh.vertices[vert_idx])
                hood_vert_type.append(2 if vert_info.is_reuse else 1 if vert_info.is_boundary else 0)
        if not need_to_refine_hood:
            return success
        
        for v1id, v2id in mesh.edges:
            if v1id in vert_old_to_new.keys() and v2id in vert_old_to_new.keys():
                x, y = vert_old_to_new[v1id], vert_old_to_new[v2id]
                if x > y: 
                    x, y = y, x
                if (x, y) not in hood_edge: 
                    hood_edge.add((x, y))
        
        hood_face = []
        for v1id, v2id, v3id in mesh.faces:
            if v1id in vert_old_to_new.keys() and v2id in vert_old_to_new.keys() and v3id in vert_old_to_new.keys():
                x, y, z = vert_old_to_new[v1id], vert_old_to_new[v2id], vert_old_to_new[v3id]
                hood_face.append([x, y, z])
        
        hood_vert_type_np = np.array(hood_vert_type)
        hood_vert_xyz_np = np.array(hood_vert_xyz)
        hood_edge_np = np.array(list(hood_edge))
        hood_face_np = np.array(hood_face)

        xyz_offset = np.mean(hood_vert_xyz_np, axis=0)
        xyz_offset[2] = 0.
        hood_vert_xyz_np -= xyz_offset

        opt = HoodOptimizer(hood_vert_type_np, hood_vert_xyz_np, hood_edge_np, hood_face_np)
        optimized_xyz = opt.optimize(self._cfg.hood_opt_cfg)
        for vid, xyz in enumerate(optimized_xyz):
            if hood_vert_type_np[vid] == 0:
                mesh.vertices[vert_new_to_old[vid]] = xyz + xyz_offset
        
        mesh_processed, vert_w_to_wo = remove_repeat_vertex_and_build_vertex_map(mesh)
        set1 = set() # vert_idx_is_boundary_and_is_hood
        set2 = set() # vert_idx_is_boundary_and_not_hood
        for vert_idx, vert_info in enumerate(vert_info_list):
            if vert_info.is_boundary:
                if vert_info.part.is_hood:
                    set1.add(vert_w_to_wo[vert_idx])
                else:
                    set2.add(vert_w_to_wo[vert_idx])
        vertex_connect_hood = list(set1.intersection(set2))

        edge_count = defaultdict(int)
        vertex_exclude = set()
        for e1, e2 in mesh_processed.edges:
            if e2 > e1: 
                e1, e2 = e2, e1
            edge_count[(e1, e2)] += 1
        for (e1, e2), cnt in edge_count.items():
            if cnt == 1: 
                vertex_exclude.add(e1), vertex_exclude.add(e2) # do not smooth boundary vertex
        
        smoother = HoodSmoother(mesh_processed, vertex_connect_hood, list(vertex_exclude.union(set2.difference(set1))))
        mesh.vertices = smoother.smooth()[vert_w_to_wo]
            
        return success

    def _triangulation(self, cfg: GarmentCfgTypeA, skip_check_self_intersection=False):
        success = True
        if self._mesh_cache is not None:
            return self._mesh_cache
        
        vert_boundary, vert_connect, vert_vertmap, affine_dict, success_sample_vert_boundary = self._sample_vert_boundary(self._cfg, cfg.boundary_dx, cfg.boundary_dense_n)
        success = success and success_sample_vert_boundary

        vert_interior = {k: generate_vert_within_boundary(vert_boundary[k], cfg.interior_num[k]) for k in vert_boundary.keys()}
        (
            mesh, keypoint_idx, boundary_idx, vert_info, success_delaunay_and_assign_z
        ) = self._delaunay_and_assign_z(
            vert_boundary, vert_interior, vert_connect, vert_vertmap, affine_dict, 
            cfg.edge_max_z, cfg.edge_width
        )
        success = success and success_delaunay_and_assign_z
        print(f"[INFO] generated mesh:{mesh}")

        success_refine_hood_inplace = self._refine_hood_inplace(mesh, vert_info)
        success = success and success_refine_hood_inplace
        print(f"[INFO] successfully refine hood inplace ? {success_refine_hood_inplace}")

        vertex_normals = mesh.vertex_normals # auto normals calculation
        success = success and sanity_check(mesh, skip_check_self_intersection=skip_check_self_intersection)
        print(f"[INFO] sanity_check completed ...")
        self._mesh_cache = (
            mesh, dict(
                success=success, 
                keypoint_idx=keypoint_idx,
                boundary_idx=boundary_idx,
                vert_info=[vi.part_name for vi in vert_info]
            )
        )

        if success:
            print(f"[INFO] generate mesh success ...")
        else:
            print(f"[WARN] generate mesh unsuccess ...")

        return self._mesh_cache
    
    def triangulation(self, **kwargs):
        return self._triangulation(self._cfg, **kwargs)
    
    def get_info_to_export(self):
        if self._mesh_cache is None:
            print(f"[WARN] not triangularized yet ...")
            return dict()
        
        return dict(
            cfg = asdict(self._cfg),
            triangulation = self._mesh_cache[1],
            meta = self._meta,
        )

    def quick_export(self, path):
        if self._mesh_cache is not None:
            mesh = self._mesh_cache[0]
            mesh.export(path)

            keypoint_idx = []
            for k in self._mesh_cache[1]["keypoint_idx"].values():
                for vid in k: 
                    keypoint_idx.append(vid)
            vertex_color = np.ones_like(mesh.vertices)
            vertex_color[keypoint_idx] = [1., 0., 0.]
            trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_color, process=False).export(path + ".key.obj")

            def numpy_serializer(obj):
                if isinstance(obj, np.ndarray): return obj.tolist()
                raise TypeError(f"Type {type(obj)} not serializable")
            with open(os.path.join(os.path.dirname(path), "mesh_info.json"), "w") as f_obj:
                json.dump(self.get_info_to_export(), f_obj, indent=4, default=numpy_serializer)
            print("[INFO] mesh saved !")
        else:
            print("[WARN] not triangularized yet")