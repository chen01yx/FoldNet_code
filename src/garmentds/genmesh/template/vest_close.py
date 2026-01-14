from dataclasses import dataclass, field
from typing import Callable, Literal

from PIL import Image, ImageDraw
import numpy as np

from .common.typea import GarmentCfgTypeA, GarmentTypeA
from ..base_cls import GarmentBoundaryEdge, GarmentPart, Point2
from ..make_collar import CollarOptimizer

import garmentds.common.utils as utils


@dataclass
class VestCloseCfg(GarmentCfgTypeA):
    """
    - sh = shoulder
    - cl = collar
    - nf = neck_f
    - nb = neck_b
    - ar = armpit
    - cn = corner
    - bt = spine_bottom
    """

    ### 2d geometry keypoints
    l_collar: Point2 = Point2(-0.12, +0.3)
    r_collar: Point2 = Point2(+0.12, +0.3)
    neck_f: Point2 = Point2(0.0, +0.10)
    neck_b: Point2 = Point2(0.0, +0.27)

    l_shoulder: Point2 = Point2(-0.20, +0.27)
    r_shoulder: Point2 = Point2(+0.20, +0.27)
    l_armpit: Point2 = Point2(-0.25, +0.1)
    r_armpit: Point2 = Point2(+0.25, +0.1)
    l_corner: Point2 = Point2(-0.25, -0.3)
    r_corner: Point2 = Point2(+0.25, -0.3)
    spine_bottom_f: Point2 = Point2(0.0, -0.3)
    spine_bottom_b: Point2 = Point2(0.0, -0.3)

    l_nf_cl: list[Point2] = field(default_factory=lambda:[(-0.06, +0.19)])
    r_nf_cl: list[Point2] = field(default_factory=lambda:[(+0.06, +0.19)])
    l_nb_cl: list[Point2] = field(default_factory=lambda:[(-0.06, +0.28)])
    r_nb_cl: list[Point2] = field(default_factory=lambda:[(+0.06, +0.28)])
    l_cl_sh: list[Point2] = field(default_factory=lambda:[])
    r_cl_sh: list[Point2] = field(default_factory=lambda:[])

    l_sh_ar: list[Point2] = field(default_factory=lambda:[(-0.225, +0.15)])
    r_sh_ar: list[Point2] = field(default_factory=lambda:[(+0.225, +0.15)])
    l_ar_cn: list[Point2] = field(default_factory=lambda:[])
    r_ar_cn: list[Point2] = field(default_factory=lambda:[])
    l_cn_bt: list[Point2] = field(default_factory=lambda:[])
    r_cn_bt: list[Point2] = field(default_factory=lambda:[])

    l_collar_o: Point2 = Point2(-0.11, +0.3)
    r_collar_o: Point2 = Point2(+0.11, +0.3)
    neck_f_o: Point2 = Point2(0.0, +0.11)
    neck_b_o: Point2 = Point2(0.0, +0.28)

    l_nf_cl_o: list[Point2] = field(default_factory=lambda:[(-0.053, +0.197)])
    r_nf_cl_o: list[Point2] = field(default_factory=lambda:[(+0.053, +0.197)])
    l_nb_cl_o: list[Point2] = field(default_factory=lambda:[(-0.053, +0.287)])
    r_nb_cl_o: list[Point2] = field(default_factory=lambda:[(+0.053, +0.287)])
    l_cl_cl_o: list[Point2] = field(default_factory=lambda:[])
    r_cl_cl_o: list[Point2] = field(default_factory=lambda:[])

    ### 3d generation args
    edge_max_z: float = 0.05
    edge_width: float = 0.09
    boundary_dx: float = 0.015
    boundary_dense_n: int = 10000
    interior_num: dict = field(default_factory=lambda: dict(
        front = 2000, back = 2000, 
        collarf = 20, collarb = 20
    ))

    def symmetry(self):
        def mirror(x, y): return (-x, y)
        for r_keypoint in [
            "r_shoulder", "r_collar", "r_armpit",
            "r_corner", "r_collar_o",
        ]:
            setattr(self, r_keypoint, mirror(*getattr(self, "l_" + r_keypoint[2:])))
        
        for keypoint in ["spine_bottom_f", "spine_bottom_b", "neck_f", "neck_b", "neck_f_o", "neck_b_o"]:
            setattr(self, keypoint, (0., getattr(self, keypoint)[1]))
        
        for r_keypoint in [
            "r_nf_cl", "r_nb_cl", "r_cl_sh", 
            "r_sh_ar", "r_ar_cn", "r_cn_bt", 
            "r_nf_cl_o", "r_nb_cl_o",
        ]:
            l_keypoint = "l_" + r_keypoint[2:]
            new_r_val = []
            for val in getattr(self, l_keypoint):
                new_r_val.append(mirror(*val))
            setattr(self, r_keypoint, new_r_val)
    
    def sanity_check(self):
        pass


class VestClose(GarmentTypeA):
    _meta = dict(name = "vest_close")
    all_part_name_type = Literal["front", "back", "collarf", "collarb"]
    all_part = dict(
        front = GarmentPart(False, False, False),
        back = GarmentPart(True, False, False),
        collarf = GarmentPart(False, False, False),
        collarb = GarmentPart(True, False, False),
    )
    all_edge: dict[all_part_name_type, list[GarmentBoundaryEdge]] = dict(
        front = [
            GarmentBoundaryEdge("l_shoulder", "l_cl_sh", "l_collar", True, connect=True, include="both"),
            GarmentBoundaryEdge("l_collar", "l_nf_cl", "neck_f", True, connect=False, include="none"),
            GarmentBoundaryEdge("neck_f", "r_nf_cl", "r_collar", False, connect=False),
            GarmentBoundaryEdge("r_collar", "r_cl_sh", "r_shoulder", False, connect=True, include="both"),
            GarmentBoundaryEdge("r_shoulder", "r_sh_ar", "r_armpit", False, connect=False, include="none"),
            GarmentBoundaryEdge("r_armpit", "r_ar_cn", "r_corner", False, connect=True, include="both"),
            GarmentBoundaryEdge("r_corner", "r_cn_bt", "spine_bottom_f", False, connect=False, include="none"),
            GarmentBoundaryEdge("spine_bottom_f", "l_cn_bt", "l_corner", True, connect=False),
            GarmentBoundaryEdge("l_corner", "l_ar_cn", "l_armpit", True, connect=True, include="both"),
            GarmentBoundaryEdge("l_armpit", "l_sh_ar", "l_shoulder", True, connect=False, include="none"),
        ],
        back = [
            GarmentBoundaryEdge("l_shoulder", "l_cl_sh", "l_collar", True, connect=True, include="both"),
            GarmentBoundaryEdge("l_collar", "l_nb_cl", "neck_b", True, connect=False, include="none"),
            GarmentBoundaryEdge("neck_b", "r_nb_cl", "r_collar", False, connect=False),
            GarmentBoundaryEdge("r_collar", "r_cl_sh", "r_shoulder", False, connect=True, include="both"),
            GarmentBoundaryEdge("r_shoulder", "r_sh_ar", "r_armpit", False, connect=False, include="none"),
            GarmentBoundaryEdge("r_armpit", "r_ar_cn", "r_corner", False, connect=True, include="both"),
            GarmentBoundaryEdge("r_corner", "r_cn_bt", "spine_bottom_b", False, connect=False, include="none"),
            GarmentBoundaryEdge("spine_bottom_b", "l_cn_bt", "l_corner", True, connect=False),
            GarmentBoundaryEdge("l_corner", "l_ar_cn", "l_armpit", True, connect=True, include="both"),
            GarmentBoundaryEdge("l_armpit", "l_sh_ar", "l_shoulder", True, connect=False, include="none"),
        ],
        collarf = [
            GarmentBoundaryEdge("l_collar", "l_cl_cl_o", "l_collar_o", True, connect=True, include="both"),
            GarmentBoundaryEdge("l_collar_o", "l_nf_cl_o", "neck_f_o", True, connect=False, include="none"),
            GarmentBoundaryEdge("neck_f_o", "r_nf_cl_o", "r_collar_o", False, connect=False),
            GarmentBoundaryEdge("r_collar_o", "r_cl_cl_o", "r_collar", False, connect=True, include="both"),
            GarmentBoundaryEdge("r_collar", "r_nf_cl", "neck_f", True, connect=False, include="end"),
            GarmentBoundaryEdge("neck_f", "l_nf_cl", "l_collar", False, connect=False, include="none"),
        ],
        collarb = [
            GarmentBoundaryEdge("l_collar", "l_cl_cl_o", "l_collar_o", True, connect=True, include="both"),
            GarmentBoundaryEdge("l_collar_o", "l_nb_cl_o", "neck_b_o", True, connect=False, include="none"),
            GarmentBoundaryEdge("neck_b_o", "r_nb_cl_o", "r_collar_o", False, connect=False),
            GarmentBoundaryEdge("r_collar_o", "r_cl_cl_o", "r_collar", False, connect=True, include="both"),
            GarmentBoundaryEdge("r_collar", "r_nb_cl", "neck_b", True, connect=False, include="end"),
            GarmentBoundaryEdge("neck_b", "l_nb_cl", "l_collar", False, connect=False, include="none"),
        ],
    ) # for triangulation, gaurantee clockwise
    reuse_edge_pair_dict = {
        ("front", "collarf") : [
            (GarmentBoundaryEdge("l_collar", "l_nf_cl", "neck_f", True, connect=False, include="none"),
             GarmentBoundaryEdge("neck_f", "l_nf_cl", "l_collar", False, connect=False, include="none"),), 
            (GarmentBoundaryEdge("neck_f", "r_nf_cl", "r_collar", False, connect=False),
             GarmentBoundaryEdge("r_collar", "r_nf_cl", "neck_f", True, connect=False, include="end"),),
        ],
        ("back", "collarb") : [
            (GarmentBoundaryEdge("l_collar", "l_nb_cl", "neck_b", True, connect=False, include="none"),
             GarmentBoundaryEdge("neck_b", "l_nb_cl", "l_collar", False, connect=False, include="none"),),
            (GarmentBoundaryEdge("neck_b", "r_nb_cl", "r_collar", False, connect=False),
             GarmentBoundaryEdge("r_collar", "r_nb_cl", "neck_b", True, connect=False, include="end"),),
        ],
    } # identify which edge should be re-used, format: (p, p'): [(e1, e1'), (e2, e2'), ...] means `e'` in `p'` should be the same as `e` in `p`

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(cfg=utils.dataclass_init_from_dict(VestCloseCfg, kwargs))
        self._cfg: VestCloseCfg
    
    def _draw(self, width: int, height: int, xy2ij: Callable[[float, float], tuple[int, int]], cfg: VestCloseCfg):
        img_np = np.zeros((height, width, 4), dtype=np.uint8)

        mask_front = self._get_mask(width, height, xy2ij, cfg, "front")
        mask_back = self._get_mask(width, height, xy2ij, cfg, "back")
        mask_collarf = self._get_mask(width, height, xy2ij, cfg, "collarf")
        mask_collarb = self._get_mask(width, height, xy2ij, cfg, "collarb")

        img_np[np.where(mask_back)] = (0, 127, 0, 255)
        img_np[np.where(mask_front)] = (0, 0, 255, 255)
        img_np[np.where(mask_collarf)] = (127, 127, 0, 255)
        img_np[np.where(mask_collarb)] = (127, 191, 0, 255)

        return self._add_annotation_and_draw_mesh(img_np, xy2ij)
    
    def _calculate_uv(
        self, 
        part_name: all_part_name_type, 
        face: np.ndarray, 
        xy_dict: dict[all_part_name_type, np.ndarray], 
        affine_dict: dict[all_part_name_type, dict[GarmentBoundaryEdge, np.ndarray]]
    ):
        success = True
        info = {}

        xy_all = np.concatenate(list(xy_dict.values()), axis=0)
        scale = np.max(xy_all.max(axis=0) - xy_all.min(axis=0))
        x, y, s = xy_dict[part_name][:, 0], xy_dict[part_name][:, 1], scale * 2.2 # leave for some space
        x, y = (x - x.min())[:, None] / s, (y - y.min())[:, None] / s

        info_str = f"{part_name}"
        if part_name == "front":
            uv = np.concatenate([x, y + 0.5], axis=1)
            success = success and self.check_bound(uv, [0.0, 0.5, 0.5, 1.0], info_str)
        elif part_name == "back":
            uv = np.concatenate([x, y], axis=1)
            success = success and self.check_bound(uv, [0.0, 0.5, 0.0, 0.5], info_str)
        elif part_name in ["collarf", "collarb"]:
            val_list = list(affine_dict[part_name].values())
            l0, l1, l2, l3, l4, l5 = [len(v) for v in val_list]
            def set_val(arr: np.ndarray, start_idx: int, total_len: int, val): arr[start_idx: start_idx + total_len] = val
            u, v = np.zeros(len(x)), np.zeros(len(x))

            set_val(u, l0, l1, np.arange(1, l1 + 1, 1) / (l1 + l2 + 1))
            set_val(u, l0 + l1, l2, np.arange(l1 + 1, l1 + 1 + l2, 1) / (l1 + l2 + 1))
            set_val(u, l0 + l1 + l2, l3, 1.)
            set_val(u, l0 + l1 + l2 + l3, l4, np.arange(l4 + l5, l5, -1) / (l4 + l5 + 1))
            set_val(u, l0 + l1 + l2 + l3 + l4, l5, np.arange(l5, 0, -1) / (l4 + l5 + 1))

            set_val(v, 0, l0, np.linspace(0., 1., l0))
            set_val(v, l0, l1 + l2, 1.)
            set_val(v, l0 + l1 + l2, l3, np.linspace(1., 0., l3))

            collarf_length = sum([len(x) for x in affine_dict["collarf"].values()]) - 1
            collarb_length = sum([len(x) for x in affine_dict["collarb"].values()]) - 1
            thispart_length = sum([len(x) for x in affine_dict[part_name].values()]) - 1
            scale_u = 0.5 * thispart_length / max(collarf_length, collarb_length)

            u, v = u[:, None], v[:, None]
            if part_name == "collarf":
                uv = np.concatenate([u * scale_u + 0.5, v * 0.05 + 0.75], axis=1)
                o = np.ones(len(x), dtype=int)
                o[:l0 + l1 + l2 + l3 + l4 + l5] = 0.
                uv = CollarOptimizer(np.concatenate([x, y], axis=1), uv, o, face).optimize()
                success = success and self.check_bound(uv, [0.50, 1.00, 0.75, 1.00], info_str)
            elif part_name == "collarb":
                uv = np.concatenate([u * scale_u + 0.5, v * 0.05 + 0.50], axis=1)
                o = np.ones(len(x), dtype=int)
                o[:l0 + l1 + l2 + l3 + l4 + l5] = 0.
                uv = CollarOptimizer(np.concatenate([x, y], axis=1), uv, o, face).optimize()
                success = success and self.check_bound(uv, [0.50, 1.00, 0.50, 0.75], info_str)
            else:
                raise ValueError(part_name)
        else:
            raise ValueError(part_name)
        
        return uv, success, info


class VestCloseSP(VestClose):
    """Vest Close Single Part"""
    _meta = dict(name="vest_close_sp")
    all_part_name_type = VestClose.all_part_name_type
    def _calculate_uv(
        self, 
        part_name: all_part_name_type, 
        face: np.ndarray, 
        xy_dict: dict[all_part_name_type, np.ndarray], 
        affine_dict: dict[all_part_name_type, dict[GarmentBoundaryEdge, np.ndarray]]
    ):
        success = True
        info = {}

        xy_all = np.concatenate(list(xy_dict.values()), axis=0)
        scale = np.max(xy_all.max(axis=0) - xy_all.min(axis=0))
        x, y, s = xy_dict[part_name][:, 0], xy_dict[part_name][:, 1], scale * 2.2 # leave for some space
        x, y = (x - xy_all[:, 0].min())[:, None] / s, (y - xy_all[:, 1].min())[:, None] / s

        info_str = f"{part_name}"
        if part_name in ["front", "collarf"]:
            uv = np.concatenate([x, y], axis=1)
            success = success and self.check_bound(uv, [0.0, 0.5, 0.0, 0.5], info_str)
        elif part_name in ["back", "collarb"]:
            uv = np.concatenate([x, y + 0.5], axis=1)
            success = success and self.check_bound(uv, [0.0, 0.5, 0.5, 1.0], info_str)
        else:
            raise ValueError(part_name)
        
        return uv, success, info