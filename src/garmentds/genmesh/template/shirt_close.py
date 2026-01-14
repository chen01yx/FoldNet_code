from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from .common.typea import GarmentCfgTypeA, GarmentTypeA
from ..base_cls import GarmentBoundaryEdge, GarmentPart, Point2
from ..make_collar import CollarOptimizer

import garmentds.common.utils as utils


@dataclass
class ShirtCloseCfg(GarmentCfgTypeA):
    """
    - sh = shoulder
    - cl = collar
    - nf = neck_f
    - nb = neck_b
    - st = sleeve_top
    - sb = sleeve_bottom
    - ar = armpit
    - cn = corner
    - bt = spine_bottom
    """

    ### 2d geometry keypoints
    l_collar: Point2 = Point2(-0.10, +0.3)
    r_collar: Point2 = Point2(+0.10, +0.3)
    l_neck_f: Point2 = Point2(-0.005, +0.20)
    r_neck_f: Point2 = Point2(+0.005, +0.20)
    neck_c: Point2 = Point2(0.0, +0.20)
    neck_b: Point2 = Point2(0.0, +0.27)

    l_shoulder: Point2 = Point2(-0.3, +0.2)
    r_shoulder: Point2 = Point2(+0.3, +0.2)
    l_armpit: Point2 = Point2(-0.25, +0.1)
    r_armpit: Point2 = Point2(+0.25, +0.1)
    l_corner: Point2 = Point2(-0.25, -0.3)
    r_corner: Point2 = Point2(+0.25, -0.3)
    spine_bottom_f: Point2 = Point2(0.0, -0.3)
    spine_bottom_b: Point2 = Point2(0.0, -0.3)

    l_sleeve_top: Point2 = Point2(-0.4, +0.1)
    r_sleeve_top: Point2 = Point2(+0.4, +0.1)
    l_sleeve_bottom: Point2 = Point2(-0.3, +0.02)
    r_sleeve_bottom: Point2 = Point2(+0.3, +0.02)

    l_nf_cl: list[Point2] = field(default_factory=lambda:[Point2(-0.05, +0.24)])
    r_nf_cl: list[Point2] = field(default_factory=lambda:[Point2(+0.05, +0.24)])
    l_nb_cl: list[Point2] = field(default_factory=lambda:[Point2(-0.06, +0.28)])
    r_nb_cl: list[Point2] = field(default_factory=lambda:[Point2(+0.06, +0.28)])
    l_cl_sh: list[Point2] = field(default_factory=lambda:[Point2(-0.20, +0.27)])
    r_cl_sh: list[Point2] = field(default_factory=lambda:[Point2(+0.20, +0.27)])
    
    l_sh_st: list[Point2] = field(default_factory=lambda:[])
    r_sh_st: list[Point2] = field(default_factory=lambda:[])
    l_st_sb: list[Point2] = field(default_factory=lambda:[])
    r_st_sb: list[Point2] = field(default_factory=lambda:[])
    l_sb_ar: list[Point2] = field(default_factory=lambda:[])
    r_sb_ar: list[Point2] = field(default_factory=lambda:[])

    l_sh_ar: list[Point2] = field(default_factory=lambda:[])
    r_sh_ar: list[Point2] = field(default_factory=lambda:[])
    l_ar_cn: list[Point2] = field(default_factory=lambda:[])
    r_ar_cn: list[Point2] = field(default_factory=lambda:[])
    l_cn_bt: list[Point2] = field(default_factory=lambda:[])
    r_cn_bt: list[Point2] = field(default_factory=lambda:[])
    l_bt_nf: list[Point2] = field(default_factory=lambda:[])
    r_bt_nf: list[Point2] = field(default_factory=lambda:[])

    l_collar_o: Point2 = Point2(-0.116, +0.303)
    r_collar_o: Point2 = Point2(+0.116, +0.303)
    l_neck_f_o: Point2 = Point2(-0.025, +0.18)
    r_neck_f_o: Point2 = Point2(+0.025, +0.18)
    neck_b_o: Point2 = Point2(0.0, +0.255)

    l_nf_cl_o: list[Point2] = field(default_factory=lambda:[Point2(-0.07, +0.22)])
    r_nf_cl_o: list[Point2] = field(default_factory=lambda:[Point2(+0.07, +0.22)])
    l_nb_cl_o: list[Point2] = field(default_factory=lambda:[Point2(-0.07, +0.27)])
    r_nb_cl_o: list[Point2] = field(default_factory=lambda:[Point2(+0.07, +0.27)])
    l_cl_cl_o: list[Point2] = field(default_factory=lambda:[])
    r_cl_cl_o: list[Point2] = field(default_factory=lambda:[])
    l_nf_nf_o: list[Point2] = field(default_factory=lambda:[])
    r_nf_nf_o: list[Point2] = field(default_factory=lambda:[])
    l_nc_nf: list = field(default_factory=lambda:[])
    r_nc_nf: list = field(default_factory=lambda:[])

    ### 3d generation args
    edge_max_z: float = 0.05
    edge_width: float = 0.09
    boundary_dx: float = 0.015
    boundary_dense_n: int = 10000
    interior_num: dict = field(default_factory=lambda: dict(
        front = 2000, back = 2000, 
        leftf = 150, leftb = 150, 
        rightf = 150, rightb = 150,
        collarfl = 10, collarfr = 10, collarb = 20
    ))
  
    def symmetry(self):
        def mirror(x, y): return Point2(-x, y)
        for r_keypoint in [
            "r_collar", "r_neck_f", 
            "r_shoulder", "r_armpit", "r_corner", 
            "r_sleeve_top", "r_sleeve_bottom",
            "r_collar_o", "r_neck_f_o",
        ]:
            setattr(self, r_keypoint, mirror(*getattr(self, "l_" + r_keypoint[2:])))
        
        for keypoint in ["neck_b", "neck_c", "spine_bottom_b", "spine_bottom_f", "neck_b_o"]:
            setattr(self, keypoint, Point2(0., getattr(self, keypoint)[1]))
        
        for r_keypoint in [
            "r_nf_cl", "r_nb_cl", "r_cl_sh", 
            "r_sh_st", "r_st_sb", "r_sb_ar", 
            "r_sh_ar", "r_ar_cn", "r_cn_bt", "r_bt_nf", 
            "r_nf_cl_o", "r_nb_cl_o", "r_cl_cl_o", "r_nf_nf_o", "r_nc_nf"
        ]:
            l_keypoint = "l_" + r_keypoint[2:]
            new_r_val = []
            for val in getattr(self, l_keypoint):
                new_r_val.append(mirror(*val))
            setattr(self, r_keypoint, new_r_val)
    
    def sanity_check(self):
        pass


class ShirtClose(GarmentTypeA):
    _meta = dict(name="shirt_close")
    all_part_name_type = Literal["front", "back", "leftf", "leftb", "rightf", "rightb", "collarfl", "collarfr", "collarb"]
    all_part = dict(
        front = GarmentPart(False, False, False),
        back = GarmentPart(True, False, False),
        leftf = GarmentPart(False, False, False),
        leftb = GarmentPart(True, False, False),
        rightf = GarmentPart(False, False, False),
        rightb = GarmentPart(True, False, False),
        collarfl = GarmentPart(False, True, False),
        collarfr = GarmentPart(False, True, False),
        collarb = GarmentPart(True, True, False),
    )
    all_edge: dict[all_part_name_type, list[GarmentBoundaryEdge]] = dict(
        front = [
            GarmentBoundaryEdge("l_shoulder", "l_cl_sh", "l_collar", True, connect=True, include="both"),
            GarmentBoundaryEdge("l_collar", "l_nf_cl", "l_neck_f", True, connect=False, include="end"),
            GarmentBoundaryEdge("l_neck_f", "l_nc_nf", "neck_c", True, connect=False, include="none"),
            GarmentBoundaryEdge("neck_c", "r_nc_nf", "r_neck_f", True, connect=False,),
            GarmentBoundaryEdge("r_neck_f", "r_nf_cl", "r_collar", False, connect=False),
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
        leftf = [
            GarmentBoundaryEdge("l_shoulder", "l_sh_ar", "l_armpit", False, connect=False, include="none"),
            GarmentBoundaryEdge("l_armpit", "l_sb_ar", "l_sleeve_bottom", True, connect=True, include="both"),
            GarmentBoundaryEdge("l_sleeve_bottom", "l_st_sb", "l_sleeve_top", True, connect=False, include="none"),
            GarmentBoundaryEdge("l_sleeve_top", "l_sh_st", "l_shoulder", True, connect=True, include="both"),
        ],
        leftb = [
            GarmentBoundaryEdge("l_shoulder", "l_sh_ar", "l_armpit", False, connect=False, include="none"),
            GarmentBoundaryEdge("l_armpit", "l_sb_ar", "l_sleeve_bottom", True, connect=True, include="both"),
            GarmentBoundaryEdge("l_sleeve_bottom", "l_st_sb", "l_sleeve_top", True, connect=False, include="none"),
            GarmentBoundaryEdge("l_sleeve_top", "l_sh_st", "l_shoulder", True, connect=True, include="both"),
        ],
        rightf = [
            GarmentBoundaryEdge("r_shoulder", "r_sh_st", "r_sleeve_top", False, connect=True, include="both"),
            GarmentBoundaryEdge("r_sleeve_top", "r_st_sb", "r_sleeve_bottom", False, connect=False, include="none"),
            GarmentBoundaryEdge("r_sleeve_bottom", "r_sb_ar", "r_armpit", False, connect=True, include="both"),
            GarmentBoundaryEdge("r_armpit", "r_sh_ar", "r_shoulder", True, connect=False, include="none"),
        ],
        rightb = [
            GarmentBoundaryEdge("r_shoulder", "r_sh_st", "r_sleeve_top", False, connect=True, include="both"),
            GarmentBoundaryEdge("r_sleeve_top", "r_st_sb", "r_sleeve_bottom", False, connect=False, include="none"),
            GarmentBoundaryEdge("r_sleeve_bottom", "r_sb_ar", "r_armpit", False, connect=True, include="both"),
            GarmentBoundaryEdge("r_armpit", "r_sh_ar", "r_shoulder", True, connect=False, include="none"),
        ],
        collarfl = [
            GarmentBoundaryEdge("l_collar", "l_nf_cl", "l_neck_f", True, connect=False, include="end"),
            GarmentBoundaryEdge("l_neck_f", "l_nf_nf_o", "l_neck_f_o", False, connect=False, include="none"),
            GarmentBoundaryEdge("l_neck_f_o", "l_nf_cl_o", "l_collar_o", False, connect=False),
            GarmentBoundaryEdge("l_collar_o", "l_cl_cl_o", "l_collar", False, connect=True, include="both"),
        ],
        collarfr = [
            GarmentBoundaryEdge("r_neck_f", "r_nf_cl", "r_collar", False, connect=False),
            GarmentBoundaryEdge("r_collar", "r_cl_cl_o", "r_collar_o", True, connect=True, include="both"),
            GarmentBoundaryEdge("r_collar_o", "r_nf_cl_o", "r_neck_f_o", False, connect=False, include="end"),
            GarmentBoundaryEdge("r_neck_f_o", "r_nf_nf_o", "r_neck_f", False, connect=False, include="none"),
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
        ("front", "leftf") : [
            (GarmentBoundaryEdge("l_armpit", "l_sh_ar", "l_shoulder", True, connect=False, include="none"),
             GarmentBoundaryEdge("l_shoulder", "l_sh_ar", "l_armpit", False, connect=False, include="none"),)
        ],
        ("back", "leftb") : [
            (GarmentBoundaryEdge("l_armpit", "l_sh_ar", "l_shoulder", True, connect=False, include="none"),
             GarmentBoundaryEdge("l_shoulder", "l_sh_ar", "l_armpit", False, connect=False, include="none"),)
        ],
        ("front", "rightf") : [
            (GarmentBoundaryEdge("r_shoulder", "r_sh_ar", "r_armpit", False, connect=False, include="none"),
             GarmentBoundaryEdge("r_armpit", "r_sh_ar", "r_shoulder", True, connect=False, include="none"),)
        ],
        ("front", "collarfl") : [
            (GarmentBoundaryEdge("l_collar", "l_nf_cl", "l_neck_f", True, connect=False, include="end"),
             GarmentBoundaryEdge("l_collar", "l_nf_cl", "l_neck_f", True, connect=False, include="end"),), 
        ],
        ("front", "collarfr") : [
            (GarmentBoundaryEdge("r_neck_f", "r_nf_cl", "r_collar", False, connect=False),
             GarmentBoundaryEdge("r_neck_f", "r_nf_cl", "r_collar", False, connect=False),), 
        ],
        ("back", "collarb") : [
            (GarmentBoundaryEdge("l_collar", "l_nb_cl", "neck_b", True, connect=False, include="none"),
             GarmentBoundaryEdge("neck_b", "l_nb_cl", "l_collar", False, connect=False, include="none"),),
            (GarmentBoundaryEdge("neck_b", "r_nb_cl", "r_collar", False, connect=False),
             GarmentBoundaryEdge("r_collar", "r_nb_cl", "neck_b", True, connect=False, include="end"),),
        ],
        ("collarfl", "collarb") : [
            (GarmentBoundaryEdge("l_collar_o", "l_cl_cl_o", "l_collar", False, connect=True, include="both"),
             GarmentBoundaryEdge("l_collar", "l_cl_cl_o", "l_collar_o", True, connect=True, include="both"),),
        ],
        ("collarfr", "collarb") : [
            (GarmentBoundaryEdge("r_collar", "r_cl_cl_o", "r_collar_o", True, connect=True, include="both"),
             GarmentBoundaryEdge("r_collar_o", "r_cl_cl_o", "r_collar", False, connect=True, include="both"),),
        ],
    } # identify which edge should be re-used, format: (p, p'): [(e1, e1'), (e2, e2'), ...] means `e'` in `p'` should be the same as `e` in `p`

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(cfg=utils.dataclass_init_from_dict(ShirtCloseCfg, kwargs))
        self._cfg: ShirtCloseCfg
        
    def _draw(self, width: int, height: int, xy2ij: Callable[[float, float], tuple[int, int]], cfg: ShirtCloseCfg):
        img_np = np.zeros((height, width, 4), dtype=np.uint8)
        
        mask_front = self._get_mask(width, height, xy2ij, cfg, "front")
        mask_back = self._get_mask(width, height, xy2ij, cfg, "back")

        mask_left = self._get_mask(width, height, xy2ij, cfg, "leftf")
        mask_right = self._get_mask(width, height, xy2ij, cfg, "rightf")
        
        mask_collarfl = self._get_mask(width, height, xy2ij, cfg, "collarfl")
        mask_collarfr = self._get_mask(width, height, xy2ij, cfg, "collarfr")
        mask_collarb = self._get_mask(width, height, xy2ij, cfg, "collarb")

        img_np[np.where(mask_back)] = (0, 127, 0, 255)
        img_np[np.where(mask_front)] = (127, 0, 255, 255)

        img_np[np.where(mask_left)] = (255, 0, 0, 255)
        img_np[np.where(mask_right)] = (255, 127, 0, 255)

        img_np[np.where(mask_collarb)] = (127, 191, 0, 255)
        img_np[np.where(mask_collarfl)] = (255, 127, 0, 255)
        img_np[np.where(mask_collarfr)] = (127, 127, 0, 255)

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

        val_list = list(affine_dict[part_name].values())
        def set_val(arr: np.ndarray, start_idx: int, total_len: int, val): arr[start_idx: start_idx + total_len] = val
        collarfl_length = sum([len(x) for x in affine_dict["collarfl"].values()]) - 1
        collarfr_length = sum([len(x) for x in affine_dict["collarfr"].values()]) - 1
        collarb_length = sum([len(x) for x in affine_dict["collarb"].values()]) - 1
        thispart_length = sum([len(x) for x in affine_dict[part_name].values()]) - 1
        scale_u_collar = 0.25 * thispart_length / max(collarfl_length, collarfr_length, collarb_length)

        info_str = f"{part_name}"
        if part_name == "front":
            uv = np.concatenate([x, y + 0.5], axis=1)
            success = success and self.check_bound(uv, [0.00, 0.50, 0.50, 1.00], info_str)
        elif part_name == "back":
            uv = np.concatenate([x, y], axis=1)
            success = success and self.check_bound(uv, [0.0, 0.5, 0.0, 0.5], info_str)
        elif part_name == "collarfl":
            l0, l1, l2, l3 = [len(v) for v in val_list]
            u, v = np.zeros(len(x)), np.zeros(len(x))

            set_val(u, 0, l0, np.arange(1, l0 + 1, 1) / l0)
            set_val(u, l0, l1, 1.)
            set_val(u, l0 + l1, l2, np.arange(l2, 0, -1) / l2)

            set_val(v, 0, l0, 1.)
            set_val(v, l0, l1, np.arange(l1, 0, -1) / (l1 + 1))
            set_val(v, l0 + l1 + l2, l3, np.linspace(0., 1., l3))

            u, v = u[:, None], v[:, None]
            uv = np.concatenate([u * scale_u_collar + 0.50, v * 0.05 + 0.75], axis=1)
            o = np.ones(len(x), dtype=int)
            o[:l0 + l1 + l2 + l3] = 0.
            uv = CollarOptimizer(np.concatenate([x, y], axis=1), uv, o, face).optimize()
            info["uv01"] = np.clip(np.concatenate([(uv[:, [0]] - 0.5) / scale_u_collar, (uv[:, [1]] - 0.75) / 0.05], axis=1), a_min=0.0, a_max=1.0)
            success = success and self.check_bound(uv, [0.50, 0.75, 0.75, 1.00], info_str)
        elif part_name == "collarfr":
            l0, l1, l2, l3 = [len(v) for v in val_list]
            u, v = np.zeros(len(x)), np.zeros(len(x))

            set_val(u, 0, l0, np.arange(0, l0, 1) / l0)
            set_val(u, l0, l1, 1.)
            set_val(u, l0 + l1, l2, np.arange(l2 - 1, -1, -1) / l2)
            
            set_val(v, 0, l0, 1.)
            set_val(v, l0, l1, np.linspace(1., 0., l1))
            set_val(v, l0 + l1 + l2, l3, np.arange(1, l3 + 1, 1) / (l3 + 1))

            u, v = u[:, None], v[:, None] # [0, 1]
            uv = np.concatenate([u * scale_u_collar + 0.75, v * 0.05 + 0.75], axis=1)
            o = np.ones(len(x), dtype=int)
            o[:l0 + l1 + l2 + l3] = 0.
            uv = CollarOptimizer(np.concatenate([x, y], axis=1), uv, o, face).optimize()
            info["uv01"] = np.clip(np.concatenate([(uv[:, [0]] - 0.75) / scale_u_collar, (uv[:, [1]] - 0.75) / 0.05], axis=1), a_min=0.0, a_max=1.0)
            success = success and self.check_bound(uv, [0.75, 1.00, 0.75, 1.00], info_str)
        elif part_name == "collarb":
            l0, l1, l2, l3, l4, l5 = [len(v) for v in val_list]
            u, v = np.zeros(len(x)), np.zeros(len(x))

            set_val(u, l0, l1, np.arange(1, l1 + 1, 1) / (l1 + l2 + 1))
            set_val(u, l0 + l1, l2, np.arange(l1 + 1, l1 + 1 + l2, 1) / (l1 + l2 + 1))
            set_val(u, l0 + l1 + l2, l3, 1.)
            set_val(u, l0 + l1 + l2 + l3, l4, np.arange(l4 + l5, l5, -1) / (l4 + l5 + 1))
            set_val(u, l0 + l1 + l2 + l3 + l4, l5, np.arange(l5, 0, -1) / (l4 + l5 + 1))

            set_val(v, 0, l0, np.linspace(0., 1., l0))
            set_val(v, l0, l1 + l2, 1.)
            set_val(v, l0 + l1 + l2, l3, np.linspace(1., 0., l3))

            u, v = u[:, None], v[:, None]
            uv = np.concatenate([u * scale_u_collar + 0.5, v * 0.05 + 0.50], axis=1)
            o = np.ones(len(x), dtype=int)
            o[:l0 + l1 + l2 + l3 + l4 + l5] = 0.
            uv = CollarOptimizer(np.concatenate([x, y], axis=1), uv, o, face).optimize()
            info["uv01"] = np.clip(np.concatenate([(uv[:, [0]] - 0.5) / scale_u_collar, (uv[:, [1]] - 0.5) / 0.05], axis=1), a_min=0.0, a_max=1.0)
            success = success and self.check_bound(uv, [0.50, 1.00, 0.50, 0.75], info_str)            
        elif part_name == "leftf":
            uv = np.concatenate([x + 0.75, y], axis=1)
            success = success and self.check_bound(uv, [0.75, 1.00, 0.00, 0.25], info_str)
        elif part_name == "leftb":
            uv = np.concatenate([x + 0.5, y], axis=1)
            success = success and self.check_bound(uv, [0.50, 0.75, 0.00, 0.25], info_str)
        elif part_name == "rightf":
            uv = np.concatenate([x + 0.75, y + 0.25], axis=1)
            success = success and self.check_bound(uv, [0.75, 1.00, 0.25, 0.50], info_str)
        elif part_name == "rightb":
            uv = np.concatenate([x + 0.5, y + 0.25], axis=1)
            success = success and self.check_bound(uv, [0.50, 0.75, 0.25, 0.50], info_str)
        else:
            raise ValueError(part_name)
        
        return uv, success, info