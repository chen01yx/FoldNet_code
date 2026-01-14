from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from .common.typea import GarmentCfgTypeA, GarmentTypeA
from ..base_cls import GarmentBoundaryEdge, GarmentPart, Point2

import garmentds.common.utils as utils


@dataclass
class HoodedCloseCfg(GarmentCfgTypeA):
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
    - ht = hood_top
    """

    ### 2d geometry keypoints
    l_collar: Point2 = Point2(-0.09, +0.3)
    r_collar: Point2 = Point2(+0.09, +0.3)
    neck_b: Point2 = Point2(0.0, +0.25)
    neck_f: Point2 = Point2(0.0, +0.27)

    l_shoulder: Point2 = Point2(-0.3, +0.2)
    r_shoulder: Point2 = Point2(+0.3, +0.2)
    l_armpit: Point2 = Point2(-0.25, +0.1)
    r_armpit: Point2 = Point2(+0.25, +0.1)
    l_corner: Point2 = Point2(-0.25, -0.3)
    r_corner: Point2 = Point2(+0.25, -0.3)
    spine_bottom_f: Point2 = Point2(0.0, -0.3)
    spine_bottom_b: Point2 = Point2(0.0, -0.3)

    l_sleeve_top: Point2 = Point2(-0.55, -0.05)
    r_sleeve_top: Point2 = Point2(+0.55, -0.05)
    l_sleeve_bottom: Point2 = Point2(-0.47, -0.13)
    r_sleeve_bottom: Point2 = Point2(+0.47, -0.13)

    l_nf_cl: list[Point2] = field(default_factory=lambda:[(-0.04, +0.28)])
    r_nf_cl: list[Point2] = field(default_factory=lambda:[(+0.04, +0.28)])
    l_nb_cl: list[Point2] = field(default_factory=lambda:[(-0.04, +0.26)])
    r_nb_cl: list[Point2] = field(default_factory=lambda:[(+0.04, +0.26)])
    l_cl_sh: list[Point2] = field(default_factory=lambda:[(-0.20, +0.27)])
    r_cl_sh: list[Point2] = field(default_factory=lambda:[(+0.20, +0.27)])
    
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

    hood_top: Point2 = Point2(0.0, 0.51)
    l_co_ht: list[Point2] = field(default_factory=lambda:[Point2(-0.11, +0.38), Point2(-0.08, +0.47)])
    r_co_ht: list[Point2] = field(default_factory=lambda:[Point2(+0.11, +0.38), Point2(+0.08, +0.47)])

    ### 3d generation args
    edge_max_z: float = 0.05
    edge_width: float = 0.09
    boundary_dx: float = 0.015
    boundary_dense_n: int = 10000
    interior_num: dict = field(default_factory=lambda: dict(
        front = 2000, back = 2000, 
        leftf = 300, leftb = 300, 
        rightf = 300, rightb = 300, 
        hood = 1000,
    ))

    def symmetry(self):
        def mirror(x, y): return Point2(-x, y)
        for r_keypoint in [
            "r_shoulder", "r_collar", "r_armpit",
            "r_corner", "r_sleeve_top", "r_sleeve_bottom",
        ]:
            setattr(self, r_keypoint, mirror(*getattr(self, "l_" + r_keypoint[2:])))
        
        for keypoint in ["spine_bottom_f", "spine_bottom_b", "neck_f", "neck_b", "hood_top"]:
            setattr(self, keypoint, Point2(0., getattr(self, keypoint)[1]))
        
        for r_keypoint in [
            "r_nf_cl", "r_nb_cl", "r_cl_sh", 
            "r_sh_st", "r_st_sb", "r_sb_ar", 
            "r_sh_ar", "r_ar_cn", "r_cn_bt", 
            "r_co_ht", 
        ]:
            l_keypoint = "l_" + r_keypoint[2:]
            new_r_val = []
            for val in getattr(self, l_keypoint):
                new_r_val.append(mirror(*val))
            setattr(self, r_keypoint, new_r_val)
    
    def sanity_check(self):
        pass


class HoodedClose(GarmentTypeA):
    _meta = dict(name="hooded_close")
    all_part_name_type = Literal["front", "back", "leftf", "leftb", "rightf", "rightb", "hood"]
    all_part = dict(
        front = GarmentPart(False, False, False),
        back = GarmentPart(True, False, False),
        leftf = GarmentPart(False, False, False),
        leftb = GarmentPart(True, False, False),
        rightf = GarmentPart(False, False, False),
        rightb = GarmentPart(True, False, False),
        hood = GarmentPart(True, False, True)
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
        hood = [
            GarmentBoundaryEdge("l_collar", "l_co_ht", "hood_top", False, connect=True),
            GarmentBoundaryEdge("hood_top", "r_co_ht", "r_collar", True, connect=True, include="both"),
            GarmentBoundaryEdge("r_collar", "r_nb_cl", "neck_b", True, connect=False, include="end"),
            GarmentBoundaryEdge("neck_b", "l_nb_cl", "l_collar", False, connect=False, include="none"),
        ]
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
        ("back", "rightb") : [
            (GarmentBoundaryEdge("r_shoulder", "r_sh_ar", "r_armpit", False, connect=False, include="none"),
             GarmentBoundaryEdge("r_armpit", "r_sh_ar", "r_shoulder", True, connect=False, include="none"),)
        ],
        ("back", "hood") : [
            (GarmentBoundaryEdge("neck_b", "r_nb_cl", "r_collar", False, connect=False),
             GarmentBoundaryEdge("r_collar", "r_nb_cl", "neck_b", True, connect=False, include="end"),),
            (GarmentBoundaryEdge("l_collar", "l_nb_cl", "neck_b", True, connect=False, include="none"),
             GarmentBoundaryEdge("neck_b", "l_nb_cl", "l_collar", False, connect=False, include="none")),
        ]
    } # identify which edge should be re-used, format: (p, p'): [(e1, e1'), (e2, e2'), ...] means `e'` in `p'` should be the same as `e` in `p`

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(cfg=utils.dataclass_init_from_dict(HoodedCloseCfg, kwargs))
        self._cfg: HoodedCloseCfg
    
    def _draw(self, width: int, height: int, xy2ij: Callable[[float, float], tuple[int, int]], cfg: HoodedCloseCfg):
        img_np = np.zeros((height, width, 4), dtype=np.uint8)
        
        mask_front = self._get_mask(width, height, xy2ij, cfg, "front")
        mask_back = self._get_mask(width, height, xy2ij, cfg, "back")
        mask_left = self._get_mask(width, height, xy2ij, cfg, "leftf")
        mask_right = self._get_mask(width, height, xy2ij, cfg, "rightf")
        mask_hood = self._get_mask(width, height, xy2ij, cfg, "hood")

        img_np[np.where(mask_hood)] = (127, 191, 0, 255)
        img_np[np.where(mask_front)] = (0, 0, 255, 255)
        img_np[np.where(mask_back)] = (0, 127, 0, 255)
        img_np[np.where(mask_left)] = (255, 0, 0, 255)
        img_np[np.where(mask_right)] = (255, 127, 0, 255)
        
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
        x, y = (x - xy_all[:, 0].min())[:, None] / s, (y - xy_all[:, 1].min())[:, None] / s

        info_str = f"{part_name}"
        if part_name in ["front", "leftf", "rightf"]:
            uv = np.concatenate([x, y], axis=1)
            success = success and self.check_bound(uv, [0.0, 0.5, 0.0, 0.5], info_str)
        elif part_name in ["back", "leftb", "rightb"]:
            uv = np.concatenate([x, y + 0.5], axis=1)
            success = success and self.check_bound(uv, [0.0, 0.5, 0.5, 1.0], info_str)
        elif part_name == "hood":
            uv = np.concatenate([x + 0.5, y + 0.5], axis=1)
            success = success and self.check_bound(uv, [0.50, 1.00, 0.50, 1.00], info_str)
        else:
            raise ValueError(part_name)
        
        return uv, success, info