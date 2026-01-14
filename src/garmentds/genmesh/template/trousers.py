from dataclasses import dataclass, field
from typing import Callable, Literal

from PIL import Image, ImageDraw
import numpy as np

from .common.typea import GarmentCfgTypeA, GarmentTypeA
from ..base_cls import GarmentBoundaryEdge, GarmentPart, Point2

import garmentds.common.utils as utils


@dataclass
class TrousersCfg(GarmentCfgTypeA):
    """
    - co = corner
    - lo = leg_o
    - li = leg_i
    - cr = crotch
    - tc = top center
    """

    ### 2d geometry keypoints
    l_corner: Point2 = Point2(-0.25, +0.40)
    r_corner: Point2 = Point2(+0.25, +0.40)
    l_leg_o: Point2 = Point2(-0.27, -0.40)
    r_leg_o: Point2 = Point2(+0.27, -0.40)
    l_leg_i: Point2 = Point2(-0.10, -0.41)
    r_leg_i: Point2 = Point2(+0.10, -0.41)
    crotch: Point2 = Point2(0.0, +0.2)
    top_ctr_f: Point2 = Point2(0.0, +0.4)
    top_ctr_b: Point2 = Point2(0.0, +0.4)

    l_tc_co: list[Point2] = field(default_factory=lambda:[])
    r_tc_co: list[Point2] = field(default_factory=lambda:[])
    l_co_lo: list[Point2] = field(default_factory=lambda:[Point2(-0.26, 0.0)])
    r_co_lo: list[Point2] = field(default_factory=lambda:[Point2(+0.26, 0.0)])
    l_lo_li: list[Point2] = field(default_factory=lambda:[])
    r_lo_li: list[Point2] = field(default_factory=lambda:[])
    l_li_cr: list[Point2] = field(default_factory=lambda:[Point2(-0.06, -0.1)])
    r_li_cr: list[Point2] = field(default_factory=lambda:[Point2(+0.06, -0.1)])

    ### 3d generation args
    edge_max_z: float = 0.05
    edge_width: float = 0.09
    boundary_dx: float = 0.015
    boundary_dense_n: int = 10000
    interior_num: dict = field(default_factory=lambda: dict(
        front = 2000, back = 2000, 
    ))

    def symmetry(self):
        def mirror(x, y): return Point2(-x, y)
        for r_keypoint in [
            "r_corner", "r_leg_o", "r_leg_i",
        ]:
            setattr(self, r_keypoint, mirror(*getattr(self, "l_" + r_keypoint[2:])))
        
        for keypoint in ["crotch", "top_ctr_b", "top_ctr_f"]:
            setattr(self, keypoint, Point2(0., getattr(self, keypoint)[1]))
        
        for r_keypoint in [
            "r_tc_co", "r_co_lo", "r_lo_li", "r_li_cr",
        ]:
            l_keypoint = "l_" + r_keypoint[2:]
            new_r_val = []
            for val in getattr(self, l_keypoint):
                new_r_val.append(mirror(*val))
            setattr(self, r_keypoint, new_r_val)
    
    def sanity_check(self):
        pass


class Trousers(GarmentTypeA):
    _meta = dict(name="trousers")
    all_part_name_type = Literal["front", "back"]
    all_part = dict(
        front = GarmentPart(False, False, False),
        back = GarmentPart(True, False, False),
    )
    all_edge: dict[all_part_name_type, list[GarmentBoundaryEdge]] = dict(
        front = [
            GarmentBoundaryEdge("l_corner", "l_tc_co", "top_ctr_f", True, connect=False, include="none"),
            GarmentBoundaryEdge("top_ctr_f", "r_tc_co", "r_corner", False, connect=False),
            GarmentBoundaryEdge("r_corner", "r_co_lo", "r_leg_o", False, connect=True, include="both"),
            GarmentBoundaryEdge("r_leg_o", "r_lo_li", "r_leg_i", False, connect=False, include="none"),
            GarmentBoundaryEdge("r_leg_i", "r_li_cr", "crotch", False, connect=True),
            GarmentBoundaryEdge("crotch", "l_li_cr", "l_leg_i", True, connect=True, include="both"),
            GarmentBoundaryEdge("l_leg_i", "l_lo_li", "l_leg_o", True, connect=False, include="none"),
            GarmentBoundaryEdge("l_leg_o", "l_co_lo", "l_corner", True, connect=True, include="both"),
        ],
        back = [
            GarmentBoundaryEdge("l_corner", "l_tc_co", "top_ctr_b", True, connect=False, include="none"),
            GarmentBoundaryEdge("top_ctr_b", "r_tc_co", "r_corner", False, connect=False),
            GarmentBoundaryEdge("r_corner", "r_co_lo", "r_leg_o", False, connect=True, include="both"),
            GarmentBoundaryEdge("r_leg_o", "r_lo_li", "r_leg_i", False, connect=False, include="none"),
            GarmentBoundaryEdge("r_leg_i", "r_li_cr", "crotch", False, connect=True),
            GarmentBoundaryEdge("crotch", "l_li_cr", "l_leg_i", True, connect=True, include="both"),
            GarmentBoundaryEdge("l_leg_i", "l_lo_li", "l_leg_o", True, connect=False, include="none"),
            GarmentBoundaryEdge("l_leg_o", "l_co_lo", "l_corner", True, connect=True, include="both"),
        ],
    )

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(cfg=utils.dataclass_init_from_dict(TrousersCfg, kwargs))
        self._cfg: TrousersCfg
    
    def _draw(self, width: int, height: int, xy2ij: Callable[[float, float], tuple[int, int]], cfg: TrousersCfg):
        img_np = np.zeros((height, width, 4), dtype=np.uint8)

        mask_front = self._get_mask(width, height, xy2ij, cfg, "front")
        mask_back = self._get_mask(width, height, xy2ij, cfg, "back")

        img_np[np.where(mask_front)] = (0, 0, 255, 255)
        img_np[np.where(np.logical_and(mask_back, np.logical_not(mask_front)))] = (0, 127, 0, 255)

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
        else:
            raise ValueError(part_name)
        
        return uv, success, info