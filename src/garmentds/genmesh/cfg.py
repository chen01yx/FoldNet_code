from typing import Literal, Union
from dataclasses import asdict
import numpy as np

from .template import *
from ..genmesh.base_cls import Point2
import garmentds.common.utils as utils

available_category_type = Literal[
    "tshirt", "tshirt_cp", "trousers", 
    "vest", "vest_sp", "vest_close", "vest_close_sp",
    "shirt", "shirt_close", "hooded", "hooded_close"
]


rand = np.random.rand


def rand_xy(x_range: tuple[float, float], y_range: tuple[float, float]):
    return Point2(
        float(utils.map_01_ab(rand(), *x_range)), 
        float(utils.map_01_ab(rand(), *y_range)),
    )


def rand_unit():
    """[-1, +1]"""
    return Point2(rand(2) * 2. - 1.)


class ShoulderSleeveBodyCfgGenerator:
    @staticmethod
    def randomize_shoulder_sleeve_body(
        cfg: Union[TShirtCfg, ShirtCfg, ShirtCloseCfg, HoodedCfg, HoodedCloseCfg], 
        sleeve_length_str: Literal["short", "long"], 
        dist_version=0, 
    ):
        # shoulder
        if dist_version in [0, 1, 2]:
            cfg.l_shoulder = rand_xy((-0.28, -0.32), (0.24, 0.28))
        elif dist_version == 3:
            cfg.l_shoulder = rand_xy((-0.26, -0.30), (0.24, 0.28))
        else:
            raise ValueError(f"Invalid distribution version: {dist_version}")
        cfg.l_cl_sh = [Point2((cfg.l_collar.x + cfg.l_shoulder.x) / 2, (cfg.l_collar.y * 1.5 + cfg.l_shoulder.y) / 2.5) + rand_unit() * 0.01]

        # sleeve
        if dist_version in [0, 1, 2]:
            sleeve_width = utils.map_01_ab(rand(), 0.10, 0.18)
        elif dist_version == 3:
            sleeve_width = utils.map_01_ab(rand(), 0.14, 0.18)
        else:
            raise ValueError(f"Invalid distribution version: {dist_version}")
        armpit_theta = np.radians(utils.map_01_ab(rand(), 85, 95))
        if sleeve_length_str == "short":
            if dist_version in [0, 1, 2]:
                sleeve_theta = np.radians(utils.map_01_ab(rand(), 60, 75))
                sleeve_length = utils.map_01_ab(rand(), 0.18, 0.24)
            elif dist_version == 3:
                sleeve_theta = np.radians(utils.map_01_ab(rand(), 55, 70))
                sleeve_length = utils.map_01_ab(rand(), 0.16, 0.20)
            else:
                raise ValueError(f"Invalid distribution version: {dist_version}")
        elif sleeve_length_str == "long":
            sleeve_theta = np.radians(utils.map_01_ab(rand(), 60, 75))
            sleeve_length = utils.map_01_ab(rand(), 0.40, 0.60)
        else:
            raise NotImplementedError(sleeve_length_str)
        cfg.l_armpit = cfg.l_shoulder + Point2(sleeve_width * np.cos(armpit_theta), -sleeve_width * np.sin(armpit_theta)) * 1.3
        cfg.l_sleeve_top = cfg.l_shoulder + Point2(-sleeve_length * np.sin(sleeve_theta), -sleeve_length * np.cos(sleeve_theta))
        cfg.l_sleeve_bottom = cfg.l_sleeve_top + Point2(sleeve_width * np.cos(sleeve_theta), -sleeve_width * np.sin(sleeve_theta))

        # body
        if dist_version in [0, 1, 2]:
            cfg.l_corner = cfg.l_armpit + rand_xy((-0.02, +0.02), (-0.30, -0.60))
        elif dist_version == 3:
            cfg.l_corner = cfg.l_armpit + rand_xy((-0.02, +0.02), (-0.40, -0.60))
        else:
            raise ValueError(f"Invalid distribution version: {dist_version}")
        cfg.spine_bottom_b = Point2(0., cfg.l_corner.y + utils.map_01_ab(rand(), -0.03, +0.03))

        if isinstance(cfg, (TShirtCfg, ShirtCloseCfg, HoodedCloseCfg)):
            cfg.spine_bottom_f = Point2(0., cfg.spine_bottom_b.y)
        elif isinstance(cfg, (ShirtCfg, HoodedCfg)):
            cfg.l_spine_bottom = Point2(cfg.l_neck_f.x, cfg.spine_bottom_b.y)
        else:
            raise NotImplementedError(type(cfg))


class ShirtLikeCfgGenerator:
    @staticmethod
    def shirt_like_cfg_randomize_shoulder_sleeve_body_3dargs(
        cfg: Union[TShirtCfg, ShirtCfg, ShirtCloseCfg], 
        sleeve_length_str: Literal["short", "long"],
        dist_version=0, 
    ):
        ShoulderSleeveBodyCfgGenerator.randomize_shoulder_sleeve_body(cfg, sleeve_length_str, dist_version=dist_version)

        # 3d args
        cfg.edge_max_z = utils.map_01_ab(rand(), 0.04, 0.06)
        cfg.edge_width = utils.map_01_ab(rand(), 0.08, 0.10)
        if sleeve_length_str == "short":
            pass
        elif sleeve_length_str == "long":
            for part_str in ["leftf", "rightf", "leftb", "rightb"]:
                cfg.interior_num[part_str] *= 2
        else:
            raise NotImplementedError(sleeve_length_str)


class TShirtCfgGenerator(ShirtLikeCfgGenerator):
    @staticmethod
    def tshirt_default(sleeve_length_str: Literal["short", "long"]):
        if sleeve_length_str == "short":
            return TShirtCfg()
        elif sleeve_length_str == "long":
            return TShirtCfg(
                l_sleeve_top=Point2(-0.55, -0.05),
                r_sleeve_top=Point2(+0.55, -0.05),
                l_sleeve_bottom=Point2(-0.47, -0.13),
                r_sleeve_bottom=Point2(+0.47, -0.13),
            )
        else:
            raise ValueError(f"Invalid sleeve length: {sleeve_length_str}")

    @staticmethod
    def tshirt_random(sleeve_length_str: Literal["short", "long"], dist_version=0):
        cfg = TShirtCfg()

        # high-collar or short-collar
        if dist_version == 0:
            collar_theta = np.random.choice([
                np.radians(utils.map_01_ab(rand(), 0, 40)),  # short-collar
                np.radians(utils.map_01_ab(rand(), 60, 85)), # high-collar
            ])
        elif dist_version == 1:
            collar_theta = np.random.choice([
                np.radians(utils.map_01_ab(rand(), 0, 40)),  # short-collar
                np.radians(utils.map_01_ab(rand(), 60, 85)), # high-collar
            ], p=[0.8, 0.2])
        elif dist_version in [2, 3]:
            collar_theta = np.random.choice([
                np.radians(utils.map_01_ab(rand(), 0, 40)),  # short-collar
                np.radians(utils.map_01_ab(rand(), 60, 85)), # high-collar
            ], p=[1.0, 0.0])
        else:
            raise ValueError(f"Invalid distribution version: {dist_version}")
        print(f"[INFO] collar_theta: {np.degrees(collar_theta)}")
        
        square_collar = False
        if collar_theta < np.pi / 4:
            collar_width = utils.map_01_ab(rand(), 0.01, 0.02)
            if dist_version == 0:
                collar_type_idx = np.random.choice(3, p=[0.4, 0.3, 0.3])
            elif dist_version == 1:
                collar_type_idx = np.random.choice(3, p=[0.6, 0.2, 0.2])
            elif dist_version in [2, 3]:
                collar_type_idx = np.random.choice(3, p=[1.0, 0.0, 0.0])
            else:
                raise ValueError(f"Invalid distribution version: {dist_version}")
            print(f"[INFO] collar_type_idx:{collar_type_idx}")
            if collar_type_idx == 0: 
                ## thin collar
                cfg.l_collar = rand_xy((-0.10, -0.14), (0.28, 0.32))
                cfg.neck_f = rand_xy((0., 0.), (0.18, 0.22))
                if dist_version == 0:
                    cfg.neck_b = rand_xy((0., 0.), (cfg.neck_f.y+0.05, cfg.neck_f.y+0.14))
                elif dist_version in [1, 2, 3]:
                    cfg.neck_b = rand_xy((0., 0.), (cfg.l_collar.y-0.04, cfg.l_collar.y+0.02))
                else:
                    raise ValueError(f"Invalid distribution version: {dist_version}")
            elif collar_type_idx == 1:
                ## mid and square collar
                cfg.l_collar = rand_xy((-0.10, -0.14), (0.28, 0.32))
                cfg.neck_f = rand_xy((0., 0.), (0.18, 0.22))
                cfg.neck_b = rand_xy((0., 0.), (cfg.neck_f.y+0.015, cfg.neck_f.y+0.025))        
                cfg.l_nf_cl = [Point2((cfg.l_collar.x*85 + cfg.neck_f.x*15)/100, (cfg.l_collar.y+ cfg.neck_f.y*3)/4) + rand_unit() * 0.01]
                cfg.l_nb_cl = [Point2((cfg.l_collar.x*85 + cfg.neck_b.x*15)/100, (cfg.l_collar.y+ cfg.neck_b.y*3)/4)]
                square_collar = True              
            elif collar_type_idx == 2:
                ## wide collar
                cfg.l_collar = rand_xy((-0.21, -0.19), (0.28, 0.32))
                cfg.neck_f = rand_xy((0., 0.), (cfg.l_collar.y-0.05, cfg.l_collar.y-0.03))
                cfg.neck_b = rand_xy((0., 0.), (cfg.neck_f.y+0.02, cfg.neck_f.y+0.04))
                for part_str in ["collarf", "collarb"]:
                    cfg.interior_num[part_str] *= 2
            else:
                raise ValueError(f"Invalid collar type index: {collar_type_idx}")
        else:
            collar_width = utils.map_01_ab(rand(), 0.04, 0.07)
            cfg.l_collar = rand_xy((-0.10, -0.14), (0.28, 0.32))
            cfg.neck_f = rand_xy((0., 0.), (cfg.l_collar.y-0.04, cfg.l_collar.y-0.03))
            cfg.neck_b = rand_xy((0., 0.), (cfg.neck_f.y+0.015, cfg.neck_f.y+0.025))  
            for part_str in ["collarf", "collarb"]:
                cfg.interior_num[part_str] *= 4

        # collar
        if not square_collar:
            cfg.l_nf_cl = [Point2((cfg.l_collar.x + cfg.neck_f.x) / 2, (cfg.l_collar.y + cfg.neck_f.y * 2) / 3) + rand_unit() * 0.01]
            cfg.l_nb_cl = [Point2((cfg.l_collar.x + cfg.neck_b.x) / 2, (cfg.l_collar.y + cfg.neck_b.y * 2) / 3)]

        cfg.l_collar_o = cfg.l_collar + Point2(collar_width * np.cos(collar_theta), collar_width * np.sin(collar_theta))
        cfg.neck_f_o = cfg.neck_f + Point2(0., collar_width)
        cfg.neck_b_o = cfg.neck_b + Point2(0., collar_width)

        f_theta = np.arctan2((cfg.neck_f.x - cfg.l_collar.x), (cfg.l_collar.y - cfg.neck_f.y))
        b_theta = (np.arctan2((cfg.neck_b.x - cfg.l_collar.x), (cfg.l_collar.y - cfg.neck_b.y)) + collar_theta) / 2
        cfg.l_nf_cl_o = [x + Point2(collar_width * np.cos(f_theta), collar_width * np.sin(f_theta)) for x in cfg.l_nf_cl]
        cfg.l_nb_cl_o = [x + Point2(collar_width * np.cos(b_theta), collar_width * np.sin(b_theta)) for x in cfg.l_nb_cl]

        # parts other than collar
        TShirtCfgGenerator.shirt_like_cfg_randomize_shoulder_sleeve_body_3dargs(cfg, sleeve_length_str, dist_version=dist_version)

        # force symmetry
        cfg.symmetry()

        return cfg


class ShirtCfgGenerator(ShirtLikeCfgGenerator):
    @staticmethod
    def shirt_default(is_close: bool):
        if is_close:
            return ShirtCloseCfg()
        else:
            return ShirtCfg()
    
    @staticmethod
    def shirt_random(is_close: bool, sleeve_length_str: Literal["short", "long"]):
        cfg = ShirtCfg() if not is_close else ShirtCloseCfg()

        # collar
        cfg.l_collar = rand_xy((-0.08, -0.12), (0.28, 0.32))
        cfg.neck_b = Point2(0., cfg.l_collar.y + utils.map_01_ab(rand(), -0.04, +0.02))
        if is_close:
            cfg.l_neck_f = rand_xy((-0.004, -0.006), (0.18, 0.22))
            cfg.neck_c = Point2(0., cfg.l_neck_f.y)
        else:
            cfg.l_neck_f = rand_xy((-0.008, -0.012), (0.18, 0.22))
        cfg.l_nf_cl = [Point2((cfg.l_collar.x + cfg.l_neck_f.x) / 2, (cfg.l_collar.y + cfg.l_neck_f.y) / 2) + rand_unit() * 0.01]
        cfg.l_nb_cl = [Point2((cfg.l_collar.x + cfg.neck_b.x) / 2, (cfg.l_collar.y + cfg.neck_b.y * 2) / 3)]

        collar_width_1 = utils.map_01_ab(rand(), 0.01, 0.03)
        collar_width_2 = collar_width_1 * utils.map_01_ab(rand(), 1.0, 1.5)
        collar_width_avg = (collar_width_1 + collar_width_2) / 2
        collar_theta = np.radians(utils.map_01_ab(rand(), 40, 70))
        cfg.l_collar_o = cfg.l_collar + Point2(-collar_width_1, 0.)
        cfg.l_neck_f_o = cfg.l_neck_f + Point2(-collar_width_2 * np.cos(collar_theta), -collar_width_2 * np.sin(collar_theta))
        cfg.neck_b_o = cfg.neck_b + Point2(0., -collar_width_1)

        f_theta = (2 * np.arctan2((cfg.l_neck_f.x - cfg.l_collar.x), (cfg.l_collar.y - cfg.l_neck_f.y)) + collar_theta + 0.) / 2
        b_theta = (np.arctan2((cfg.neck_b.x - cfg.l_collar.x), (cfg.l_collar.y - cfg.neck_b.y)) + 0.) / 2
        cfg.l_nf_cl_o = [x + Point2(-collar_width_avg * np.cos(f_theta), -collar_width_avg * np.sin(f_theta)) for x in cfg.l_nf_cl]
        cfg.l_nb_cl_o = [x + Point2(-collar_width_1 * np.cos(b_theta), -collar_width_1 * np.sin(b_theta)) for x in cfg.l_nb_cl]

        ShirtLikeCfgGenerator.shirt_like_cfg_randomize_shoulder_sleeve_body_3dargs(cfg, sleeve_length_str)

        # force symmetry
        cfg.symmetry()

        return cfg


class TrousersCfgGenerator:
    @staticmethod
    def trousers_default(length_str: Literal["short", "long"]):
        if length_str == "long":
            return TrousersCfg()
        elif length_str == "short":
            return TrousersCfg(
                l_leg_o=(-0.26, -0.),
                r_leg_o=(+0.26, -0.),
                l_co_lo=[(-0.255, +0.2)],
                r_co_lo=[(+0.255, +0.2)],
                l_leg_i=(-0.05, -0.01),
                r_leg_i=(+0.05, -0.01),
                l_li_cr=[(-0.025, +0.13)],
                r_li_cr=[(+0.025, +0.13)],
            )
        else:
            raise ValueError(f"Invalid trousers length: {length_str}")
    
    @staticmethod
    def trousers_random(length_str: Literal["short", "long"]):
        cfg = TrousersCfg()

        # random cfg
        cfg.top_ctr_b = cfg.top_ctr_f = Point2(0., 0.50)
        cfg.l_corner = rand_xy((-0.16, -0.28), (0.49, 0.53))
        cfg.crotch = Point2(0., utils.map_01_ab(rand(), 0.25, 0.35))
        if length_str == "long":
            cfg.l_leg_o = cfg.l_corner + rand_xy((-0.04, -0.02), (-0.80, -1.00))
            l_leg_i_x, l_leg_i_y = cfg.l_leg_o + rand_xy((0.12, 0.20), (-0.02, -0.00))
        elif length_str == "short":
            cfg.l_leg_o = cfg.l_corner + rand_xy((-0.04, -0.02), (-0.40, -0.60))
            l_leg_i_x, l_leg_i_y = cfg.l_leg_o + rand_xy((0.18, 0.24), (-0.02, -0.00))
        else:
            raise NotImplementedError(length_str)
        l_leg_i_x = min(l_leg_i_x, -0.02)
        cfg.l_leg_i = Point2(l_leg_i_x, l_leg_i_y)

        cfg.l_co_lo = [(cfg.l_corner + cfg.l_leg_o) / 2 + rand_unit() * 0.02]
        l_li_cr_x, l_li_cr_y = (cfg.l_leg_i + cfg.crotch) / 2 + rand_unit() * 0.02
        l_li_cr_x = min(l_li_cr_x, -0.01)
        cfg.l_li_cr = [Point2(l_li_cr_x, l_li_cr_y)]

        # 3d args
        cfg.edge_max_z = utils.map_01_ab(rand(), 0.04, 0.06)
        cfg.edge_width = utils.map_01_ab(rand(), 0.08, 0.10)
        if length_str == "long":
            pass
        elif length_str == "short":
            for part_str in ["front", "back"]:
                cfg.interior_num[part_str] = int(cfg.interior_num[part_str] * 0.7)
        else:
            raise NotImplementedError(length_str)

        # force symmetry
        cfg.symmetry()

        return cfg


class VestCfgGenerator:
    @staticmethod
    def vest_default(is_close: bool):
        if is_close:
            return VestCloseCfg()
        else:
            return VestCfg()
    
    @staticmethod
    def vest_random(is_close: bool):
        cfg = VestCfg() if not is_close else VestCloseCfg()

        # collar
        cfg.l_collar = rand_xy((-0.10, -0.14), (0.28, 0.32))
        collar_width = utils.map_01_ab(rand(), 0.01, 0.02)
        collar_theta = np.radians(utils.map_01_ab(rand(), 0, 15))
        if is_close:
            cfg.neck_f = rand_xy((0., 0.), (0.08, 0.12))
            cfg.l_nf_cl = [Point2((cfg.l_collar.x + cfg.neck_f.x) / 2, (cfg.l_collar.y + cfg.neck_f.y * 2) / 3) + rand_unit() * 0.01]

            cfg.l_collar_o = cfg.l_collar + Point2(collar_width * np.cos(collar_theta), collar_width * np.sin(collar_theta))
            cfg.neck_f_o = cfg.neck_f + Point2(0., collar_width)

            f_theta = np.arctan2((cfg.neck_f.x - cfg.l_collar.x), (cfg.l_collar.y - cfg.neck_f.y))
            cfg.l_nf_cl_o = [x + Point2(collar_width * np.cos(f_theta), collar_width * np.sin(f_theta)) for x in cfg.l_nf_cl]
        else:
            cfg.l_neck_f = rand_xy((-0.008, -0.012), (0.08, 0.12))
            cfg.l_nf_cl = [Point2((cfg.l_collar.x + cfg.l_neck_f.x) / 2, (cfg.l_collar.y + cfg.l_neck_f.y * 2) / 3) + rand_unit() * 0.01]

            cfg.l_collar_o = cfg.l_collar + Point2(collar_width * np.cos(collar_theta), collar_width * np.sin(collar_theta))
            cfg.l_neck_f_o = cfg.l_neck_f + Point2(0., collar_width)

            f_theta = np.arctan2((cfg.l_neck_f.x - cfg.l_collar.x), (cfg.l_collar.y - cfg.l_neck_f.y))
            cfg.l_nf_cl_o = [x + Point2(collar_width * np.cos(f_theta), collar_width * np.sin(f_theta)) for x in cfg.l_nf_cl]
        
        cfg.neck_b = rand_xy((0., 0.), (0.25, 0.29))
        cfg.l_nb_cl = [Point2((cfg.l_collar.x + cfg.neck_b.x) / 2, (cfg.l_collar.y + cfg.neck_b.y * 2) / 3)]

        cfg.neck_b_o = cfg.neck_b + Point2(0., collar_width)
        b_theta = (np.arctan2((cfg.neck_b.x - cfg.l_collar.x), (cfg.l_collar.y - cfg.neck_b.y)) + collar_theta) / 2
        cfg.l_nb_cl_o = [x + Point2(collar_width * np.cos(b_theta), collar_width * np.sin(b_theta)) for x in cfg.l_nb_cl]

        # shoulder
        cfg.l_shoulder = cfg.l_collar + rand_xy((-0.06, -0.10), (-0.02, -0.04))
        cfg.l_armpit = cfg.l_shoulder + rand_xy((-0.04, -0.06), (-0.15, -0.19))
        cfg.l_sh_ar = [Point2((cfg.l_shoulder.x + cfg.l_armpit.x) / 2, (cfg.l_shoulder.y + cfg.l_armpit.y * 2) / 3) + rand_unit() * 0.01]

        # body
        cfg.l_corner = cfg.l_armpit + rand_xy((-0.02, +0.02), (-0.42, -0.52))
        cfg.spine_bottom_b = Point2(0., cfg.l_corner.y + utils.map_01_ab(rand(), -0.03, +0.03))

        if is_close:
            cfg.spine_bottom_f = Point2(0., cfg.spine_bottom_b.y)
        else:
            cfg.l_spine_bottom = Point2(cfg.l_neck_f.x, cfg.spine_bottom_b.y)
        
        # 3d args
        cfg.edge_max_z = utils.map_01_ab(rand(), 0.04, 0.06)
        cfg.edge_width = utils.map_01_ab(rand(), 0.08, 0.10)

        # force symmetry
        cfg.symmetry()

        return cfg


class HoodedCfgGenerator:
    @staticmethod
    def hooded_default(is_close: bool):
        if is_close:
            return HoodedCloseCfg()
        else:
            return HoodedCfg()
    
    @staticmethod
    def hooded_random(is_close: bool, sleeve_length_str: Literal["short", "long"]):
        cfg = HoodedCfg() if not is_close else HoodedCloseCfg()

        # collar
        cfg.l_collar = rand_xy((-0.06, -0.08), (0.28, 0.32))
        if is_close:
            cfg.neck_f = rand_xy((0., 0.), (0.25, 0.29))
            cfg.l_nf_cl = [Point2((cfg.l_collar.x + cfg.neck_f.x) / 2, (cfg.l_collar.y + cfg.neck_f.y * 2) / 3)]
        else:
            cfg.l_neck_f = rand_xy((-0.008, -0.012), (0.25, 0.29))
            cfg.l_nf_cl = [Point2((cfg.l_collar.x + cfg.l_neck_f.x) / 2, (cfg.l_collar.y + cfg.l_neck_f.y * 2) / 3)]
        cfg.neck_b = rand_xy((0., 0.), (0.23, 0.27))
        cfg.l_nb_cl = [Point2((cfg.l_collar.x + cfg.neck_b.x) / 2, (cfg.l_collar.y + cfg.neck_b.y * 2) / 3)]

        # hood
        cfg.hood_top = Point2(0., cfg.l_collar.y + 0.22 + utils.map_01_ab(rand(), -0.02, +0.02))
        p1 = Point2(cfg.l_collar.x * 1.0, (cfg.l_collar.y * 0.6 + cfg.hood_top.y * 0.4)) + rand_unit() * 0.01
        p2 = Point2((cfg.hood_top.x + p1.x * 2.5) / 3.5, (cfg.hood_top.y * 2 + p1.y) / 3) + rand_unit() * 0.01
        cfg.l_co_ht = [p1, p2]

        ShoulderSleeveBodyCfgGenerator.randomize_shoulder_sleeve_body(cfg, sleeve_length_str)

        # force symmetry
        cfg.symmetry()

        # 3d args
        cfg.edge_max_z = utils.map_01_ab(rand(), 0.08, 0.10)
        cfg.edge_width = utils.map_01_ab(rand(), 0.10, 0.12)
        if sleeve_length_str == "short":
            for part_str in ["leftf", "rightf", "leftb", "rightb"]:
                cfg.interior_num[part_str] = int(cfg.interior_num[part_str] * 0.5)
        elif sleeve_length_str == "long":
            pass
        else:
            raise NotImplementedError(sleeve_length_str)

        return cfg


def generate_cfg(
    category: available_category_type,
    method: Literal["default", "random"] = "default",
    description: str = "default",
    **kwargs, 
):
    if category in ["tshirt", "tshirt_sp"]:
        if method == "default":
            if description in ["default", "short_sleeve"]:
                return TShirtCfgGenerator.tshirt_default("short")
            elif description == "long_sleeve":
                return TShirtCfgGenerator.tshirt_default("long")
            else:
                raise NotImplementedError(description)
        elif method == "random":
            if description == "default":
                sleeve_length_str = ["short", "long"][np.random.randint(2)]
            elif description == "short_sleeve":
                sleeve_length_str = "short"
            elif description == "long_sleeve":
                sleeve_length_str = "long"
            else:
                raise NotImplementedError(description)
            return TShirtCfgGenerator.tshirt_random(sleeve_length_str, **kwargs)
        else:
            raise NotImplementedError(method)
    elif category == "trousers":
        if method == "default":
            if description in ["default", "long"]:
                return TrousersCfgGenerator.trousers_default("long")
            elif description == "short":
                return TrousersCfgGenerator.trousers_default("short")
            else:
                raise NotImplementedError(description)
        elif method == "random":
            if description == "default":
                trousers_length_str = ["long", "short"][np.random.randint(2)]
            elif description == "long":
                trousers_length_str = "long"
            elif description == "short":
                trousers_length_str = "short"
            else:
                raise NotImplementedError(description)
            return TrousersCfgGenerator.trousers_random(trousers_length_str)
        else:
            raise NotImplementedError(method)
    elif category in ["vest", "vest_sp"]:
        if method == "default":
            if description == "default":
                return VestCfgGenerator.vest_default(is_close=False)
            else:
                raise NotImplementedError(description)
        elif method == "random":
            if description == "default":
                return VestCfgGenerator.vest_random(is_close=False)
            else:
                raise NotImplementedError(description)
        else:
            raise NotImplementedError(method)
    elif category in ["vest_close", "vest_close_sp"]:
        if method == "default":
            if description == "default":
                return VestCfgGenerator.vest_default(is_close=True)
            else:
                raise NotImplementedError(description)
        elif method == "random":
            if description == "default":
                return VestCfgGenerator.vest_random(is_close=True)
            else:
                raise NotImplementedError(description)
        else:
            raise NotImplementedError(method)
    elif category in ["shirt"]:
        if method == "default":
            if description == "default":
                return ShirtCfgGenerator.shirt_default(is_close=False)
            else:
                raise NotImplementedError(description)
        elif method == "random":
            if description == "default":
                sleeve_length_str = ["short", "long"][np.random.randint(2)]
            elif description == "short_sleeve":
                sleeve_length_str = "short"
            elif description == "long_sleeve":
                sleeve_length_str = "long"
            else:
                raise NotImplementedError(description)
            return ShirtCfgGenerator.shirt_random(is_close=False, sleeve_length_str=sleeve_length_str)
        else:
            raise NotImplementedError(method)
    elif category in ["shirt_close"]:
        if method == "default":
            if description == "default":
                return ShirtCfgGenerator.shirt_default(is_close=True)
            else:
                raise NotImplementedError(description)
        elif method == "random":
            if description == "default":
                sleeve_length_str = ["short", "long"][np.random.randint(2)]
            elif description == "short_sleeve":
                sleeve_length_str = "short"
            elif description == "long_sleeve":
                sleeve_length_str = "long"
            else:
                raise NotImplementedError(description)
            return ShirtCfgGenerator.shirt_random(is_close=True, sleeve_length_str=sleeve_length_str)
        else:
            raise NotImplementedError(method)
    elif category in ["hooded"]:
        if method == "default":
            if description == "default":
                return HoodedCfgGenerator.hooded_default(is_close=False)
            else:
                raise NotImplementedError(description)
        elif method == "random":
            if description == "default":
                sleeve_length_str = ["short", "long"][np.random.randint(2)]
            elif description == "short_sleeve":
                sleeve_length_str = "short"
            elif description == "long_sleeve":
                sleeve_length_str = "long"
            else:
                raise NotImplementedError(description)
            return HoodedCfgGenerator.hooded_random(is_close=False, sleeve_length_str=sleeve_length_str)
        else:
            raise NotImplementedError(method)
    elif category in ["hooded_close"]:
        if method == "default":
            if description == "default":
                return HoodedCfgGenerator.hooded_default(is_close=True)
            else:
                raise NotImplementedError(description)
        elif method == "random":
            if description == "default":
                sleeve_length_str = ["short", "long"][np.random.randint(2)]
            elif description == "short_sleeve":
                sleeve_length_str = "short"
            elif description == "long_sleeve":
                sleeve_length_str = "long"
            else:
                raise NotImplementedError(description)
            return HoodedCfgGenerator.hooded_random(is_close=True, sleeve_length_str=sleeve_length_str)
        else:
            raise NotImplementedError(method)
    else:
        raise NotImplementedError(category)