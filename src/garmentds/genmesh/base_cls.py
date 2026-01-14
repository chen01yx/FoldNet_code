import abc
import copy
from dataclasses import dataclass, asdict
from typing import Callable, Optional, Literal, Any, Union
from collections import deque
import json
import os

import trimesh
import numpy as np


class Point2(tuple):
    def __new__(self, x: Union[float, tuple[float, float]], y: Optional[float] = None):
        """Point2(x, y) or Point2((x, y))"""
        if y is None: x, y = x
        return super(Point2, self).__new__(self, (float(x), float(y)))
    
    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]
    
    def __repr__(self) -> str:
        return f"Point2(x={self.x}, y={self.y})"
    
    def __add__(self, other: 'Point2'):
        return Point2((self.x + other.x), (self.y + other.y))
    
    def __truediv__(self, scalar: float):
        scalar = float(scalar)
        return Point2(self.x / scalar, self.y / scalar)
    
    def __mul__(self, scalar: float):
        scalar = float(scalar)
        return Point2(self.x * scalar, self.y * scalar)


@dataclass
class GarmentBoundaryEdge:
    start: str
    control: str
    end: str
    reverse: bool # reverse control point or not
    connect: bool # connect front layer and back layer or not, used to compute z
    include: Literal["start", "end", "both", "none"] = "start" # this property is used when e1 is 'connect' and e2 is 'not connect' and e1,e2 share a same vertex

    def __hash__(self) -> int:
        return tuple([x for x in self.__dict__.values()]).__hash__()
    
    def __eq__(self, value: object) -> bool:
        return self.__hash__() == value.__hash__()


@dataclass
class GarmentPart:
    is_back: bool
    is_collar: bool
    is_hood: bool


@dataclass
class VertInfo:
    part: GarmentPart
    part_name: str
    is_boundary: bool
    is_reuse: bool


@dataclass
class GarmentCfgABC(abc.ABC):
    @abc.abstractmethod
    def symmetry(self):
        pass

    @abc.abstractmethod
    def sanity_check(self):
        pass

    def asdict(self) -> dict:
        return asdict(self)


class GarmentTemplateABC(abc.ABC):
    def __init__(self, cfg: GarmentCfgABC) -> None:
        super().__init__()
        self._cfg = copy.deepcopy(cfg)
        self._mesh_cache = None

        self._ctrl_z_stack: deque[GarmentCfgABC] = deque(maxlen=128)
        self._ctrl_shift_z_stack: deque[GarmentCfgABC] = deque(maxlen=128)
        self._put_in_stack()

        self._cfg.sanity_check()

    @property
    def cfg(self):
        return self._cfg
    
    def _clear_cache(self):
        self._mesh_cache = None
    
    def _put_in_stack(self):
        print("[INFO] put current state in stack")
        self._ctrl_z_stack.append(copy.deepcopy(self._cfg))
        self._ctrl_shift_z_stack.clear()

    def update_keypoints(self, name: str, value: tuple[float, float], put_in_stack: bool):
        if hasattr(self._cfg, name):
            setattr(self._cfg, name, value)
        else:
            splited = name.split("_")
            idx = int(splited[-1])
            prefix = "_".join(splited[:-1])
            getattr(self._cfg, prefix)[idx] = value
        self._cfg.sanity_check()
        self._clear_cache()
        if put_in_stack: self._put_in_stack()
        
    def asdict_keypoints(self) -> dict[str, tuple[float, float]]:
        ans = {}
        for k, v in asdict(self._cfg).items():
            if isinstance(v, tuple):
                ans[k] = v
            elif isinstance(v, list):
                for idx, xy in enumerate(v):
                    ans[f"{k}_{idx}"] = xy
        return ans
    
    def access_keypoints(self, name: str):
        if hasattr(self._cfg, name):
            return getattr(self._cfg, name)
        else:
            splited = name.split("_")
            idx = int(splited[-1])
            prefix = "_".join(splited[:-1])
            return getattr(self._cfg, prefix)[idx]
    
    def ctrl_z(self):
        if len(self._ctrl_z_stack) > 1:
            new_cfg = self._ctrl_z_stack.pop()
            self._cfg = self._ctrl_z_stack.pop()
            self._ctrl_z_stack.append(copy.deepcopy(self._cfg))
            self._ctrl_shift_z_stack.append(copy.deepcopy(new_cfg))
            print("[INFO] successfully undo")
        else:
            print("[WARN] cannot undo")

    def ctrl_shift_z(self):
        if len(self._ctrl_shift_z_stack) > 0:
            self._cfg = self._ctrl_shift_z_stack.pop()
            self._ctrl_z_stack.append(copy.deepcopy(self._cfg))
            print("[INFO] successfully redo")
        else:
            print("[WARN] cannot redo")

    @abc.abstractmethod
    def draw(self, width: int, height: int, xy2ij: Callable[[float, float], tuple[int, int]]):
        pass

    @abc.abstractmethod
    def triangulation(self, *args, **kwargs):
        pass
    
    @staticmethod
    def check_bound(uv: np.ndarray, u0u1v0v1: list[float], info_str: str):
        u0, u1, v0, v1 = u0u1v0v1
        if uv[:, 0].min() < u0 or uv[:, 0].max() > u1 or uv[:, 1].min() < v0 or uv[:, 1].max() > v1:
            print(f"[ERROR] {info_str} uv generation out of boundary {u0u1v0v1}")
            return False
        return True

    def symmetry(self, put_in_stack: bool):
        self._cfg.symmetry()
        self._clear_cache()
        if put_in_stack:
            self._put_in_stack()
    
    def get_mesh_cache(self):
        return self._mesh_cache
    
    @abc.abstractmethod
    def get_info_to_export(self) -> dict:
        pass
    
    @abc.abstractmethod
    def quick_export(self, path: str):
        pass