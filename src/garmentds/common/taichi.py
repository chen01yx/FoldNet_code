import taichi as ti

import os
import inspect
from dataclasses import dataclass
import random

import numpy as np
import torch

import omegaconf


def init_taichi(taichi_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig):
    taichi_cfg = omegaconf.OmegaConf.to_container(taichi_cfg)
    taichi_cfg["arch"] = getattr(ti, taichi_cfg["arch"])

    if global_cfg.default_float == "float64":
        taichi_cfg["default_fp"] = ti.f64
    elif global_cfg.default_float == "float32":
        taichi_cfg["default_fp"] = ti.f32
    else:
        raise NotImplementedError(global_cfg.default_float)
    if global_cfg.default_int == "int64":
        taichi_cfg["default_ip"] = ti.i64
    elif global_cfg.default_int == "int32":
        taichi_cfg["default_ip"] = ti.i32
    else:
        raise NotImplementedError(global_cfg.default_int)
    
    if hasattr(global_cfg, "seed"):
        seed = global_cfg.seed
        taichi_cfg["random_seed"] = seed
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        random.seed(seed, version=2)
    
    ti.init(**taichi_cfg)


class TiFieldName:
    def __init__(self, prefix: str, filepath: str, lineno: int) -> None:
        self._prefix = prefix
        self._filepath = filepath
        self._lineno = lineno
        self._val = (prefix, filepath, lineno)

    def __hash__(self) -> int:
        return self._val.__hash__()
    
    def __eq__(self, __value: object) -> bool:
        assert isinstance(__value, TiFieldName)
        return self._val == __value._val
    
    def to_str(self, start_path=None) -> str:
        return f"{self._prefix} at file '{os.path.relpath(self._filepath, start_path)}', line {self._lineno}"


class TiFieldCreater:
    def __init__(self) -> None:
        self._result: dict[TiFieldName, list[tuple[tuple[int]]]] = {}

    @staticmethod
    def _get_name(prefix):
        return TiFieldName(prefix, inspect.currentframe().f_back.f_back.f_code.co_filename, inspect.currentframe().f_back.f_back.f_lineno)
    
    def _save_result(self, key_str, shape):
        if key_str not in self._result.keys():
            self._result[key_str] = []
        self._result[key_str].append(shape)
    
    def _modify_shape_kwargs(self, kwargs: dict):
        if not isinstance(kwargs["shape"], (list, tuple)):
            kwargs["shape"] = [kwargs["shape"]]
        kwargs["shape"] = list(kwargs["shape"])
        for i in range(len(kwargs["shape"])):
            if not isinstance(kwargs["shape"][i], int):
                kwargs["shape"][i] = int(kwargs["shape"][i])

    def ScalarField(self, dtype, *args, **kwargs) -> ti.ScalarField:
        if "shape" in kwargs.keys():
            self._modify_shape_kwargs(kwargs)
            self._save_result(self._get_name("ScalarField"), (tuple(kwargs["shape"]), ))
        return ti.field(dtype=dtype, *args, **kwargs)

    def VectorField(self, n, dtype, *args, **kwargs) -> ti.MatrixField:
        if "shape" in kwargs.keys():
            self._modify_shape_kwargs(kwargs)
            self._save_result(self._get_name("VectorField"), (tuple(kwargs["shape"]), (n, )))
        return ti.Vector.field(n=n, dtype=dtype, *args, **kwargs)

    def MatrixField(self, n, m, dtype, *args, **kwargs) -> ti.MatrixField:
        if "shape" in kwargs.keys():
            self._modify_shape_kwargs(kwargs)
            self._save_result(self._get_name("MatrixField"), (tuple(kwargs["shape"]), (n, m)))
        return ti.Matrix.field(n=n, m=m, dtype=dtype, *args, **kwargs)
    
    def StructField(self, cls, **kwargs) -> ti.StructField:
        if "shape" in kwargs.keys():
            if not isinstance(kwargs["shape"], (list, tuple)):
                kwargs["shape"] = (kwargs["shape"], )
            self._save_result(self._get_name("StructField"), (tuple(kwargs["shape"]), cls().get_shape()))
        return cls.field(**kwargs)
    
    def LogSparseField(self, shape):
        if not isinstance(shape, (list, tuple)):
            shape = (shape, )
        self._save_result(self._get_name("SparseField"), (tuple(shape), ))
    
    @property
    def result(self) -> dict:
        return self._result

    @staticmethod
    def _multiple(*args):
        ans = 1
        for x in args:
            ans *= x
        return ans
    
    @staticmethod
    def _calculate_size(field_name: TiFieldName, shape_list: tuple[tuple[int]]) -> int:
        if field_name._prefix == "SparseField":
            return 0
        
        size = 0
        for full_shape in shape_list:
            if len(full_shape) == 1:
                shape, = full_shape
                size += TiFieldCreater._multiple(*shape)
            else:
                shape, mn = full_shape
                size += TiFieldCreater._multiple(*shape) * TiFieldCreater._multiple(*mn)
        return size

    def get_report(self, start_path=None, key="size", reverse=True) -> str:
        @dataclass
        class T:
            name: str
            size: float
            cnt: int
            def __hash__(self) -> int:
                return (self.name, self.size, self.cnt).__hash__()
        
        total_size = 0
        str_dict = {}
        for k, v in self._result.items():
            t = T(k.to_str(start_path), TiFieldCreater._calculate_size(k, v), len(v))
            str_dict[t] = v
            total_size += t.size

        str_list = []
        total_field_cnt = 0
        for k in sorted(str_dict.keys(), key=lambda x: getattr(x, key), reverse=reverse):
            total_field_cnt += k.cnt
            str_list.append(f"{k.name}, {k.cnt}x, total size:{k.size} ~ {k.size / total_size * 100:.1f}%, {str_dict[k]}")
        return "\n".join([f"Total fields: {total_field_cnt}x"] + str_list)


GLOBAL_CREATER = TiFieldCreater()


@ti.dataclass
class Triplet:
    value: float
    row: int
    column: int

    def get_shape(self):
        return (3, )