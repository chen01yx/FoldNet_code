import os
import logging
from dataclasses import dataclass, fields
import copy
from typing import Union, Callable, Any
import datetime
import random
import time
from collections import defaultdict
import json

import numpy as np
import torch

import hydra
import omegaconf


def torch_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().copy()


def torch_dict_clone(d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in d.items()}


def torch_dict_to_numpy_dict(d: dict[str, Union[torch.Tensor, int, float, dict]]) -> dict[str, Union[np.ndarray, int, float, dict]]:
    ret = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            ret[k] = torch_to_numpy(v)
        elif isinstance(v, dict):
            ret[k] = torch_dict_to_numpy_dict(v)
        elif isinstance(v, (float, int)):
            ret[k] = v
        else:
            raise TypeError(type(v))
    return ret


def torch_dict_to_list_dict(d: dict[str, Union[torch.Tensor, int, float, dict]]) -> dict[str, Union[list, int, float, dict]]:
    ret = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            ret[k] = torch_to_numpy(v).tolist()
        elif isinstance(v, dict):
            ret[k] = torch_dict_to_list_dict(v)
        elif isinstance(v, (float, int)):
            ret[k] = v
        else:
            raise TypeError(type(v))
    return ret


def extract_single_batch(d: dict[str, Union[np.ndarray, dict, str, float, int]], batch_idx):
    """each array in ret is (1, ...)"""
    ret = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            ret[k] = v[[batch_idx], ...]
        elif isinstance(v, dict):
            ret[k] = extract_single_batch(v, batch_idx)
        elif isinstance(v, (str, float, int)):
            ret[k] = v
        else:
            raise TypeError(type(v))
    return ret


def merge_single_batch(d1: dict[str, Union[np.ndarray, dict, str, float, int]], d2: dict[str, Union[np.ndarray, dict, str, float, int]]):
    ret = {}
    for k, v1 in d1.items():
        v2 = d2[k]
        assert isinstance(v1, type(v2)), f"{type(v1)} {type(v2)}"
        if isinstance(v1, np.ndarray):
            ret[k] = np.concatenate([v1, v2], axis=0)
        elif isinstance(v1, dict):
            ret[k] = merge_single_batch(v1, v2)
        elif isinstance(v1, (str, float, int)):
            ret[k] = v1
        else:
            raise TypeError(type(v1))
    return ret


def get_folder_size(folder):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                size = os.path.getsize(fp)
                total_size += size
            except OSError:
                pass  # Ignore files that can't be accessed
    return total_size


_path_handler = hydra.utils.to_absolute_path


def set_path_handler(path_handler: Callable):
    assert callable(path_handler)
    global _path_handler
    _path_handler = path_handler


def default_path_handler():
    return hydra.utils.to_absolute_path


def get_path_handler():
    """
    to_absolute_path.
    
    If the input path is relative, it's interpreted as relative to the original working directory.
    """
    global _path_handler
    return _path_handler


def init_omegaconf():
    def func_load(s, v=""):
        d = omegaconf.OmegaConf.load(_path_handler(str(s)))
        x = eval(f"d{str(v)}")
        return x
    
    def func_mean(s):
        return sum(s) / len(s)
    
    def func_eval(s):
        result = eval(str(s))
        print(f"eval [{str(s)}] ... result is [{type(result)} {result}]")
        return result
    
    omegaconf.OmegaConf.clear_resolvers()
    omegaconf.OmegaConf.register_new_resolver("_load_", func_load)
    omegaconf.OmegaConf.register_new_resolver("_mean_", func_mean)
    omegaconf.OmegaConf.register_new_resolver("_eval_", func_eval)
    if not omegaconf.OmegaConf.has_resolver("now"):
        omegaconf.OmegaConf.register_new_resolver(
            "now",
            lambda pattern: datetime.datetime.now().strftime(pattern),
            use_cache=True,
            replace=True,
        )


def resolve_overwrite(cfg):
    overwrite_cfg = getattr(cfg, "overwrite", omegaconf.DictConfig({}))
    if overwrite_cfg is None:
        overwrite_cfg = omegaconf.DictConfig({})
    return omegaconf.OmegaConf.merge(cfg, overwrite_cfg)


def custom_serializer(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (omegaconf.ListConfig, omegaconf.DictConfig)):
        return omegaconf.OmegaConf.to_container(obj, resolve=True)
    else:
        raise TypeError(f"Type {type(obj)} not serializable")


def dump_json(path: str, data, default_serializer=custom_serializer):
    with open(path, "w") as f:
        json.dump(data, f, indent=4, default=default_serializer)


def load_json(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def dataclass_init_from_dict(dataclass_type: Any, data: dict[str, Any]) -> Any:
    fieldnames = {field.name: field for field in fields(dataclass_type)}
    init_kwargs = {}

    for key, value in data.items():
        if key in fieldnames:
            field = fieldnames[key]
            if hasattr(field.type, "__dataclass_fields__") and isinstance(value, dict):
                init_kwargs[key] = dataclass_init_from_dict(field.type, value)
            else:
                init_kwargs[key] = value

    return dataclass_type(**init_kwargs)


def seed_all(seed: int):
    """seed all random number generators except taichi"""
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


def map_01_ab(x, a, b):
    """`a + x * (b - a)` map a uniform distribution x in [0, 1] to [a, b]"""
    return a + x * (b - a)


def format_int(x: int, x_max: int):
    """
    Example:
    ```
    N = 100
    for x in range(N):
        print(format_int(x, N-1)) # 00, ..., 99
    ```
    """
    return str(x).zfill(len(str(x_max)))


def ddp_is_rank_0() -> bool:
    return int(os.environ.get('LOCAL_RANK', 0)) == 0 and int(os.environ.get('NODE_RANK', 0)) == 0


class Timer:
    def __init__(self, name="timer", logger: logging.Logger = None):
        self.name = str(name)
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        self.disable = False
        self.status = dict(total=defaultdict(float), count=defaultdict(int))

    def timer(self, func):
        def wrapper(*args, **kwargs):
            if not self.disable:
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                name, time_elapsed = func.__qualname__, (end_time - start_time) * 1e3
                self.status["total"][name] += time_elapsed
                self.status["count"][name] += 1
                total, count = self.status["total"][name], self.status["count"][name]
                self.logger.info(f"Timer {self.name}: {name} averaged {total / count:.0f} ms ({total:.0f} / {count}), current {time_elapsed:.0f} ms")
            else:
                result = func(*args, **kwargs)
            return result
        return wrapper
    
    def context_manager(self, name):
        class CM:
            def __init__(cm, name: str):
                cm.name = name
            
            def __enter__(cm):
                cm.start_time = time.time()
                return cm

            def __exit__(cm, exc_type, exc_val, exc_tb):
                cm.end_time = time.time()
                name, time_elapsed = cm.name, (cm.end_time - cm.start_time) * 1e3
                self.status["total"][name] += time_elapsed
                self.status["count"][name] += 1
                total, count = self.status["total"][name], self.status["count"][name]
                self.logger.info(f"Timer {self.name}: {name} averaged {total / count:.0f} ms ({total:.0f} / {count}), current {time_elapsed:.0f} ms")
        return CM(name)