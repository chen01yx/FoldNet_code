import logging
logger = logging.getLogger(__name__)

import json
import os
from dataclasses import dataclass, asdict
from multiprocessing import Manager
import multiprocessing as mp
from typing import Optional, Union, Literal, Callable
from collections import defaultdict, deque
import re
import time
import copy
import pprint
import shutil
import pickle

import omegaconf
import trimesh
import tqdm

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.v2 as transforms
from diffusers import DDPMScheduler

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

import garmentds
import garmentds.common.utils as utils
from garmentds.foldenv.fold_net import ConditionalUnet1D, Backbone, PositionalEncoding, MLP, reparametrize, kl_divergence

learn_timer = utils.Timer(name="fold_learn", logger=logger)
CACHE_DIR = os.path.join(garmentds.CACHE_BASE_DIR, "fold_learn")
os.makedirs(CACHE_DIR, exist_ok=True)


class TorchProfilerCallback(Callback):
    def __init__(self, profiler: torch.profiler.profile) -> None:
        super().__init__()
        self.profiler = profiler

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self.profiler.step()


def get_profiler(wait=1, warmup=1, active=2, repeat=1):
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile'),
        record_shapes=True,
        with_stack=True,
    )


def plot_wrap_fig(
    denses: list[np.ndarray], 
    titles: list[Union[str, list[str]]], 
    colorbars: list[Optional[str]], 
    plot_batch_size: int,
    width_unit=4.,
    height_unit=6.,
):
    plot_fig_num = len(denses)
    fig = plt.figure(figsize=(width_unit * plot_fig_num, height_unit * plot_batch_size))
    spec = fig.add_gridspec(
        nrows=plot_batch_size, 
        ncols=3 * plot_fig_num, 
        width_ratios=[6, 1, 2] * plot_fig_num, 
        height_ratios=[1] * plot_batch_size,
    )

    for batch_idx in range(plot_batch_size):
        for img_idx, img_b in enumerate(denses):
            img = img_b[batch_idx]
            ax = fig.add_subplot(spec[batch_idx, 3 * img_idx])
            if colorbars[img_idx] is not None:
                cmap = plt.get_cmap(colorbars[img_idx])
                ax.imshow(img, cmap=cmap)
            else:
                ax.imshow(img)
            if isinstance(titles[img_idx], str):
                ax.set_title(titles[img_idx])
            else:
                assert isinstance(titles[img_idx], list)
                assert isinstance(titles[img_idx][batch_idx], str)
                ax.set_title(titles[img_idx][batch_idx])
            if colorbars[img_idx] is not None:
                vmin, vmax = img.min(), img.max()
                cax = fig.add_subplot(spec[batch_idx, 3 * img_idx + 1])
                cbar = fig.colorbar(
                    mappable=ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=cmap), 
                    cax=cax,
                )

    return fig


def plot_wrap(denses: list[np.ndarray], tag: str, titles: list[Union[str, list[str]]], colorbars: list[Optional[str]],
              plot_batch_size: int, global_step: int, writer: SummaryWriter):
    """
    Args:
        - denses: List of [B, H, W]
        - tag: str
        - titles: List of str
    """
    assert len(denses) == len(titles) == len(colorbars), f"{len(denses)} {len(titles)} {len(colorbars)}"
    fig = plot_wrap_fig(denses, titles, colorbars, plot_batch_size)
    writer.add_figure(tag, fig, global_step=global_step)
    writer.close()
    plt.close()


def plot_hist_wrap(data: np.ndarray, tag: str, titles: list[str], global_step: int, writer: SummaryWriter):
    fig = plt.figure(figsize=(12, 6))
    spec = fig.add_gridspec(nrows=2, ncols=4)

    for i in range(8):
        ax = fig.add_subplot(spec[i // 4, i % 4])
        ax.hist(data[:, i])
        ax.set_title(titles[i])

    writer.add_figure(tag, fig, global_step=global_step)
    writer.close()
    plt.close()


@dataclass
class FoldData:
    action_rel: np.ndarray
    action_abs: np.ndarray
    action_is_correct: np.ndarray
    state: np.ndarray
    img_path: str
    cam_param: Optional[list[dict]]
    step_min: int
    step_max: int


A_DIM = S_DIM = 8


def load_json(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    return data


class TF:
    def __init__(self, height: int, width: int, random_rgb_permutation: bool):
        rgb_tfs = [
            transforms.Resize((height, width)), 
            transforms.ColorJitter(brightness=0.5, hue=0.3),
            transforms.Lambda(lambda x: torch.clamp(x, 0., 1.)),
        ]
        if random_rgb_permutation:
            rgb_tfs.append(transforms.Lambda(lambda x: x[:, np.random.permutation(3), :, :]))
        self.rgb = transforms.Compose(rgb_tfs)
        
        self.mask = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.Lambda(lambda x: torch.logical_xor(x != 0, torch.rand_like(x) < 0.02))
        ])
        self.tcp = transforms.Compose([transforms.Resize((height, width))])
    
    def tf(self, rgb: np.ndarray, mask: np.ndarray, tcp: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb = self.rgb(torch.tensor(rgb))
        mask = self.mask(torch.tensor(mask))
        tcp = self.tcp(torch.tensor(tcp))
        return rgb, mask, tcp


class FoldPolicyDataset(Dataset):
    def __init__(
        self, 
        data_list: list[FoldData], indices: np.ndarray, name: str, 
        height: int, width: int,
        obs_horizon: int, sta_horizon: int, actpred_len: int, use_tcp_mask: bool, 
        dtype: str, dtype_obs: str, allow_resample_when_getitem_error: bool,
        random_rgb_permutation: bool, 
    ):
        super().__init__()
        self.data_list = Manager().list(data_list) # use Manager() to avoid memory leak issue
        self.indices = indices.copy()
        self.name = str(name)
        self.height = int(height)
        self.width = int(width)
        self.obs_horizon = int(obs_horizon)
        self.sta_horizon = int(sta_horizon)
        self.actpred_len = int(actpred_len)
        self.use_tcp_mask = bool(use_tcp_mask)
        self.dtype = np.dtype(dtype)
        self.dtype_obs: torch.dtype = getattr(torch, dtype_obs)
        self.tf = TF(self.height, self.width, random_rgb_permutation)
        self.allow_resample_when_getitem_error = bool(allow_resample_when_getitem_error)

        print(f"fold policy dataset {self.name}: size {len(self)}")
    
    def __len__(self):
        return len(self.indices)
    
    @staticmethod
    def action_rel_dict_to_arr(json_data: dict):
        return np.array(json_data["xyz_l"] + json_data["xyz_r"] + [json_data["picker_l"], json_data["picker_r"]])

    @staticmethod
    def action_abs_dict_to_arr(json_data: dict):
        return np.concatenate([
            json_data["tcp_xyz"]["left"], json_data["tcp_xyz"]["right"], 
            [json_data["gripper_state"]["left"], json_data["gripper_state"]["right"]]
        ])
    
    @staticmethod
    def action_arr_to_dict(action_list: np.ndarray) -> dict[str, Union[np.ndarray, float]]:
        return dict(
            xyz_l = action_list[0:3], xyz_r = action_list[3:6],
            picker_l = action_list[6], picker_r = action_list[7],
        )
    
    @staticmethod
    def state_dict_to_arr(json_data: dict):
        return np.concatenate([
            json_data["tcp_xyz"]["left"], json_data["tcp_xyz"]["right"], 
            [json_data["gripper_state"]["left"], json_data["gripper_state"]["right"]]
        ])
    
    @staticmethod
    @learn_timer.timer
    def load_img_and_process(img_path: str, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
        img = np.load(img_path)
        rgb = img[0:3, :, :].astype(dtype) / 255.
        mask = (img[3:4, :, :] == 1).astype(dtype) # 1 is garment's mask
        return rgb, mask
    
    @staticmethod
    @learn_timer.timer
    def compute_tcp_mask(cam_param: dict, height: int, width: int, tcp_l: np.ndarray, tcp_r: np.ndarray, dtype: np.dtype) -> np.ndarray:
        xyzw_l, xyzw_r = np.array([*tcp_l, 1.]), np.array([*tcp_r, 1.])
        e4x4, i3x3 = np.array(cam_param["camera_extrinsics"]), np.array(cam_param["camera_intrinsics"])
        tcp_mask = np.zeros((height, width), dtype=dtype)
        model_matrix = np.diag([1., -1., -1., 1.]) # blender model matrix
        for lr, xyzw in enumerate([xyzw_l, xyzw_r]):
            xyzw_c = np.linalg.inv(e4x4 @ model_matrix) @ xyzw
            u, v, w = i3x3 @ (xyzw_c[:3])
            i, j = int(v / w), int(u / w)
            if 0 <= i < height and 0 <= j < width:
                tcp_mask[i, j] = {0: 1., 1: -1.}[lr]
        return tcp_mask[None, :, :]
    
    def __getitem__(self, idx):
        if self.allow_resample_when_getitem_error:
            while True:
                try:
                    return self._getitem_impl(idx)
                except Exception as e:
                    idx = np.random.randint(len(self))
        else:
            return self._getitem_impl(idx)

    @learn_timer.timer
    def _get_all_observation(self, data: FoldData, state_all: np.ndarray):
        rgb_all = []
        mask_all = []
        tcp_all = []
        for obs_idx in range(self.obs_horizon):
            def compute_idx(match): 
                return str(np.clip(int(match.group(1)) - obs_idx, data.step_min, data.step_max)).zfill(4)
            img_path = re.sub(r'(?<=/)(\d+)(?=\.npy$)', compute_idx, data.img_path)
            rgb, mask = self.load_img_and_process(img_path, self.dtype)  # [3, 480, 640], [1, 480, 640]
            if self.use_tcp_mask:
                tcp = self.compute_tcp_mask(
                    data.cam_param[obs_idx], rgb.shape[1], rgb.shape[2], 
                    state_all[obs_idx][0:3], state_all[obs_idx][3:6], self.dtype
                ) # [1, 480, 640]
            else:
                tcp = np.zeros_like(mask)
            rgb_all.append(rgb)
            mask_all.append(mask)
            tcp_all.append(tcp)

        rgb_all = np.array(rgb_all)
        mask_all = np.array(mask_all)
        tcp_all = np.array(tcp_all)
        rgb_all, mask_all, tcp_all = self.tf.tf(rgb_all, mask_all, tcp_all)
        assert rgb_all.shape == (self.obs_horizon, 3, self.height, self.width), rgb_all.shape
        assert mask_all.shape == (self.obs_horizon, 1, self.height, self.width), mask_all.shape
        assert tcp_all.shape == (self.obs_horizon, 1, self.height, self.width), tcp_all.shape
        return (
            rgb_all.to(dtype=self.dtype_obs), 
            mask_all.to(dtype=self.dtype_obs), 
            tcp_all.to(dtype=self.dtype_obs)
        )
    
    @learn_timer.timer
    def _getitem_impl(self, idx):
        data = self.data_list[self.indices[idx]]
        rgb_all, mask_all, tcp_all = self._get_all_observation(data, data.state)

        return dict(
            action_rel=data.action_rel.astype(self.dtype), 
            action_abs=data.action_abs.astype(self.dtype), 
            action_is_correct=data.action_is_correct.astype(self.dtype), 
            state=data.state.astype(self.dtype), 
            rgb=rgb_all, 
            mask=mask_all, 
            tcp=tcp_all, 
        )


def is_success_demo(meta_info_demo_json: str, err_th: float):
    meta_info_demo = load_json(meta_info_demo_json)
    err = meta_info_demo["err"] * 1000
    extra_info = dict(
        err=err, fail_reason=[], 
        path=os.path.relpath(meta_info_demo_json, utils.get_path_handler()(".")),
    )

    gt_dir = os.path.join(os.path.dirname(meta_info_demo_json), "..", "fold_gt")
    for dir, dirs, files in os.walk(gt_dir):
        for file in files:
            if file == "meta_info_gt.json":
                meta_info_gt_json = os.path.join(dir, file)
                meta_info_gt = load_json(meta_info_gt_json)
                if meta_info_gt["ik_fail_count"] > 0:
                    extra_info["fail_reason"].append("gt_ik_fail")
    
    if meta_info_demo["ik_fail_count"] > 0:
        extra_info["fail_reason"].append("run_ik_fail")
    
    if err > err_th:
        extra_info["fail_reason"].append("err_th")
    
    return len(extra_info["fail_reason"]) == 0, extra_info


def _quick_find(data_dirs: list[str], target_file_name: str, use_cache: bool):
    """if target_file_name exists, then skip searching in subdirectories."""
    version_str = "v1"
    tic = time.time()

    def get_cache_file(data_dir: str):
        return os.path.join(data_dir, ".index_cache.json")

    def read_cache_file(data_dir: str):
        cache_file = get_cache_file(data_dir)
        try:
            cache_data: dict = load_json(cache_file)
            assert cache_data.get("target_file_name", None) == target_file_name
            assert cache_data.get("version", None) == version_str
            return [os.path.join(data_dir, p) for p in cache_data["target_paths"]]
        except Exception as e:
            print(f"Exception: {e}")
        return None
    
    def write_cache_file(data_dir: str, target_path_in_this_dir: list[str]):
        cache_file = get_cache_file(data_dir)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(dict(
                version=version_str,
                target_file_name=target_file_name,
                target_paths=[os.path.relpath(p, data_dir) for p in target_path_in_this_dir],
            ), f, indent=4)
    
    def search_in_dir(data_dir: str, target_path_in_this_dir: list[str]):
        target_exists = False
        for f in os.listdir(data_dir):
            file_path = os.path.join(data_dir, f)
            if os.path.isfile(file_path) and f == target_file_name:
                target_path_in_this_dir.append(file_path)
                target_exists = True
                break
        if not target_exists:
            for d in os.listdir(data_dir):
                dir_path = os.path.join(data_dir, d)
                if os.path.isdir(dir_path):
                    search_in_dir(dir_path, target_path_in_this_dir)
    
    target_paths = []
    assert isinstance(data_dirs, list)
    for data_dir in data_dirs:
        if use_cache:
            target_path_in_this_dir = read_cache_file(data_dir)
        else:
            target_path_in_this_dir = None
        if target_path_in_this_dir is None:
            target_path_in_this_dir = []
            search_in_dir(data_dir, target_path_in_this_dir)
            write_cache_file(data_dir, target_path_in_this_dir)
        for p in target_path_in_this_dir:
            target_paths.append(p)
        
    target_paths = sorted(target_paths)

    toc = time.time()
    print(f"quick_find {target_file_name} in {toc - tic:.2f}s, found {len(target_paths)} files.")
    return target_paths


def _compute_statistics(all_action_rel_path_list: list[str], all_action_abs_path_list: list[str], statistics_sample: int):
    all_action_rel = []
    sample_idx = np.random.choice(len(all_action_rel_path_list), min(statistics_sample, len(all_action_rel_path_list)), replace=False)
    for idx in tqdm.tqdm(sample_idx):
        p = all_action_rel_path_list[idx]
        all_action_rel.append(FoldPolicyDataset.action_rel_dict_to_arr(load_json(p)))
    
    all_action_abs = []
    sample_idx = np.random.choice(len(all_action_abs_path_list), min(statistics_sample, len(all_action_abs_path_list)), replace=False)
    for idx in tqdm.tqdm(sample_idx):
        p = all_action_abs_path_list[idx]
        all_action_abs.append(FoldPolicyDataset.action_abs_dict_to_arr(load_json(p)))
    
    stat = dict(
        rel = dict(
            min=np.min(all_action_rel, axis=0),
            max=np.max(all_action_rel, axis=0),
            mean=np.mean(all_action_rel, axis=0),
            std=np.std(all_action_rel, axis=0),
        ),
        abs = dict(
            min=np.min(all_action_abs, axis=0),
            max=np.max(all_action_abs, axis=0),
            mean=np.mean(all_action_abs, axis=0),
            std=np.std(all_action_abs, axis=0),
        )
    )
    return stat


class _make_dataset_Pool:
    def __init__(
        self, num_workers: int, use_tcp_mask: bool, 
        actpred_len: int, sta_horizon: int, obs_horizon: int
    ):
        self.fold_data_list_train = Manager().list()
        self.fold_data_list_valid = Manager().list()
        self.all_action_rel_path = Manager().list()
        self.all_action_abs_path = Manager().list()
        self.num_workers = int(num_workers)
        self.use_tcp_mask = bool(use_tcp_mask)
        self.actpred_len = int(actpred_len)
        self.sta_horizon = int(sta_horizon)
        self.obs_horizon = int(obs_horizon)
        self.message_queue = mp.Queue(maxsize=self.num_workers * 2)
        self.correct_action_cnt = mp.Value("i", 0)
        self.wrong_action_cnt = mp.Value("i", 0)

    def worker(self, name: Literal["train", "valid"]):
        def load_json_cache(path: str, json_cache: dict):
            """return processed result if path exists, otherwise none"""
            if path in json_cache.keys():
                return json_cache[path]
            if os.path.exists(path):
                data = load_json(path)
            else:
                data = None
            json_cache[path] = data
            return data
        
        def create_folddata_from_path(
            action_rel_path: str, action_abs_path: str, state_path: str,
            img_path: str, cam_path: str, step_min: int, step_max: int, 
            actpred_len: int, sta_horizon: int, obs_horizon: int, 
            json_cache: dict, dtype: np.dtype = np.float32
        ):
            def get_all_action(action_rel_path: str, action_abs_path: str):
                action_rel = np.zeros((actpred_len, A_DIM), dtype=dtype)
                action_abs = np.zeros((actpred_len, A_DIM), dtype=dtype) # absolute action is equivalent to future state
                action_is_correct = np.ones((actpred_len, ), dtype=dtype)
                for action_idx in range(actpred_len):
                    def compute_idx(match): 
                        return str(int(match.group(1)) + action_idx)
                    p = re.sub(r'(?<=/)(\d+)(?=\.json$)', compute_idx, action_rel_path)
                    action_data = load_json_cache(p, json_cache)
                    if action_data is not None:  # if not exist, auto pad 0
                        action_rel[action_idx] = FoldPolicyDataset.action_rel_dict_to_arr(action_data)
                        action_is_correct[action_idx] = float(action_data.get("is_correct_action", True))

                for action_idx in range(actpred_len):
                    def compute_idx(match): 
                        return str(np.clip(int(match.group(1)) + action_idx, step_min, step_max))
                    p = re.sub(r'(?<=/)(\d+)(?=\.json$)', compute_idx, action_abs_path)
                    action_data = load_json_cache(p, json_cache)
                    if action_data is not None:
                        action_abs[action_idx] = FoldPolicyDataset.state_dict_to_arr(action_data)
                    else:
                        assert action_idx > 0, f"state {p} not found and action_idx is 0"
                        action_abs[action_idx] = action_abs[action_idx - 1]
                
                for action_idx in range(1, actpred_len):
                    # if previous action is wrong, current action is also wrong
                    action_is_correct[action_idx] = action_is_correct[action_idx] * action_is_correct[action_idx - 1]
                
                return action_rel, action_abs, action_is_correct
            
            def get_all_state(state_path: str):
                state_all = np.zeros((sta_horizon, S_DIM), dtype=dtype)
                for state_idx in range(sta_horizon):
                    def compute_idx(match): 
                        return str(np.clip(int(match.group(1)) - state_idx, step_min, step_max))
                    state_path = re.sub(r'(?<=/)(\d+)(?=\.json$)', compute_idx, state_path)
                    state_all[state_idx] = FoldPolicyDataset.state_dict_to_arr(load_json_cache(state_path, json_cache))
                return state_all

            def get_all_cam_param(cam_path: str):
                cam_param_all = []
                for obs_idx in range(obs_horizon):
                    def compute_idx(match): 
                        return str(np.clip(int(match.group(1)) - obs_idx, step_min, step_max)).zfill(4)
                    cam_path_i = re.sub(r'(?<=/)(\d+)(?=\.json$)', compute_idx, cam_path)
                    cam_param_all.append(load_json_cache(cam_path_i, json_cache))
                return cam_param_all
            
            action_rel, action_abs, action_is_correct = get_all_action(action_rel_path, action_abs_path)
            state = get_all_state(state_path)
            if self.use_tcp_mask:
                cam_param = get_all_cam_param(cam_path)
            else:
                cam_param = None
            return FoldData(action_rel, action_abs, action_is_correct, state, img_path, cam_param, step_min, step_max)

        while True:
            json_cache = dict()
            meta_info_path = self.message_queue.get()
            if meta_info_path is None:
                break
            meta_info = load_json(meta_info_path)
            start_step = meta_info["start_step"]
            end_step = meta_info["end_step"]
            base_dir = os.path.dirname(meta_info_path)
            for action_idx in range(start_step, end_step):
                action_rel_path = os.path.join(base_dir, f"action/{action_idx}.json")
                action_abs_path = os.path.join(base_dir, f"state/{action_idx + 1}.json")
                state_path = os.path.join(base_dir, f"state/{action_idx}.json")
                img_path = os.path.join(base_dir, f"head_rgb_mask/{str(action_idx).zfill(4)}.npy")
                cam_path = os.path.join(base_dir, f"head_cam_param/{str(action_idx).zfill(4)}.json")
                # when action_idx == end_step - 1, action_path does not exist, but we leave here to predict zero action.
                fold_data = create_folddata_from_path(
                    action_rel_path, action_abs_path, state_path,
                    img_path, cam_path, start_step, end_step - 2,
                    self.actpred_len, self.sta_horizon, self.obs_horizon, 
                    json_cache, 
                )
                self.correct_action_cnt.value += np.sum(fold_data.action_is_correct == 1.)
                self.wrong_action_cnt.value += np.sum(fold_data.action_is_correct == 0.)
                if name == "train":
                    self.fold_data_list_train.append(fold_data)
                elif name == "valid":
                    self.fold_data_list_valid.append(fold_data)
                else:
                    raise ValueError(name)
                if os.path.exists(action_rel_path): 
                    self.all_action_rel_path.append(action_rel_path)
                if os.path.exists(action_abs_path):
                    self.all_action_abs_path.append(action_abs_path)
    
    def worker_train(self):
        self.worker("train")
    
    def worker_valid(self):
        self.worker("valid")
        
    def run(self, train_list: list[str], valid_list: list[str]):
        for worker, args_list in zip(
            [self.worker_train, self.worker_valid],
            [train_list, valid_list],
        ):
            process_list: list[mp.Process] = []
            for p in range(self.num_workers):
                p = mp.Process(target=worker, daemon=True)
                p.start()
                process_list.append(p)
            for args in tqdm.tqdm(args_list):
                self.message_queue.put(args)
            for i in range(self.num_workers):
                self.message_queue.put(None)
            for p in process_list:
                p.join()
        return dict(
            correct=int(self.correct_action_cnt.value),
            wrong=int(self.wrong_action_cnt.value),
        )


def _make_data_list(
    data_dirs: list[str], 
    make_cfg: omegaconf.DictConfig, 
    dataset_cfg: omegaconf.DictConfig, 
    enable_quick_find_cache: bool,
):
    meta_info_path_list = []
    success_demo, total_demo, extra_info_all = 0, 0, []

    find_result = _quick_find(data_dirs, "meta_info_demo.json", enable_quick_find_cache)

    print(f"checking all trajectories success or not ...")
    for meta_info_demo_json in tqdm.tqdm(find_result):
        is_success, extra_info = is_success_demo(meta_info_demo_json, make_cfg.err_th)
        if is_success:
            meta_info_path_list.append(meta_info_demo_json)
            success_demo += 1
        total_demo += 1
        extra_info_all.append(extra_info)
    
    print(f"success rate = ({success_demo} / {total_demo} = {success_demo / total_demo:.2f})")
    if make_cfg.print_percentile and utils.ddp_is_rank_0():
        def print_percentile(data: list[float], tag: str):
            for p in [0, 10, 25, 50, 75, 90, 100]:
                v = np.percentile(data, p, method="nearest")
                print(f"{tag} percentile {p} = {v:.2f}, path = {extra_info_all[data.index(v)]['path']}")
        print_percentile([x["err"] for x in extra_info_all], "error")
    
    print("split train and valid trajectories ...")
    info = {"train": [], "valid": []}
    meta_info_path_list = sorted(meta_info_path_list)
    vr, tr = float(make_cfg.valid_ratio), make_cfg.train_ratio
    if tr is None: 
        tr = 1. - vr
    tr = float(tr)
    for meta_info_path in tqdm.tqdm(meta_info_path_list):
        rand_num = np.random.rand()
        assert vr + tr <= 1.
        if rand_num < vr:
            info["valid"].append(meta_info_path)
        elif rand_num < vr + tr:
            info["train"].append(meta_info_path)
    
    print(f"making data dict ...")
    pool = _make_dataset_Pool(
        make_cfg.num_workers, dataset_cfg.use_tcp_mask, 
        dataset_cfg.actpred_len, dataset_cfg.sta_horizon, dataset_cfg.obs_horizon,
    )
    correct_info = pool.run(info["train"], info["valid"])
    info.update(correct_info)
    fold_data_list_train: list[FoldData] = list(pool.fold_data_list_train)
    fold_data_list_valid: list[FoldData] = list(pool.fold_data_list_valid)
    all_action_rel_path_list = pool.all_action_rel_path
    all_action_abs_path_list = pool.all_action_abs_path

    print(f"compute statistics ...")
    stat = _compute_statistics(all_action_rel_path_list, all_action_abs_path_list, make_cfg.statistics_sample)

    return fold_data_list_train, fold_data_list_valid, stat, info


def make_dataset(
    data_dirs: list[str], 
    make_cfg: omegaconf.DictConfig, 
    dataset_cfg: omegaconf.DictConfig, 
    random_seed: int, 
    enable_quick_find_cache: bool,
    enable_pkl_data_cache: bool,  
):
    MAX_CACHE = 128
    def idx_to_cache_dir(i):
        return os.path.join(CACHE_DIR, "dataset", str(i % MAX_CACHE))
    
    def is_correct_cache_file(json_file: str):
        data: dict = load_json(json_file)
        return (
            data.get("data_dirs", []) == data_dirs and
            data.get("make_cfg", {}).get("valid_ratio", None) == make_cfg.valid_ratio and
            data.get("make_cfg", {}).get("train_ratio", None) == make_cfg.train_ratio and
            data.get("make_cfg", {}).get("err_th", None) == make_cfg.err_th and
            data.get("dataset_cfg", {}).get("obs_horizon", None) == dataset_cfg.obs_horizon and
            data.get("dataset_cfg", {}).get("sta_horizon", None) == dataset_cfg.sta_horizon and
            data.get("dataset_cfg", {}).get("actpred_len", None) == dataset_cfg.actpred_len and
            data.get("dataset_cfg", {}).get("use_tcp_mask", None) == dataset_cfg.use_tcp_mask and
            data.get("random_seed", None) == random_seed
        )

    def dump_cache_file(fold_data_list_train, fold_data_list_valid, stat, info):
        cache_dir = get_new_cache_dir()
        print(f"dump cache dir to {cache_dir}")
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, "meta.json"), "w") as f:
            json.dump(dict(
                data_dirs=data_dirs,
                make_cfg=omegaconf.OmegaConf.to_container(make_cfg),
                dataset_cfg=omegaconf.OmegaConf.to_container(dataset_cfg),
                random_seed=random_seed,
            ), f, indent=4)
        pickle_file = os.path.join(cache_dir, "data.pkl")
        with open(pickle_file, "wb") as f:
            pickle.dump(dict(
                fold_data_list_train=fold_data_list_train,
                fold_data_list_valid=fold_data_list_valid,
                stat=stat, info=info,
            ), f)
        print(f"dump successfully, file size: {os.path.getsize(pickle_file) / (2 ** 20):.2f} MB")
    
    def load_cache_file(pickle_file: str):
        print(f"load cache data from {pickle_file}")
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)
        fold_data_list_train = data["fold_data_list_train"]
        fold_data_list_valid = data["fold_data_list_valid"]
        stat, info = data["stat"], data["info"]
        print(f"load successfully, file size: {os.path.getsize(pickle_file) / (2 ** 20):.2f} MB")
        return fold_data_list_train, fold_data_list_valid, stat, info
    
    def find_correct_cache_dir():
        for i in range(MAX_CACHE):
            fn = os.path.join(idx_to_cache_dir(i), "meta.json")
            if os.path.exists(fn):
                if is_correct_cache_file(fn):
                    return idx_to_cache_dir(i)
        return None
    
    def get_new_cache_dir():
        for i in range(MAX_CACHE):
            if not os.path.exists(os.path.join(idx_to_cache_dir(i))):
                # remove the next slot
                if os.path.exists(os.path.join(idx_to_cache_dir((i + 1) % MAX_CACHE))):
                    shutil.rmtree(os.path.join(idx_to_cache_dir((i + 1) % MAX_CACHE)))
                # return the current dir
                return os.path.join(idx_to_cache_dir(i))
        raise ValueError(f"this is bug, should not happen, {CACHE_DIR}")
    
    d = find_correct_cache_dir()
    if (d is not None) and enable_pkl_data_cache:
        fold_data_list_train, fold_data_list_valid, stat, info = load_cache_file(os.path.join(d, "data.pkl"))
    else:
        fold_data_list_train, fold_data_list_valid, stat, info = _make_data_list(data_dirs, make_cfg, dataset_cfg, enable_quick_find_cache)
        dump_cache_file(fold_data_list_train, fold_data_list_valid, stat, info)
    
    print(f"correct action count: {info['correct']} wrong action count: {info['wrong']}")
    trds = FoldPolicyDataset(fold_data_list_train, np.random.permutation(len(fold_data_list_train)), "train", **dataset_cfg)
    vlds = FoldPolicyDataset(fold_data_list_valid, np.random.permutation(len(fold_data_list_valid)), "valid", **dataset_cfg)
    return trds, vlds, stat, info


class FoldPolicyModule(pl.LightningModule):
    ACTION_REPR_ABS = 0
    ACTION_REPR_REL = 1
    # ACTION_REPR_ABS_REL = 2
    def __init__(self, model_kwargs: dict, learn_kwargs: dict):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.model_kwargs = copy.deepcopy(model_kwargs)
        self.learn_kwargs = copy.deepcopy(learn_kwargs)

        self.obs_horizon = int(model_kwargs["obs_horizon"])
        self.sta_horizon = int(model_kwargs["sta_horizon"])
        self.actpred_len = int(model_kwargs["actpred_len"])
        self.height = int(model_kwargs["height"])
        self.width = int(model_kwargs["width"])
        self.dtype_str = str(model_kwargs["dtype"])

        self.xyz_unit = float(model_kwargs["xyz_unit"])
        self.normalize_method = str(model_kwargs["normalize_method"])
        self.action_repr_gripper = dict(
            absolute=self.ACTION_REPR_ABS, relative=self.ACTION_REPR_REL, # abs_rel=self.ACTION_REPR_ABS_REL
        )[model_kwargs["action_repr_gripper"]]
        self.action_repr_tcp_xyz = dict(
            absolute=self.ACTION_REPR_ABS, relative=self.ACTION_REPR_REL, # abs_rel=self.ACTION_REPR_ABS_REL
        )[model_kwargs["action_repr_tcp_xyz"]]
        self.use_masked_rgb = bool(model_kwargs["use_masked_rgb"])
        self.use_tcp_mask = bool(model_kwargs["use_tcp_mask"])
        self.weight_gripper = float(model_kwargs["weight_gripper"])
        self.weight_tcp_xyz = float(model_kwargs["weight_tcp_xyz"])
        self.register_buffer("stat_rel_min", torch.zeros(A_DIM).float())
        self.register_buffer("stat_rel_max", torch.zeros(A_DIM).float())
        self.register_buffer("stat_abs_min", torch.zeros(A_DIM).float())
        self.register_buffer("stat_abs_max", torch.zeros(A_DIM).float())

        self.model_name = str(model_kwargs["seq2seq"]["name"])
        self.token_dim_obs = int(model_kwargs["token_dim_obs"])
        self.token_dim_sta = int(model_kwargs["token_dim_sta"])
        self.sta_enc = MLP(
            input_dim=S_DIM, output_dim=self.token_dim_sta, 
            **model_kwargs["sta_enc"], last_no_activate=True,
        )
        self.obs_enc = Backbone(
            self.token_dim_obs, self.height, self.width, 
            dict(act="bchw", mlp="feat", dp="feat")[self.model_name], **model_kwargs["obs_enc"]
        )

        sub_cfg = model_kwargs["seq2seq"][self.model_name]
        if self.model_name == "act":
            assert self.token_dim_obs == self.token_dim_sta
            self.token_dim = self.token_dim_act = self.token_dim_obs
            self.pe = PositionalEncoding(d_model = self.token_dim)
            self.transformer = nn.Transformer(
                d_model=self.token_dim, batch_first=True, 
                **sub_cfg["transformer"]
            )
            self.action_dec = MLP(
                input_dim=self.token_dim, output_dim=A_DIM, 
                **sub_cfg["action_dec"], last_no_activate=True, 
            )
            self.loss_func = dict(l1_loss=nn.L1Loss(reduction="none"), mse_loss=nn.MSELoss(reduction="none"))[learn_kwargs["xyz_loss"]]

            self.use_cvae = sub_cfg["cvae"]["use_cvae"]
            if self.use_cvae:
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=self.token_dim, batch_first=True, 
                    **{k: v for k, v in sub_cfg["cvae"]["enc"].items() if k!= "num_layers"}
                )
                self.cls_embed = nn.Embedding(1, self.token_dim)
                self.cvae_act_proj = nn.Linear(A_DIM, self.token_dim)
                self.cvae_sta_proj = nn.Linear(S_DIM, self.token_dim)
                self.cvae_enc = nn.TransformerEncoder(enc_layer, sub_cfg["cvae"]["enc"]["num_layers"])
                self.cvae_latent_dim = sub_cfg["cvae"]["latent_dim"]
                self.cvae_latent_proj = nn.Linear(self.token_dim, self.cvae_latent_dim * 2)
                self.cvae_latent_out_proj = nn.Linear(self.cvae_latent_dim, self.token_dim)
                self.kl_weight = sub_cfg["cvae"]["kl_weight"]
        elif self.model_name == "mlp":
            in_d = self.token_dim_obs * self.obs_horizon + self.token_dim_sta * self.sta_horizon
            self.token_dim_act = sub_cfg["token_dim_act"]
            self.mlp = MLP(
                input_dim=in_d, output_dim=self.token_dim_act * self.actpred_len, 
                **sub_cfg["mlp"], last_no_activate=False
            )
            self.action_dec = MLP(
                input_dim=self.token_dim_act, output_dim=A_DIM, 
                **sub_cfg["action_dec"], last_no_activate=True, 
            )
            self.loss_func = dict(l1_loss=nn.L1Loss(reduction="none"), mse_loss=nn.MSELoss(reduction="none"))[learn_kwargs["xyz_loss"]]
        elif self.model_name == "dp":
            gd = self.token_dim_obs * self.obs_horizon + self.token_dim_sta * self.sta_horizon
            self.cond_unet = ConditionalUnet1D(input_dim=A_DIM, global_cond_dim=gd, **sub_cfg["unet"])
            self.ddpm = DDPMScheduler(**sub_cfg["ddpm"])
            self.loss_func = nn.MSELoss(reduction="none")
        else:
            raise ValueError(self.model_name)
            
        self.accumulate_grad_batches = int(learn_kwargs["accumulate_grad_batches"])

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.learn_kwargs["optimizer"]["name"])(
            params=self.parameters(),
            **self.learn_kwargs["optimizer"]["cfg"]
        )
        if self.model_name != "act":
            scheduler = getattr(torch.optim.lr_scheduler, self.learn_kwargs["scheduler"]["name"])(
                optimizer, **(self.learn_kwargs["scheduler"]["cfg"]),
            )
        else:
            warmup_steps = self.learn_kwargs["scheduler"]["warmup_steps"]
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer=optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(optimizer, 1. / warmup_steps, 1., total_iters=warmup_steps),
                    torch.optim.lr_scheduler.MultiStepLR(optimizer, []),
                ],
                milestones=[warmup_steps],
            )
        return [optimizer], [scheduler]
    
    def set_statistics(self, statistics: dict[str, np.ndarray]):
        self.stat_rel_min[...] = torch.from_numpy(statistics["rel"]["min"])
        self.stat_rel_max[...] = torch.from_numpy(statistics["rel"]["max"])
        self.stat_abs_min[...] = torch.from_numpy(statistics["abs"]["min"])
        self.stat_abs_max[...] = torch.from_numpy(statistics["abs"]["max"])
    
    def get_scale_offset(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        for k in ["stat_rel_min", "stat_rel_max", "stat_abs_min", "stat_abs_max"]:
            getattr(self, k).to(device=device)
        if self.action_repr_tcp_xyz == self.ACTION_REPR_ABS:
            action_min_t, action_max_t = self.stat_abs_min[0:6], self.stat_abs_max[0:6]
        elif self.action_repr_tcp_xyz == self.ACTION_REPR_REL:
            action_min_t, action_max_t = self.stat_rel_min[0:6], self.stat_rel_max[0:6]
        else:
            raise ValueError(self.action_repr_tcp_xyz)
        if self.action_repr_gripper == self.ACTION_REPR_ABS:
            action_min_g, action_max_g = self.stat_abs_min[6:8], self.stat_abs_max[6:8]
        elif self.action_repr_gripper == self.ACTION_REPR_REL:
            action_min_g, action_max_g = self.stat_rel_min[6:8], self.stat_rel_max[6:8]
        else:
            raise ValueError(self.action_repr_gripper)
        action_min = torch.cat([action_min_t, action_min_g])
        action_max = torch.cat([action_max_t, action_max_g])
        
        scale = (action_max - action_min) / 2
        offset = (action_min + action_max) / 2
        return scale, offset

    def normalize(self, action: torch.Tensor) -> torch.Tensor:
        scale, offset = self.get_scale_offset(action.device)
        if self.normalize_method == "const":
            unit = torch.tensor([self.xyz_unit] * 6 + [1.0] * 2, device=action.device, dtype=action.dtype)
            return action / unit
        elif self.normalize_method == "min_max":
            return (action - offset) / scale
        else:
            raise ValueError(self.normalize_method)
 
    def denormalize(self, action: torch.Tensor) -> torch.Tensor:
        scale, offset = self.get_scale_offset(action.device)
        if self.normalize_method == "const":
            unit = torch.tensor([self.xyz_unit] * 6 + [1.0] * 2, device=action.device, dtype=action.dtype)
            return action * unit
        elif self.normalize_method == "min_max":
            return action * scale + offset
        else:
            raise ValueError(self.normalize_method)
    
    def compute_action_gt(self, action_rel: torch.Tensor, action_abs: torch.Tensor, state: torch.Tensor):
        action_gt = torch.zeros_like(action_rel)
        if self.action_repr_tcp_xyz == self.ACTION_REPR_ABS:
            action_gt[..., 0:6] = action_abs[..., 0:6]
        elif self.action_repr_tcp_xyz == self.ACTION_REPR_REL:
            action_gt[..., 0:6] = action_rel[..., 0:6]
        else:
            raise ValueError(self.action_repr_tcp_xyz)
        if self.action_repr_gripper == self.ACTION_REPR_ABS:
            action_gt[..., 6:8] = action_abs[..., 6:8]
        elif self.action_repr_gripper == self.ACTION_REPR_REL:
            action_gt[..., 6:8] = action_rel[..., 6:8]
        else:
            raise ValueError(self.action_repr_gripper)
        return action_gt

    def compute_action_pred(self, batch: dict[str, torch.Tensor], mode: Literal["train", "infer"], **kwargs):
        """Compute predicted action. Note that the action may be relative or absolute depending on the model."""
        state, rgb, mask, tcp = batch["state"], batch["rgb"], batch["mask"], batch["tcp"]
        B, TO, C, H, W = rgb.shape
        TA = self.actpred_len
        assert state.shape == (B, self.sta_horizon, S_DIM), state.shape
        assert rgb.shape == (B, self.obs_horizon, 3, H, W), rgb.shape
        assert mask.shape == (B, self.obs_horizon, 1, H, W), mask.shape
        assert tcp.shape == (B, self.obs_horizon, 1, H, W), tcp.shape
        
        if self.use_masked_rgb:
            obs = rgb * mask
        else:
            obs = rgb
        if self.use_tcp_mask:
            obs = torch.concat([obs, torch.sign(tcp)], dim=2)
        device, dtype, CO = obs.device, obs.dtype, obs.shape[2]

        detail_info = dict()
        if mode == "train":
            action_gt = self.compute_action_gt(batch["action_rel"], batch["action_abs"], batch["state"]) # [B, TA, DA]
            action_gt_norm = self.normalize(action_gt)
            assert action_gt.shape == (B, TA, A_DIM), action_gt.shape
            detail_info["action_gt"] = action_gt
            detail_info["action_gt_norm"] = action_gt_norm
        if self.model_name == "act":
            obs_token: torch.Tensor = self.obs_enc(obs.view(-1, CO, H, W)) # [B*TO, D, H_, W_]
            _, D, H_, W_ = obs_token.shape
            assert H_ == self.obs_enc.height_feat and W_ == self.obs_enc.width_feat
            obs_token = obs_token.view(B * TO, self.token_dim, H_ * W_).transpose(1, 2).reshape(B, TO * H_ * W_, self.token_dim)
            sta_token: torch.Tensor = self.sta_enc(state.view(-1, S_DIM)) # [B*TO, D]
            sta_token = sta_token.view(B, TO, self.token_dim)
            if not self.use_cvae:
                src = torch.concat([obs_token, sta_token], dim=1)
            else:
                if mode == "train":
                    cvae_cls_token = torch.unsqueeze(self.cls_embed.weight, axis=0).repeat(B, 1, 1) # [B, 1, D]
                    cvae_act_token = self.cvae_act_proj(action_gt_norm.view(-1, A_DIM)).view(B, TA, self.token_dim) # [B, TA, D]
                    cvae_sta_token = self.cvae_sta_proj(state[:, 0, :]).view(B, 1, self.token_dim) # [B, 1, D]
                    cvae_input = torch.concat([cvae_cls_token, cvae_sta_token, cvae_act_token], dim=1)
                    cvae_enc_out = self.cvae_enc(self.pe(cvae_input)) # [B, TA+2, D]
                    cvae_latent_token = cvae_enc_out[:, 0, :] # [B, D]
                    cvae_latent_info = self.cvae_latent_proj(cvae_latent_token) # [B, DL * 2]
                    mu = cvae_latent_info[:, 0:self.cvae_latent_dim]
                    logvar = cvae_latent_info[:, self.cvae_latent_dim:self.cvae_latent_dim * 2]
                    cvae_latent_sample = reparametrize(mu, logvar)
                    assert cvae_latent_sample.shape == (B, self.cvae_latent_dim), cvae_latent_sample.shape
                    detail_info["mu"] = mu
                    detail_info["logvar"] = logvar
                else:
                    cvae_latent_sample = torch.zeros((B, self.cvae_latent_dim), device=device, dtype=dtype)
                cvae_latent_out_token = self.cvae_latent_out_proj(cvae_latent_sample).view(B, 1, self.token_dim)
                src = torch.concat([obs_token, sta_token, cvae_latent_out_token], dim=1)
            tgt = torch.zeros((B, TA, self.token_dim), device=device, dtype=dtype)
            act_token: torch.Tensor = self.transformer(self.pe(src), self.pe(tgt)) # [B, TA, D]
            act_pred_norm: torch.Tensor = self.action_dec(act_token.view(-1, self.token_dim_act)) # [B*TA, DA]
            act_pred_norm = act_pred_norm.view(B, TA, A_DIM)
        elif self.model_name == "mlp":
            obs_token = self.obs_enc(obs.view(-1, CO, H, W))
            obs_token = obs_token.view(B, TO * self.token_dim_obs)
            sta_token: torch.Tensor = self.sta_enc(state.view(-1, S_DIM)) # [B*TO, D]
            sta_token = sta_token.view(B, TO * self.token_dim_sta)
            act_token = self.mlp(F.relu(torch.concat([obs_token, sta_token], dim=1))) # [B, TA*D]
            act_pred_norm: torch.Tensor = self.action_dec(act_token.view(-1, self.token_dim_act)) # [B*TA, DA]
            act_pred_norm = act_pred_norm.view(B, TA, A_DIM)
        elif self.model_name == "dp":
            obs_token: torch.Tensor = self.obs_enc(obs.view(-1, CO, H, W)) # B*L, D
            sta_token: torch.Tensor = self.sta_enc(state.view(-1, A_DIM)) # B*L, D
            global_cond = torch.concat([obs_token.view(B, -1), sta_token.view(B, -1)], dim=1)
            if mode == "train":
                timesteps = torch.randint(0, self.ddpm.config.num_train_timesteps, (B, ), device=device, dtype=torch.long)
                noise = torch.randn_like(action_gt_norm, device=device)
                act_noised = self.ddpm.add_noise(action_gt_norm, noise, timesteps)
                noise_pred = self.cond_unet(sample=act_noised, timestep=timesteps, global_cond=global_cond)
                act_pred_norm = act_noised - noise_pred # not used for bp
                detail_info["noise"] = noise
                detail_info["noise_pred"] = noise_pred
            elif mode == "infer":
                act_pred_norm = torch.randn((B, TA, A_DIM), device=device, dtype=dtype)
                self.ddpm.set_timesteps(kwargs["ddpm_inference_timestep"])
                with torch.no_grad():
                    for t in self.ddpm.timesteps:
                        model_output = self.cond_unet(sample=act_pred_norm, timestep=t, global_cond=global_cond)
                        act_pred_norm = self.ddpm.step(model_output, t, act_pred_norm).prev_sample
            else:
                raise ValueError(mode)
        else:
            raise ValueError(self.model_name)
        
        act_pred = self.denormalize(act_pred_norm)
        detail_info["rgb"] = rgb
        detail_info["obs"] = obs
        detail_info["action_pred"] = act_pred
        detail_info["action_pred_norm"] = act_pred_norm
        return act_pred, detail_info
    
    def convert_pred_action_seq_to_abs_action_seq(self, action_seq: torch.Tensor, state: torch.Tensor):
        assert action_seq.shape[1:] == (self.actpred_len, A_DIM) and state.shape[1:] == (self.sta_horizon, S_DIM), f"{action_seq.shape}, {state.shape}"
        if self.action_repr_tcp_xyz == self.ACTION_REPR_ABS:
            xyz = action_seq[:, :, 0:6]
        elif self.action_repr_tcp_xyz == self.ACTION_REPR_REL:
            xyz = state[:, [0], 0:6] + torch.cumsum(action_seq[:, :, 0:6], dim=1)
        else:
            raise ValueError(self.action_repr_tcp_xyz)
        if self.action_repr_gripper == self.ACTION_REPR_ABS:
            gripper = action_seq[:, :, 6:8]
        elif self.action_repr_gripper == self.ACTION_REPR_REL:
            gripper = state[:, [0], 6:8] + torch.cumsum(action_seq[:, :, 6:8], dim=1)
        else:
            raise ValueError(self.action_repr_gripper)
        return torch.concat([xyz, gripper], dim=-1)
    
    def forward_all(self, batch: dict[str, torch.Tensor]):
        # forward net
        action_pred, detail_info = self.compute_action_pred(batch, mode="train")
        result_mask = batch["action_is_correct"][:, :, None].repeat(1, 1, A_DIM)

        # transform action and prepare err and loss's input
        ap, at = detail_info["action_pred_norm"], detail_info["action_gt_norm"]
        lf = self.loss_func
        if self.model_name in ["act", "mlp"]:
            pred, gt = ap, at # [B, TA, DA]
            err = torch.mean(torch.abs(
                (detail_info["action_pred"] - detail_info["action_gt"]) * result_mask
            ), dim=(0, 1))
        elif self.model_name == "dp":
            pred, gt = detail_info["noise_pred"], detail_info["noise"] # [B, TA, DA]
            err = torch.mean(torch.abs(
                (self.denormalize(pred) - self.denormalize(gt)) * result_mask
            ), dim=(0, 1))
        else:
            raise ValueError(self.model_name)

        # compute loss
        loss_xyz_l = (lf(pred[..., 0:3], gt[..., 0:3]) * result_mask[..., 0:3]).mean()
        loss_xyz_r = (lf(pred[..., 3:6], gt[..., 3:6]) * result_mask[..., 3:6]).mean()
        loss_pl = (lf(pred[..., 6], gt[..., 6]) * result_mask[..., 6]).mean()
        loss_pr = (lf(pred[..., 7], gt[..., 7]) * result_mask[..., 7]).mean()
        total_loss = (loss_xyz_l + loss_xyz_r) * self.weight_tcp_xyz + (loss_pl + loss_pr) * self.weight_gripper
        loss_dict = dict(xyz_l=loss_xyz_l, xyz_r=loss_xyz_r, p_l=loss_pl, p_r=loss_pr)

        if self.model_name == "act" and self.use_cvae:
            kl_loss = torch.mean(kl_divergence(detail_info["mu"], detail_info["logvar"]))
            total_loss = total_loss + self.kl_weight * kl_loss
            loss_dict["kl"] = kl_loss

        # prepare result
        info_dict = dict(
            err_xl = err[0], err_yl = err[1], err_zl = err[2],
            err_xr = err[3], err_yr = err[4], err_zr = err[5],
            err_pl = err[6], err_pr = err[7], err = err.sum()
        )

        return total_loss, loss_dict, info_dict, detail_info

    def log_all(self, total_loss: torch.Tensor, loss_dict: dict[str, torch.Tensor], info_dict: dict[str, torch.Tensor], name: str):
        self.log(f"{name}_loss/_total_loss", total_loss.detach().clone(), sync_dist=True)
        for k, v in loss_dict.items():
            self.log(f"{name}_loss/{k}", v.detach().clone(), sync_dist=True)
        for k, v in info_dict.items():
            self.log(f"{name}_info/{k}", v.detach().clone(), sync_dist=True)
    
    @staticmethod
    def add_marker(img: np.ndarray, mask: np.ndarray):
        u, v = np.where(mask > 0.)
        for di in range(-2, 3, 1):
            for dj in range(-2, 3, 1):
                img[np.clip(u + di, 0, mask.shape[0] - 1), np.clip(v + dj, 0, mask.shape[1] - 1)] = [1., 0., 0.]
        u, v = np.where(mask < 0.)
        for di in range(-2, 3, 1):
            for dj in range(-2, 3, 1):
                img[np.clip(u + di, 0, mask.shape[0] - 1), np.clip(v + dj, 0, mask.shape[1] - 1)] = [0., 1., 0.]
        return img
    
    def log_img(self, batch_idx: int, detail_info: dict[str, torch.Tensor]):
        plot_num = min(detail_info["action_gt"].shape[0], self.learn_kwargs["valid"]["plot_num_per_batch"])
        obs1_idx = min(self.obs_horizon - 1, 1)

        obs0_str = []
        for i in range(plot_num):
            obs0_str.append(
                "obs0"
                + "\ngt: " + " ".join([f"{x:.3f}" for x in detail_info["action_gt"][i, 0, 6:8].tolist()])
                + "\n" + " ".join([f"{x:.3f}" for x in detail_info["action_gt"][i, 0, 0:6].tolist()])
                + "\npred: " + " ".join([f"{x:.3f}" for x in detail_info["action_pred"][i, 0, 6:8].tolist()])
                + "\n" + " ".join([f"{x:.3f}" for x in detail_info["action_pred"][i, 0, 0:6].tolist()])
            )
        
        denses=[
            utils.torch_to_numpy(detail_info["obs"][:, 0, 0:3, :, :].permute(0, 2, 3, 1)),
            utils.torch_to_numpy(detail_info["rgb"][:, 0, :, :, :].permute(0, 2, 3, 1)),
            utils.torch_to_numpy(detail_info["obs"][:, obs1_idx, 0:3, :, :].permute(0, 2, 3, 1)),
            utils.torch_to_numpy(detail_info["rgb"][:, obs1_idx, :, :, :].permute(0, 2, 3, 1)),
        ]
        colorbars = [None] * 4
        titles=[obs0_str, "rgb0", f"obs{obs1_idx}", f"rgb{obs1_idx}"]
        if self.use_tcp_mask:
            tcp0 = utils.torch_to_numpy(detail_info["obs"][:, 0, 3, :, :])
            tcp1 = utils.torch_to_numpy(detail_info["obs"][:, obs1_idx, 3, :, :])
            for i in range(plot_num):
                denses[1][i] = self.add_marker(denses[1][i], tcp0[i])
                denses[3][i] = self.add_marker(denses[3][i], tcp1[i])

        plot_wrap(
            denses=denses, 
            tag=f"img/{batch_idx}",
            titles=titles,
            colorbars=colorbars,
            plot_batch_size=plot_num,
            global_step=self.global_step,
            writer=self.logger.experiment,
        )

        for s in ["action_gt", "action_pred", "action_gt_norm", "action_pred_norm"]:
            for i in [0, self.actpred_len // 2]:
                plot_hist_wrap(
                    data=utils.torch_to_numpy(detail_info[s])[:, i, [0, 1, 2, 6, 3, 4, 5, 7]], 
                    tag=f"{s}/{i}", 
                    titles=["lx", "ly", "lz", "lp", "rx", "ry", "rz", "rp"],
                    global_step=self.global_step, 
                    writer=self.logger.experiment
                )
    
    def log_img_eval(self, detail_info: dict[str, torch.Tensor], pdf_path: str):
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        obs1_idx = min(self.obs_horizon - 1, 1)
        obs0_str = (
            "obs0"
            + "\npred: " + " ".join([f"{x:.3f}" for x in detail_info["action_pred"][0, 0, 6:8].tolist()])
            + "\n" + " ".join([f"{x:.3f}" for x in detail_info["action_pred"][0, 0, 0:6].tolist()])
        )

        denses=[
            utils.torch_to_numpy(detail_info["obs"][:, 0, 0:3, :, :].permute(0, 2, 3, 1)),
            utils.torch_to_numpy(detail_info["rgb"][:, 0, :, :, :].permute(0, 2, 3, 1)),
            utils.torch_to_numpy(detail_info["obs"][:, obs1_idx, 0:3, :, :].permute(0, 2, 3, 1)),
            utils.torch_to_numpy(detail_info["rgb"][:, obs1_idx, :, :, :].permute(0, 2, 3, 1)),
        ]
        colorbars = [None] * 4
        titles=[obs0_str, "rgb0", f"obs{obs1_idx}", f"rgb{obs1_idx}"]
        if self.use_tcp_mask:
            tcp0 = utils.torch_to_numpy(detail_info["obs"][:, 0, 3, :, :])
            tcp1 = utils.torch_to_numpy(detail_info["obs"][:, obs1_idx, 3, :, :])
            for i in range(1):
                denses[1][i] = self.add_marker(denses[1][i], tcp0[i])
                denses[3][i] = self.add_marker(denses[3][i], tcp1[i])

        plot_wrap_fig(
            denses=denses,
            titles=titles,
            colorbars=colorbars,
            plot_batch_size=1
        ).savefig(pdf_path)
        plt.close()

    def training_step(self, batch, batch_idx):
        for k in ["rgb", "mask", "tcp"]:
            batch[k] = batch[k].to(dtype=getattr(torch, self.dtype_str))
        opt = self.optimizers()
        sch = self.lr_schedulers()

        total_loss, loss_dict, info_dict, detail_info = self.forward_all(batch)
        self.log_all(total_loss, loss_dict, info_dict, "train")
        
        self.manual_backward(total_loss / self.accumulate_grad_batches)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            opt.step()
            sch.step()
            opt.zero_grad()

    def validation_step(self, batch, batch_idx):
        for k in ["rgb", "mask", "tcp"]:
            batch[k] = batch[k].to(dtype=getattr(torch, self.dtype_str))
        total_loss, loss_dict, info_dict, detail_info = self.forward_all(batch)

        if batch_idx > 0:
            self.log_all(total_loss, loss_dict, info_dict, "valid")
        
        if batch_idx in self.learn_kwargs["valid"]["plot_batch_idx"]:
            self.log_img(batch_idx, detail_info)