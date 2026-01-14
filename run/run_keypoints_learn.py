from re import A
import taichi as ti

import json
import os
import sys
import random
import pathlib

import hydra
import omegaconf

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler, PyTorchProfiler
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, ModelSummary, LearningRateMonitor

import garmentds.common.utils as utils
import garmentds.keypoint_detection.utils.learn_utils as learn_utils
from garmentds.keypoint_detection.data.datamodule import KeypointsDataModule
from garmentds.keypoint_detection.models.detector import KeypointDetector

@hydra.main(config_path="../config/run", config_name=pathlib.Path(__file__).stem, version_base='1.3')
def main(cfg: omegaconf.DictConfig):
    # setup
    utils.init_omegaconf()
    omegaconf.OmegaConf.resolve(cfg)
    cfg = utils.resolve_overwrite(cfg)
    omegaconf.OmegaConf.save(cfg, os.path.join(os.getcwd(), ".hydra", "resolved.yaml"))

    # init numpy, pytorch, taichi
    torch.random.manual_seed(cfg.misc.seed)
    np.random.seed(cfg.misc.seed)
    random.seed(cfg.misc.seed)
    torch.set_float32_matmul_precision(cfg.misc.hardware.precision)
    # ti.init(
    #     arch=getattr(ti, cfg.misc.taichi.device), 
    #     default_fp=ti.f32, default_ip=ti.i32, device_memory_GB=cfg.misc.taichi.device_memory_GB, 
    #     offline_cache=True, fast_math=False, debug=False
    # )

    with open(os.path.join("command_line.json"), "w") as f_obj:
        json.dump(obj=dict(
            command_line=" ".join(sys.argv),
            run_command_dir=utils.get_path_handler()("."),
        ), fp=f_obj, indent=4)


    # data module
    dtmd = KeypointsDataModule(
        [utils.get_path_handler()(s) for s in cfg.train.path.data_paths], cfg.data,
    )
    # ti.reset(), ti.init(ti.cpu)

    # training
    ckpt_path = cfg.train.path.ckpt
    if cfg.pl.learn.test.output_dir is None:
        cfg.pl.learn.test.output_dir = os.path.join(os.getcwd(), "visualization")

    # init model
    if cfg.run.mode == "train":
        prefix, texture_type = os.path.split(cfg.train.path.data_paths[0])
        prefix, cloth_type = os.path.split(prefix)
    elif cfg.run.mode == "test":
        prefix, cloth_type = os.path.split(cfg.train.path.data_paths[0])

    if ckpt_path is not None:
        log_dir = os.path.dirname(os.path.dirname(ckpt_path))
        for d in os.listdir(log_dir):
            if d.startswith(cloth_type):
                origin_cfg_path = os.path.join(log_dir, d, "hparams.yaml")
                break
        origin_cfg = omegaconf.OmegaConf.load(origin_cfg_path)
        origin_cfg.cfg_pl.learn.test = cfg.pl.learn.test
        model = KeypointDetector.load_from_checkpoint(
            utils.get_path_handler()(ckpt_path),
            map_location=torch.device("cuda" if cfg.misc.hardware.cuda else "cpu"),
            cfg_pl=origin_cfg.cfg_pl,
        )
        ckpt_dir = os.path.dirname(os.path.dirname(ckpt_path))
        log_name =  [d for d in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, d)) 
                    and (d.startswith(cloth_type))][0]
    else:
        model = KeypointDetector(cfg.pl)
        assert isinstance(model, pl.LightningModule), "model must be a LightningModule"
        log_name = "{}_{}_{}{}{}{}".format(cloth_type, texture_type, 
                                      int(cfg.data.dataset.aug.patch.use_patch),
                                      int(cfg.pl.ablation.only_use_mask_as_input),
                                      int(cfg.pl.model.net.MaxVitUnet.pass_rgb_to_decoder),
                                      int(cfg.pl.model.net.MaxVitUnet.use_conv_on_encoder_output))

    # Set logger and profiler
    logger = pl_loggers.TensorBoardLogger(os.getcwd(), version="", name=log_name)
    if utils.ddp_is_rank_0():
        print("logger.log_dir", logger.log_dir)
        os.makedirs(logger.log_dir)
    profiler_name2class = {"Advanced": AdvancedProfiler, "Simple": SimpleProfiler, "PyTorch": PyTorchProfiler}
    profiler = profiler_name2class[cfg.misc.debug.profiler](dirpath=logger.log_dir, filename="perf_logs")

    # init trainer
    class CustomCheckpointCallback(Callback):
        def on_validation_end(self, trainer:pl.Trainer, pl_module):
            if trainer.global_step == 0: # skip sanity check
                return

            # Access validation metrics
            avgKD = trainer.callback_metrics.get("validation/avgKD", 0.000)
            meanAP = trainer.callback_metrics.get("validation/meanAP", 0.000)
            filename = f"epoch_{trainer.current_epoch}_step_{trainer.global_step}_avgKD_{avgKD:.3f}_meanAP_{meanAP:.3f}.ckpt"

            # Save the checkpoint manually
            checkpoint_path = os.path.join("ckpt", filename)
            trainer.save_checkpoint(checkpoint_path)

    p = learn_utils.get_profiler()
    trainer_kwargs = {
        "gradient_clip_val": 5.0,
        "gradient_clip_algorithm": "value",
        "accelerator": "cuda" if cfg.misc.hardware.cuda else "cpu",
        "devices": cfg.misc.hardware.gpuids if cfg.misc.hardware.gpuids else "auto",
        "strategy": "ddp_find_unused_parameters_true",
        "max_steps": cfg.train.cfg.max_steps + cfg.train.cfg.step_offset,
        "accumulate_grad_batches": cfg.train.cfg.accumulate_grad_batches,
        "logger": logger,
        "profiler": profiler,
        "limit_train_batches": cfg.train.cfg.limit_train_batches,
        "limit_val_batches": cfg.train.cfg.limit_val_batches,
        "log_every_n_steps": cfg.train.cfg.log_every_n_steps,
        "val_check_interval": cfg.train.cfg.val_check_interval,
        "check_val_every_n_epoch": None,
        "callbacks": [
            #ModelCheckpoint(
            #    every_n_train_steps=cfg.train.cfg.ckpt_every_n_steps, 
            #    save_top_k=-1, 
            #    dirpath=f'ckpt', 
            #    filename='epoch_{epoch}_step_{step}_avgKD_{validation/avgKD:.3f}_meanAP_{validation/meanAP:.3f}', 
            #    auto_insert_metric_name=False,
            #),
            CustomCheckpointCallback(), 
            ModelSummary(max_depth=cfg.misc.debug.model_summary_max_depth), 
            LearningRateMonitor(logging_interval='step'),
            learn_utils.TorchProfilerCallback(p),
        ],
    }
    trainer = pl.Trainer(**trainer_kwargs)

    with p:
        # train
        if cfg.run.mode == "train":
            print("start fitting model...")
            ckpt_path = utils.get_path_handler()(ckpt_path) if ckpt_path is not None else None
            trainer.fit(
                model=model,
                datamodule=dtmd,
                ckpt_path=ckpt_path,
            )
        elif cfg.run.mode == "eval":
            print("start evaluating model...")
            trainer.validate(
                model=model,
                datamodule=dtmd,
            )
        elif cfg.run.mode == "test":
            print("start testing model...")
            trainer.test(
                model=model,
                datamodule=dtmd,
            )
        else:
            raise ValueError(cfg.run.mode)


if __name__ == "__main__":
    main()