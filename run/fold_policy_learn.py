import json
import os
import sys
import pathlib
import pprint

import hydra
import omegaconf

import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy

import garmentds.common.utils as utils
from garmentds.foldenv.fold_learn import FoldPolicyModule, make_dataset, get_profiler, TorchProfilerCallback, learn_timer


@hydra.main(config_path="../config/run", config_name=pathlib.Path(__file__).stem, version_base='1.3')
def main(cfg: omegaconf.DictConfig):
    # setup
    utils.init_omegaconf()
    omegaconf.OmegaConf.resolve(cfg)
    cfg = utils.resolve_overwrite(cfg)
    omegaconf.OmegaConf.save(cfg, os.path.join(os.getcwd(), ".hydra", "resolved.yaml"))
    utils.seed_all(cfg.misc.seed)
    torch.set_float32_matmul_precision(cfg.misc.hardware.precision)
    learn_timer.disable = not bool(cfg.misc.debug.timer)

    with open(os.path.join("command_line.json"), "w") as f_obj:
        json.dump(obj=dict(
            command_line=" ".join(sys.argv),
            run_command_dir=utils.get_path_handler()("."),
        ), fp=f_obj, indent=4)
    
    overrides = omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.load(os.path.join(os.getcwd(), ".hydra", "overrides.yaml")))
    logger_file_name = "logger-" + "-".join([
        o.split("=")[0].split(".")[-1] + "=" + o.split("=")[1] for o in overrides if (
        not o.startswith("train.path") and 
        not o.startswith("data.loader.num_workers")
    )])
    logger = pl_loggers.TensorBoardLogger(save_dir=os.getcwd(), name=logger_file_name, version="")
    if utils.ddp_is_rank_0():
        print("logger.log_dir", logger.log_dir)
        os.makedirs(logger.log_dir)
    profiler = SimpleProfiler(dirpath=logger.log_dir, filename="perf_logs")

    # dataset and dataloader
    trds, vlds, stat, info = make_dataset(
        [utils.get_path_handler()(s) for s in cfg.train.path.data_paths], 
        cfg.data.make, cfg.data.dataset, cfg.misc.seed, 
        cfg.data.cache.quick_find, cfg.data.cache.pkl_data
    )
    trdl = DataLoader(trds, shuffle=True, **(cfg.data.loader))
    vldl = DataLoader(vlds, shuffle=False, **(cfg.data.loader))
    if utils.ddp_is_rank_0():
        with open("stat.json", "w") as f:
            json.dump(obj=stat, fp=f, indent=4, default=utils.custom_serializer)
        with open("dataset_info.json", "w") as f:
            json.dump(obj=info, fp=f, indent=4)

    # training
    ckpt_path = cfg.train.path.ckpt
    if str(ckpt_path).lower() == "none":
        ckpt_path = None
    # init model
    if ckpt_path is not None:
        model = FoldPolicyModule.load_from_checkpoint(
            utils.get_path_handler()(ckpt_path), 
            model_kwargs=omegaconf.OmegaConf.to_container(cfg.pl.model), 
            learn_kwargs=omegaconf.OmegaConf.to_container(cfg.pl.learn),
            map_location=torch.device("cpu"),
        )
    else:
        model = FoldPolicyModule(
            model_kwargs=omegaconf.OmegaConf.to_container(cfg.pl.model), 
            learn_kwargs=omegaconf.OmegaConf.to_container(cfg.pl.learn),
        )
    model.set_statistics(stat)

    # init trainer
    p = get_profiler()
    trainer_kwargs = {
        "accelerator": "cuda" if cfg.misc.hardware.cuda else "cpu",
        "devices": cfg.misc.hardware.gpuids if cfg.misc.hardware.gpuids else "auto",
        "strategy": DDPStrategy(find_unused_parameters=cfg.train.cfg.ddp_find_unused_parameters),
        "max_steps": cfg.train.cfg.max_steps + cfg.misc.step_offset,
        "logger": logger,
        "profiler": profiler,
        "limit_train_batches": cfg.train.cfg.limit_train_batches,
        "limit_val_batches": cfg.train.cfg.limit_val_batches,
        "log_every_n_steps": cfg.train.cfg.log_every_n_steps,
        "val_check_interval": cfg.train.cfg.val_check_interval,
        "check_val_every_n_epoch": None,
        "callbacks": [
            ModelCheckpoint(
                every_n_train_steps=cfg.train.cfg.ckpt_every_n_steps, 
                save_top_k=-1, 
                dirpath=f'ckpt', 
                filename='epoch_{epoch}_step_{step}', 
                auto_insert_metric_name=False
            ), 
            ModelSummary(max_depth=cfg.misc.debug.model_summary_max_depth), 
            LearningRateMonitor(logging_interval='step'),
            TorchProfilerCallback(p),
        ],
    }
    trainer = pl.Trainer(**trainer_kwargs)

    with p:
        # train
        print("start fitting model...")
        ckpt_path = utils.get_path_handler()(ckpt_path) if ckpt_path is not None else None
        trainer.fit(
            model=model,
            train_dataloaders=trdl,
            val_dataloaders=vldl,
            ckpt_path=ckpt_path,
        )


if __name__ == "__main__":
    main()