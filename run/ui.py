import json
import os
import sys
import random
import pathlib
from dataclasses import asdict

import hydra
import omegaconf

from garmentds.genmesh.app import DesignApp
import garmentds.common.utils as utils
import garmentds.common.taichi as taichi_utils

from garmentds.genmesh.cfg import generate_cfg
from garmentds.genmesh.template import *

@hydra.main(config_path="../config/run", config_name=pathlib.Path(__file__).stem, version_base='1.3')
def main(cfg: omegaconf.DictConfig):
    # setup
    utils.init_omegaconf()
    omegaconf.OmegaConf.resolve(cfg)
    cfg = utils.resolve_overwrite(cfg)
    omegaconf.OmegaConf.save(cfg, os.path.join(os.getcwd(), ".hydra", "resolved.yaml"))
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.setup.cuda)
    taichi_utils.init_taichi(cfg.setup.taichi, cfg.glb_cfg)

    # select different types of garment
    
    garment = garment_dict[cfg.garment.category](**asdict(generate_cfg(
        category=cfg.garment.category, description=cfg.garment.description, 
        **(cfg.garment.cfg)
    )))
    if cfg.use_ui:
        app = DesignApp(garment)
        app.run()
    else:
        garment.triangulation()
        garment.quick_export("mesh.obj")


if __name__ == "__main__":
    main()