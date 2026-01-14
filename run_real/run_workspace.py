import logging
logger = logging.getLogger(__name__)

import json
import pathlib
import os

import omegaconf
import hydra

from garmentds.real_galbot.workspace import RealWorkspace
import garmentds.common.utils as utils

from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear() # clear the global hydra instance to avoid issues with multiple runs

@hydra.main(config_path="../config/run_real", config_name=pathlib.Path(__file__).stem, version_base='1.3')
def main(cfg: omegaconf.DictConfig):
    logger.info(f"main pid: {os.getpid()}")
    # setup
    utils.init_omegaconf()
    omegaconf.OmegaConf.resolve(cfg)
    cfg = utils.resolve_overwrite(cfg)
    omegaconf.OmegaConf.save(cfg, os.path.join(os.getcwd(), ".hydra", "resolved.yaml"))
    
    utils.seed_all(cfg.misc.seed)
    
    workspace = RealWorkspace(cfg.realapi, cfg.env, cfg.policy, cfg.run)
    workspace.run()


if __name__ == '__main__':
    main()