import os
import sys
import pathlib

def setup_env_var():
    # determine base dir
    base_dir = os.path.abspath(__file__)
    base_dir = '/'.join(base_dir.split('/')[:-2])

    # pyflex
    os.environ["GARMENTDS_BASE_DIR"] = base_dir
    os.environ["PYFLEX_PATH"] = os.path.join(base_dir, "src/pyflex/PyFlex")
    sys.path.append(os.path.join(base_dir, "src/pyflex/PyFlex/bindings/build"))

    # openai
    # os.environ['OPENAI_API_KEY'] = "your_api_key_here"
    # os.environ['http_proxy'] = 'http://127.0.0.1:17891'
    # os.environ['https_proxy'] = 'http://127.0.0.1:17891'

setup_env_var()

import hydra
import omegaconf

import garmentds.common.utils as utils
from garmentds.gentexture import factory

@hydra.main(config_path="../config/run", config_name=pathlib.Path(__file__).stem, version_base='1.3')
def main(cfg: omegaconf.DictConfig):
    # setup
    utils.init_omegaconf()
    omegaconf.OmegaConf.resolve(cfg)
    cfg = utils.resolve_overwrite(cfg)
    omegaconf.OmegaConf.save(cfg, os.path.join(os.getcwd(), ".hydra", "resolved.yaml"))

    factory.make_cloth(**cfg)


if __name__ == "__main__":
    main()