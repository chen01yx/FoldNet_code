
import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional
import os
import tqdm
import time

import omegaconf

from garmentds.real_galbot.realapi import RealAPI, RealAPICfg, Picker
from garmentds.real_galbot.policy import FoldRealVisualPolicy, FoldRealVisualPolicyCfg
from garmentds.real_galbot.env import FoldRealEnv, FoldRealEnvCfg
from garmentds.foldenv.fold_env import RobotCfg


realapi_cfg_dataclass = RealAPICfg(
    robot_cfg=RobotCfg(
        device="cpu", ik_kwargs=dict(max_iter=16, square_err_th=1e-6),
        urdf_path="asset/galbot_one_charlie/urdf_nomtl.urdf"
    ), 
    robot_offset_l=[-0.01, 0., -0.010],
    robot_offset_r=[-0.01, 0., -0.030],
    grasp_offset_l=[+0.00, 0., -0.020],
    grasp_offset_r=[-0.02, 0., -0.040],
    grasp_force_z=0.02,
    press_enter_before_move=True,
    use_visualizer=True,
)
api = RealAPI(realapi_cfg_dataclass)
api.moveinit()
api.step(xyz_l=np.array([-0.4, -0.3, 0.1]), xyz_r=np.array([+0.4, -0.3, 0.1]))
api.step(picker_l=Picker.CLOSE, picker_r=Picker.CLOSE)
api.step(picker_l=Picker.OPEN, picker_r=Picker.OPEN)
api.step(xyz_l=np.array([-0.4, +0.3, 0.1]), xyz_r=np.array([+0.4, +0.3, 0.1]))
api.step(picker_l=Picker.CLOSE, picker_r=Picker.CLOSE)
api.step(picker_l=Picker.OPEN, picker_r=Picker.OPEN)

api.step(xyz_l=np.array([-0.1, -0.3, 0.1]), xyz_r=np.array([+0.1, -0.3, 0.1]))
api.step(picker_l=Picker.CLOSE, picker_r=Picker.CLOSE)
api.step(picker_l=Picker.OPEN, picker_r=Picker.OPEN)
api.step(xyz_l=np.array([-0.1, +0.3, 0.1]), xyz_r=np.array([+0.1, +0.3, 0.1]))
api.step(picker_l=Picker.CLOSE, picker_r=Picker.CLOSE)
api.step(picker_l=Picker.OPEN, picker_r=Picker.OPEN)