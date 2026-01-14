# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from isaacsim import SimulationApp

# URDF import, configuration and simulation sample
kit = SimulationApp({"headless": True})
import omni.kit.commands
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.extensions import get_extension_path_from_name
from pxr import Gf, PhysxSchema, Sdf, UsdLux, UsdPhysics

status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
import_config.merge_fixed_joints = False
import_config.convex_decomp = False
import_config.import_inertia_tensor = True
import_config.fix_base = False
import_config.distance_scale = 1.0
import_config.self_collision = False

extension_path = get_extension_path_from_name("isaacsim.asset.importer.urdf")
status, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path="asset/galbot_one_charlie/urdf.urdf",
    import_config=import_config,
    get_articulation_root=True,
)

print(prim_path)

omni.usd.get_context().save_as_stage(f"asset/galbot_one_charlie/urdf.usd")

kit.close()
