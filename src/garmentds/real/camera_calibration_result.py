import numpy as np
import trimesh.transformations as tra

# calibration result @ 640x480, 60 samples, 2024.12.6, yuxing chen
INTRINSICS = np.array([
    [605.2980346679688, 0.0, 320.4749755859375],
    [0.0, 605.0106811523438, 254.03187561035156],
    [0.0, 0.0, 1.0]
])
CAMERA_TO_UREE = (
    tra.translation_matrix([-0.03244379907846451, -0.1927538365125656, -0.2512200772762298]) @ 
    tra.euler_matrix(0.00482981679935078, 0.004030493331147451, 0.002252419911069797)
)
UREE_TO_URBASE = (
    tra.translation_matrix([0.3, 0.2, 0.5]) @
    tra.euler_matrix(np.pi, 0., -np.pi / 4)
)
CAMERA_TO_RMBASE = (
    tra.translation_matrix([-0.43639737367630005, -0.3684384822845459, 1.0836586952209473]) @
    tra.euler_matrix(3.0805417396116668, -0.037390576921109336, -0.8288143639449492)
)
RMBASE_TO_URBASE = UREE_TO_URBASE @ CAMERA_TO_UREE @ np.linalg.inv(CAMERA_TO_RMBASE)
RMBASE_TO_URBASE_RAW = RMBASE_TO_URBASE.copy()
r, p, y = tra.euler_from_matrix(RMBASE_TO_URBASE)
RMBASE_TO_URBASE[:3, :3] = tra.euler_matrix(0, 0, y)[:3, :3]
RMBASE_TO_URBASE[2, 3] = -0.34

# calibration result of d436 @ 640x480, 64 samples, 2025.1.17, yuxing chen
del INTRINSICS, CAMERA_TO_UREE
INTRINSICS = np.array(
    [[388.0154724121094, 0.0, 325.1499938964844],
     [0.0, 388.02178955078125, 234.33140563964844],
     [0.0, 0.0, 1.0]]
)
CAMERA_TO_UREE = (
    tra.translation_matrix([-0.031851474195718765, -0.10062260925769806, -0.13246460258960724]) @ 
    tra.euler_matrix(-0.0021131174384090394, 0.003921124046784441, -0.003341021328390815)
)

if __name__ == "__main__":
    print(INTRINSICS)
    print(CAMERA_TO_UREE)
    print(RMBASE_TO_URBASE)
    print(RMBASE_TO_URBASE_RAW)