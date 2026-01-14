from typing import Optional, Literal
import math
import argparse
import multiprocessing
import pprint

import cv2
import torch
import numpy as np
import trimesh.transformations as tra
import matplotlib.pyplot as plt
from PIL import Image

from garmentds.real.real_api_desktop import RealAPIDesktop, RobotlCfg, RobotrCfg, CameraCfg


board_size = (8, 11)


def hand_eye_calibrate(
    hand_pos_list: list[np.ndarray],
    image_list: list[Image.Image],
    cameraMatrix=None,
    debug_mode=True,
):
    square_size = 0.02
    H, W = 480, 640

    obj_points = []
    img_points = []
    hand_pos_list_find_corner_success = []
    for image, hand_pos in zip(image_list, hand_pos_list):
        image_np = np.array(image)
        assert image_np.shape == (H, W, 3)
        retval, corner = cv2.findChessboardCorners(image_np, board_size, None)
        if not retval: 
            continue # find chess board corners failed

        image_with_corner = cv2.drawChessboardCorners(image_np, board_size, corner, retval)
        xyz = []
        for i in range(board_size[1]):
            for j in range(board_size[0]):
                xyz.append([i * square_size, j * square_size, 0.])
        obj_points.append(xyz)
        img_points.append(corner)
        hand_pos_list_find_corner_success.append(hand_pos)

        if debug_mode:
            plt.figure()
            plt.imshow(image_with_corner)
            plt.show()
    
    if cameraMatrix is None:
        ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints=np.array(obj_points, dtype=np.float32),
            imagePoints=np.array(img_points, dtype=np.float32),
            imageSize=np.array([H, W], dtype=np.int32),
            cameraMatrix=None, distCoeffs=None, flags=None, criteria=None
        )
    else:
        rvec, tvec = [], []
        for o, i in zip(obj_points, img_points):
            ret, r, t = cv2.solvePnP(
                objectPoints=np.array(o, dtype=np.float32),
                imagePoints=np.array(i, dtype=np.float32),
                cameraMatrix=cameraMatrix, distCoeffs=None,
            )
            rvec.append(r), tvec.append(t)
        rvecs, tvecs = np.array(rvec, dtype=np.float32), np.array(tvec, dtype=np.float32)

    R_cam, T_cam = cv2.calibrateHandEye(
        [tra.euler_matrix(*(i[3:6]))[:3, :3] for i in hand_pos_list_find_corner_success],
        [i[0:3] for i in hand_pos_list_find_corner_success],
        rvecs, tvecs,
    )
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = R_cam
    extrinsic[:3, [3]] = T_cam
    intrinsic = np.array(cameraMatrix, dtype=np.float32)

    if debug_mode:
        print(ret)
        print(cameraMatrix)
        print(distCoeffs)
        print(rvecs)
        print(tvecs)
        print(R_cam, T_cam)
    
    return extrinsic, intrinsic


class CalibrateCamera:
    def __init__(self, target: Literal["ur5", "rm"]) -> None:
        if target == "ur5":
            self._api = RealAPIDesktop(
                robotl_cfg=RobotlCfg(use_gripper=False),
                robotr_cfg=RobotrCfg(use=False),
                camera_cfg=CameraCfg()
            )
        elif target == "rm":
            self._api = RealAPIDesktop(
                robotl_cfg=RobotlCfg(
                    use_gripper=False,
                    robot_tcp_init_mat_to_base=tra.translation_matrix([0.3, 0.2, 0.5]) @ tra.euler_matrix(math.pi, 0., -math.pi / 4),
                ),
                robotr_cfg=RobotrCfg(
                    use_gripper=False,
                    robot_joints_init_val=(0.5, 0.5, 2.0, -0.5, -1.0, -0.0),
                    robot_tcp_init_mat_to_base=tra.translation_matrix([-0.4, -0.3, 0.25]) @ tra.euler_matrix(math.pi / 2, 0., -math.pi / 4),
                ),
                camera_cfg=CameraCfg()
            )
        else:
            raise NotImplementedError(target)
        
        self.target = target

        self._vis_queue = multiprocessing.Manager().Queue()
        self._vis_process = multiprocessing.Process(target=self._cv2_process, daemon=True)
        self._vis_process.start()
        self._vis_queue.put((cv2.cvtColor(self._api.take_picture()["color"], cv2.COLOR_RGB2BGR), 0))
    
    def _cv2_process(self):
        while True:
            data = self._vis_queue.get()
            if data is None:
                print("break because of img is None")
                break
            img, idx = data
            retval, corner = cv2.findChessboardCorners(img, board_size, None)
            if not retval: 
                cv2.imshow("img", img)
            else:
                image_with_corner = cv2.drawChessboardCorners(img, board_size, corner, retval)
                cv2.imshow("img", image_with_corner)
            cv2.setWindowTitle("img", f"image {idx}")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
    
    def _move_to_pos_and_take_picture(self, xyzrpy: np.ndarray):
        mat = tra.translation_matrix(xyzrpy[0:3]) @ tra.euler_matrix(*(xyzrpy[3:6]))
        if self.target == "ur5":
            self._api.robotl_movel(self._api.robotl_cfg.robot_base_mat @ mat, acc=1.0, vel=1.0)
        elif self.target == "rm":
            self._api.robotr_movel(self._api.robotr_cfg.robot_base_mat @ mat, speed=0.2)
        return self._api.take_picture()["color"]

    def _get_xyzrpy(self):
        """relative to robot base"""
        def random_ab(a: float, b: float, *args):
            return np.random.rand(*args) * (b - a) + a
        
        if self.target == "ur5":
            xyz_mean = np.array([0.5, 0.2, 0.35])
            def generate_rot_xy():
                while True:
                    x, y = random_ab(-0.2, +0.2, 2)
                    if (x ** 2 + y ** 2) ** 0.5 > 0.2:
                        return x, y
            rot_x, rot_y = generate_rot_xy()
            angle = (rot_x ** 2 + rot_y ** 2) ** 0.5
            direc = [rot_x / angle, rot_y / angle, 0.]

            rot_1 = tra.euler_matrix(math.pi, 0., 0.) @ tra.euler_matrix(0., 0., random_ab(-0.3, +0.3))
            rot_2 = tra.rotation_matrix(angle, direc, [0., 0., -0.3]) @ rot_1
            
            mat = rot_2.copy()
            mat[:3, 3] += xyz_mean + random_ab(-0.05, +0.05, 3) * np.array([1., 1., 2.])

            xyzrpy = np.array([*mat[:3, 3], *tra.euler_from_matrix(mat)])
            return xyzrpy
        
        elif self.target == "rm":
            def generate_random_rot():
                while True:
                    xyz = np.random.randn(3)
                    x, y, z = xyz
                    if 0.3 < np.linalg.norm([x, y]) < 0.5 and abs(z) < 0.3:
                        return tra.rotation_matrix(np.linalg.norm(xyz), xyz / np.linalg.norm(xyz))
            def generate_random_pos():
                return random_ab(-0.05, +0.05, 3) * np.array([1., 1., 2.])
            
            mat = (
                tra.translation_matrix(np.array([-0.4, -0.3, 0.25]) + generate_random_pos()) @ 
                generate_random_rot() @ tra.euler_matrix(math.pi / 2, 0., -math.pi / 4)
            )
            
            xyzrpy = np.array([*mat[:3, 3], *tra.euler_from_matrix(mat)])

            return xyzrpy
        
        else:
            raise NotImplementedError(self.target)
    
    def _xyzrpy_inverse(self, xyzrpy):
        mat = tra.translation_matrix(xyzrpy[0:3]) @ tra.euler_matrix(*(xyzrpy[3:6]))
        mat_inv = np.linalg.inv(mat)
        return np.array([*mat_inv[:3, 3], *tra.euler_from_matrix(mat_inv)])

    def run(self):
        try:
            hand_pos_list = []
            image_list = []
            
            while True:
                xyzrpy = self._get_xyzrpy()
                if self.target == "ur5":
                    hand_pos_list.append(xyzrpy)
                elif self.target == "rm":
                    hand_pos_list.append(self._xyzrpy_inverse(xyzrpy))
                else:
                    raise NotImplementedError(self.target)

                color = self._move_to_pos_and_take_picture(xyzrpy)
                image_list.append(Image.fromarray(color))
                self._vis_queue.put((cv2.cvtColor(np.array(color), cv2.COLOR_RGB2BGR), len(image_list)))

                if len(image_list) >= 4 and len(image_list) % 4 == 0:
                    # calibrate ur5 first, then rm
                    # use ur5 intrinsics when calibrating rm
                    if self.target == "ur5":
                        camera_matrix = None
                    elif self.target == "rm":
                        from garmentds.real.camera_calibration_result import INTRINSICS
                        camera_matrix = INTRINSICS
                    else:
                        raise NotImplementedError(self.target)

                    extrinsic, intrinsic = hand_eye_calibrate(
                        hand_pos_list, image_list, 
                        cameraMatrix=camera_matrix, debug_mode=False,
                    )

                    print(f"len {len(image_list)} current param:")
                    print(list(tra.euler_from_matrix(extrinsic)))
                    print(tra.translation_from_matrix(extrinsic).tolist())
                    pprint.pprint(intrinsic.tolist())
                    if self.target == "rm":
                        ur5ee_to_ur5base = np.linalg.inv(self._api.robotl_cfg.robot_base_mat) @ self._api.get_robotl_ee_pose()
                        print("ur5ee to ur5base:")
                        print(list(tra.euler_from_matrix(ur5ee_to_ur5base)))
                        print(tra.translation_from_matrix(ur5ee_to_ur5base).tolist())

                    ans = input("press q to quit, any key to continue ...")
                    if ans == "q":
                        self._vis_queue.put(None)
                        self._vis_process.join()
                        self._api.close()
                        break
        except Exception as e:
            self._api.close()
            raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="ur5", choices=["ur5", "rm"])
    args = parser.parse_args()

    cal = CalibrateCamera(args.target)
    cal.run()