import multiprocessing
from typing import Optional
import signal
import threading
import os
import atexit

import numpy as np
import cv2
import open3d as o3d


def pixel_to_xyz(i: int, j: int, d: float, intrinsics: np.ndarray):
    uvw = np.array([j + 0.5, i + 0.5, 1.])
    xyz = np.linalg.inv(intrinsics) @ uvw * d
    return xyz


def xyz_to_pixel(xyz: np.ndarray, intrinsics: np.ndarray):
    uvw = intrinsics @ xyz
    uvw /= uvw[2]
    return uvw[:2].astype(int)


def _show_window_worker(img: np.ndarray, name: str, q):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    while q.empty():
        key = cv2.waitKey(100)
        if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1.0: 
            break
    cv2.destroyWindow(name)


def _vis_pc_window_worker(xyz: np.ndarray, rgb: Optional[np.ndarray]):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        assert rgb.shape == xyz.shape, "{} != {}".format(rgb.shape, xyz.shape)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.visualization.draw_geometries([pcd])


class WindowVisualizer:
    def __init__(self) -> None:
        print(f"pid: {os.getpid()}")
        self._name_idx = 0
        self._enable = True
        
        self._q = multiprocessing.Manager().Queue()
        self._processes = []
        atexit.register(self._atexit_cleanup)

    def enable(self, enable=True):
        self._enable = bool(enable)
    
    def show(self, img: np.ndarray, name: Optional[str] = None):
        if self._enable:
            self._name_idx += 1
            if name is None:
                name = f"window_{self._name_idx}"
            p = multiprocessing.Process(target=_show_window_worker, args=(img, name, self._q), daemon=True)
            p.start()
            self._processes.append(p)
    
    def vis_pc(self, xyz: np.ndarray, rgb: np.ndarray = None):
        if self._enable:
            p = multiprocessing.Process(target=_vis_pc_window_worker, args=(xyz, rgb), daemon=True)
            p.start()
            self._processes.append(p)
    
    def _atexit_cleanup(self):
        self._q.put("stop")
        for p in self._processes:
            p.join()


vis = WindowVisualizer()


class SIGUSR1Exception(Exception):
    pass


try:
    import pynput

    class CtrlB_SIGUSR1_Catcher:
        def __init__(self):
            def handle_sigusr1(signum, frame):
                raise SIGUSR1Exception("Received SIGUSR1 signal!")
            signal.signal(signal.SIGUSR1, handle_sigusr1)
            print("register SIGUSR1 handler")

            self.ctrl_pressed = False
            self.listener_thread = threading.Thread(target=self.start_keyboard_listener, daemon=True)
            self.listener_thread.start()
            print("start CtrlB_SIGUSR1_Catcher listener_thread")

        def start_keyboard_listener(self):
            with pynput.keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
                listener.join()
        
        def on_release(self, key):
            if key == pynput.keyboard.Key.ctrl_l or key == pynput.keyboard.Key.ctrl_r:
                self.ctrl_pressed = False
        
        def on_press(self, key):
            if key == pynput.keyboard.Key.ctrl_l or key == pynput.keyboard.Key.ctrl_r:
                self.ctrl_pressed = True
            try:
                if key.char == "b" and self.ctrl_pressed:
                    print("Ctrl+B pressed, send SIGUSR1!")
                    os.kill(os.getpid(), signal.SIGUSR1)
            except AttributeError:
                pass
    
    _catcher = CtrlB_SIGUSR1_Catcher()

except ImportError as e:
    print(f"{e}\nignore ...")

except Exception as e:
    raise e


if __name__ == "__main__":
    vis.show((np.random.rand(480, 640, 3) * 255).astype(np.uint8))
    vis.vis_pc(np.random.rand(100, 3))
