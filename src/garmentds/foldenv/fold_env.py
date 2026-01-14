import pyflex # type: ignore

import logging
logger = logging.getLogger(__name__)

import os
import multiprocessing as mp
import subprocess
import time
import json
import atexit
from dataclasses import dataclass, field, asdict
import copy
from typing import Optional, Literal, Iterable, Callable, Any
from collections import deque, defaultdict
import pprint
import random
import socket

import psutil
import numpy as np
import trimesh
import trimesh.transformations as tra
from PIL import Image
import torch

import tqdm

import batch_urdf

import garmentds.common.utils as utils
from garmentds.foldenv.socket_utils import sendall, recvall

env_timer = utils.Timer(name="fold_env", logger=logger)


@dataclass
class RobotCfg:
    urdf_path: str = "asset/galbot_one_charlie/urdf.urdf"
    mesh_dir: str = "asset/galbot_one_charlie/meshes"
    device: str = "cuda:0"

    arm_l_joints: list[str] = field(default_factory=lambda: ["left_arm_joint1", "left_arm_joint2", "left_arm_joint3", "left_arm_joint4", "left_arm_joint5", "left_arm_joint6", "left_arm_joint7"])
    tcp_l: str = "left_gripper_tcp_link"
    arm_r_joints: list[str] = field(default_factory=lambda: ["right_arm_joint1", "right_arm_joint2", "right_arm_joint3", "right_arm_joint4", "right_arm_joint5", "right_arm_joint6", "right_arm_joint7"])
    tcp_r: str = "right_gripper_tcp_link"

    base_link: str = "base_link"
    base_pos: np.ndarray = field(default_factory=lambda: [
        0.0, -0.75, -0.455, 0.70710678, 0., 0., 0.70710678
    ])
    gripper_l_joints: list[str] = field(default_factory=lambda: [
        "left_gripper_l1_joint", "left_gripper_l2_joint", "left_gripper_l3_joint",
        "left_gripper_r1_joint", "left_gripper_r2_joint", "left_gripper_r3_joint",
    ])
    gripper_r_joints: list[str] = field(default_factory=lambda: [
        "right_gripper_l1_joint", "right_gripper_l2_joint", "right_gripper_l3_joint",
        "right_gripper_r1_joint", "right_gripper_r2_joint", "right_gripper_r3_joint",
    ])
    gripper_open_val: float = 0.7
    gripper_close_val: float = 0.0

    leg_joints: list[str] = field(default_factory=lambda: ["leg_joint1", "leg_joint2", "leg_joint3", "leg_joint4"])

    init_qpos: dict[str, float] = field(default_factory=lambda: dict(
        leg_joint1=0.10, leg_joint2=0.80, leg_joint3=0.95, leg_joint4=0.0,
        left_arm_joint1=+0.0,  left_arm_joint2=+0.4,  left_arm_joint3=-1.5,  left_arm_joint4=-1.5,  left_arm_joint5=0.0,  left_arm_joint6=+0.5,  left_arm_joint7=-0.5,
        right_arm_joint1=-0.0, right_arm_joint2=-0.4, right_arm_joint3=+1.5, right_arm_joint4=+1.5, right_arm_joint5=0.0, right_arm_joint6=-0.5, right_arm_joint7=+0.5,
        left_gripper_l1_joint=0.7, left_gripper_l2_joint=0.7, left_gripper_l3_joint=0.7,
        left_gripper_r1_joint=0.7, left_gripper_r2_joint=0.7, left_gripper_r3_joint=0.7,
        right_gripper_l1_joint=0.7, right_gripper_l2_joint=0.7, right_gripper_l3_joint=0.7,
        right_gripper_r1_joint=0.7, right_gripper_r2_joint=0.7, right_gripper_r3_joint=0.7,
        head_joint1=0., head_joint2=0.25,
    ))

    # ik params
    ik_init_qpos: dict[str, float] = field(default_factory=lambda: dict(
        leg_joint1=0.10, leg_joint2=0.80, leg_joint3=0.95, leg_joint4=0.0,
        left_arm_joint1=+0.5,  left_arm_joint2=-0.4,  left_arm_joint3=-2.0,  left_arm_joint4=-1.5,  left_arm_joint5=+0.5,  left_arm_joint6=+0.5,  left_arm_joint7=-0.5,
        right_arm_joint1=-0.5, right_arm_joint2=+0.4, right_arm_joint3=+2.0, right_arm_joint4=+1.5, right_arm_joint5=-0.5, right_arm_joint6=-0.5, right_arm_joint7=+0.5,
        left_gripper_l1_joint=0.7, left_gripper_l2_joint=0.7, left_gripper_l3_joint=0.7,
        left_gripper_r1_joint=0.7, left_gripper_r2_joint=0.7, left_gripper_r3_joint=0.7,
        right_gripper_l1_joint=0.7, right_gripper_l2_joint=0.7, right_gripper_l3_joint=0.7,
        right_gripper_r1_joint=0.7, right_gripper_r2_joint=0.7, right_gripper_r3_joint=0.7,
        head_joint1=0., head_joint2=0.25,
    ))
    ik_gripper_z_deg_max: tuple[float, float, float, float] = (60.0, 0.05, 120.0, 0.10) # between z = 0.05 and z = 0.10, max angle is 60 to 120
    ik_move_leg: bool = True
    ik_move_leg_dy_range: tuple[float, float] = (0.05, 0.15) # (for negative and for positive)
    ik_move_leg_py_th_pos: tuple[float, float] = (0.2, 0.2 + 0.15 * 1.2)
    ik_move_leg_py_th_neg: tuple[float, float] = (-0.2 - 0.05 * 1.2, -0.2)
    ik_move_leg_pz_th: tuple[float, float] = (0.10, 0.20)
    ik_move_leg_joint4_disty_th: tuple[float, float] = (0.3, 0.7)
    ik_move_leg_joint4_max_val: float = np.pi / 4
    
    ik_kwargs: dict = field(default_factory=lambda: dict(max_iter=64, square_err_th=1e-10))

    def __post_init__(self):
        self.urdf_path = utils.get_path_handler()(self.urdf_path)
        self.mesh_dir = utils.get_path_handler()(self.mesh_dir)


@dataclass
class FoldEnvCfg:
    # misc params
    use_tqdm: bool = True

    # cloth params
    cloth_obj_path: str = "your/path/to/mesh.obj"
    cloth_scale: float = 0.5

    # render params
    render: bool = True
    render_per_n_substep: int = None # None means render per step (not substep)
    render_output_dir: str = "tmp_render_output"
    render_mode: list[Literal["head", "side", "mesh"]] = field(default_factory=lambda: ["head", "side", "mesh"])
    render_remove_tmp_files: bool = False
    render_method: Literal["isaacsim", "blender"] = "isaacsim"
    render_process_num: int = 4
    render_set: Literal["train", "valid"] = "train"

    render_blender_command: str = "blender"
    render_blender_save_blend: bool = False
    render_blender_engine: Literal["cycles", "eevee"] = "cycles"
    render_blender_camera_type: Literal["d435", "d436"] = "d436"
    render_blender_hide_picker: bool = True
    render_blender_debug_level: int = 0
    render_blender_camera_size_level: int = 2
    render_blender_skip_first_cuda: bool = False

    # robot params
    robot_cfg: RobotCfg = field(default_factory=lambda: RobotCfg())

    # physics params
    n_steps_per_secend: int = 5
    n_substep: int = 5
    n_pyflex_steps_per_substep: int = 1
    sim_dt_robot: float = field(init=False)
    sim_dt_pyflex: float = field(init=False)

    collision_radius: Optional[float] = None # dynamically determined by cloth mesh
    cloth_mass: float = 0.2 # total clothes mass
    cloth_stiff_stretch: float = 2.0
    cloth_stiff_bend: float = 0.02
    dynamicFriction: float = 0.5 # Coefficient of friction used when colliding against shapes
    particleFriction: float = 1.0 # Coefficient of friction used when colliding particles
    numSolverIterations: int = 20 # Number of iterations used in the constraint solver
    numSolverSubsteps: int = 80 # Number of substeps used in the constraint solver

    grasp_threshold: dict[str, np.ndarray] = field(default_factory=lambda: dict(
        left=np.array([.04, .02, .02]), right=np.array([.04, .02, .02]), init_cloth=np.array([.02, .02, .02]), 
    ))
    grasp_squeeze_factor: dict[str, np.ndarray] = field(default_factory=lambda: dict(
        left=np.array([.5, 1., 1.]), right=np.array([.5, 1., 1.]), init_cloth=np.array([1., 1., 1.])
    ))

    # env params
    init_cloth_step: int = 5
    init_cloth_picker_z: float = 0.02
    init_cloth_rot_z_range: list[float] = field(default_factory=lambda: [-np.pi, +np.pi])
    init_cloth_rot_y_pi_prob: float = 0.5
    init_cloth_rot_resample_when_y_out_of_range: list[float] = field(default_factory=lambda: [-1., +1.]) # when y is out of range, resample the rotation angle
    init_cloth_rot_resample_max_try: int = 100
    init_cloth_vel_range: list[float] = field(default_factory=lambda: [5.0, 10.0])
    init_cloth_vel_coeff: list[float] = field(default_factory=lambda: [1., 1., .5])
    init_cloth_move_random_pick_place: bool = False
    init_cloth_move_h_range: list[float] = field(default_factory=lambda: [0.03, 0.06])
    init_cloth_move_r_range: list[float] = field(default_factory=lambda: [0.05, 0.10])
    init_cloth_xy_random: float = 0.05
    init_cloth_xy_offset: list[float] = field(default_factory=lambda: [0.0, 0.0])

    post_fold_step: int = 20
    dr_args: Optional[dict[str, tuple[float, float]]] = field(default_factory=lambda: dict(
        dynamicFriction = (0.75, 1.25),
        particleFriction = (0.75, 1.25),
        cloth_stiff_stretch = (0.75, 1.25), 
        cloth_stiff_bend = (0.75, 1.25)
    ))

    # ======================================================================== #

    def __post_init__(self):
        self.cloth_obj_path = utils.get_path_handler()(self.cloth_obj_path)
        if self.render_per_n_substep is None:
            self.render_per_n_substep = self.n_substep
        if not self.render:
            self.render_mode = []
        self.sim_dt_robot = 1. / self.n_steps_per_secend / self.n_substep
        self.sim_dt_pyflex = self.sim_dt_robot / self.n_pyflex_steps_per_substep


class RenderProcess:
    tmp_dir: str = utils.get_path_handler()(".tmp")


class RenderProcessIsaacsim(RenderProcess):
    def __init__(
        self, 
        env_cfg: FoldEnvCfg, 
    ):
        self._env_cfg = copy.deepcopy(env_cfg)
        self._room_num = env_cfg.render_process_num
        self._queue = mp.Queue(maxsize=self._room_num * 2)
        self._render_process_random_seed = int(np.random.randint(2 ** 31))

        self._p = mp.Process(target=self._worker, daemon=True)
        self._p.start()
    
    def sync(self):
        self._queue.put("sync")
    
    def join(self):
        self._queue.put("sync")
        self._queue.put("join")
    
    @env_timer.timer
    def send_message(self, message: dict[Literal["input_folder", "output_dir", "step_idx"], Any]):
        self._queue.put(message)
    
    @staticmethod
    def relpath(p: str):
        """relative path to the origin working directory"""
        return os.path.relpath(utils.get_path_handler()(p), utils.get_path_handler()("."))

    def _generate_init_data(self):
        return dict(
            cloth_obj_path=self.relpath(self._env_cfg.cloth_obj_path),
            seed=self._render_process_random_seed,
            cameras_name=[n for n in self._env_cfg.render_mode if n != "mesh"],
        )

    def _prepare_socket_input(self):
        inputs = []
        for i in range(self._room_num):
            data = self._queue.get()
            if data == "sync":
                inputs = inputs + (self._room_num - len(inputs)) * [None]
                break
            elif data == "join":
                assert len(inputs) == 0
                inputs = "exit"
                break
            else:
                inputs.append(data)
        return inputs

    def _worker(self):
        tmp_filename = os.path.abspath(os.path.join(self.tmp_dir, f"{os.getpid()}.log"))
        os.makedirs(os.path.dirname(tmp_filename), exist_ok=True)
        tmp_file = open(tmp_filename, "w")
        def myprint(msg: str):
            print(msg, flush=True, file=tmp_file)

        myprint(f"tmp_file:{tmp_file} created")
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', 0))
        server_socket.listen(1)
        host, port = server_socket.getsockname()
        myprint(f"Server listening on {host}:{port}")
    
        gpuids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if len(gpuids) > 1: 
            print("[WARN] multiple gpus are selected, but only one will be used for isaac rendering")
        gpuid = int(gpuids[0])
        docker_dir = "/root/garmentds/"
        command = (
            f"docker start isaac-sim-450-{gpuid} && docker exec -d -w {docker_dir} isaac-sim-450-{gpuid} bash -c '"
            f"/isaac-sim/python.sh src/garmentds/foldenv/isaacsim_utils/isaacsim_socket.py "
            f"--port {port} --host {host} --roomnum {self._room_num} --uid {os.getuid()} --gid {os.getgid()} "
            f"> {self.relpath(tmp_filename) + '.2'} 2>&1'"
        )
        subprocess.run(command, shell=True)

        myprint("Waiting for connection from container...")
        client_socket, _ = server_socket.accept()
        client_socket = client_socket
        myprint("Connected to container")

        sendall(client_socket, json.dumps(self._generate_init_data()))
        recv_data = recvall(client_socket)
        assert recv_data == "init complete", recv_data
        while True:
            data = self._prepare_socket_input()
            if data == "exit":
                break

            sendall(client_socket, json.dumps(data))
            recv_data = recvall(client_socket)
            assert recv_data == "render complete", recv_data
            myprint(f"Received data from container: {recv_data}")
        
        sendall(client_socket, json.dumps("exit"))
        recv_data = recvall(client_socket)
        assert recv_data == "exit", recv_data
        myprint(f"Received exit signal from container: {recv_data}")
        
        myprint(f"close {client_socket.getsockname()}")
        client_socket.close()

        # close file
        tmp_file.close()
        if self._env_cfg.render_remove_tmp_files:
            os.remove(tmp_filename)


class RenderProcessBlender(RenderProcess):
    multiprocess_debug = False
    def __init__(
        self, 
        env_cfg: FoldEnvCfg, 
        sleep_time: float = 0.1, 
        timeout_message_sec: Optional[float] = None, # DEBUG
        timeout_join_worker_sec: Optional[float] = None, # DEBUG
        timeout_join_blender_sec: Optional[float] = 20., # DEBUG
    ):
        process_num = env_cfg.render_process_num
        if env_cfg.render_blender_debug_level == 0:
            pass # no debug
        elif env_cfg.render_blender_debug_level == 1:
            process_num = 1 # only one process
        else:
            raise ValueError(f"Invalid render_debug_level {env_cfg.render_blender_debug_level}")
        
        self._env_cfg = copy.deepcopy(env_cfg)
        self._picker_num = 3

        self._sleep_time = float(sleep_time)
        self._process_num = int(process_num)
        self._render_process_random_seed = int(np.random.randint(2 ** 31))
        self._timeout_message_sec = timeout_message_sec
        self._timeout_join_worker_sec = timeout_join_worker_sec
        self._timeout_join_blender_sec = timeout_join_blender_sec
        atexit.register(self.terminate)

        visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if self._env_cfg.render_blender_skip_first_cuda:
            is_running_queue_size = (len(visible_devices) - 1) * self._process_num
        else:
            is_running_queue_size = len(visible_devices) * self._process_num

        self._queue = mp.Queue()
        self._is_running_queue = mp.Queue(maxsize=is_running_queue_size)
        self._process_list: list[mp.Process] = []
        for process_idx in range(process_num):
            process = mp.Process(target=self._worker, args=(process_idx, ))
            process.start()
            self._process_list.append(process)
    
    def _sleep(self):
        time.sleep(np.random.rand() * self._sleep_time) # random sleep to avoid all process run at the same time

    def _worker(self, process_idx: int):
        def worker_print(*args, **kwargs):
            if not self.multiprocess_debug:
                return
            print(f"[DEBUG] worker {process_idx}: ", end="")
            kwargs["flush"] = True
            print(*args, **kwargs)
        
        worker_print(f"pid: {os.getpid()}, start")
        blender_py_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "blender_script.py")
        blender_scene_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "scene.blend")
        
        visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if self._env_cfg.render_blender_skip_first_cuda:
            cuda_id = visible_devices[process_idx % (len(visible_devices) - 1) + 1]
        else:
            cuda_id = visible_devices[process_idx % len(visible_devices)]
        
        # launch blender process
        tmp_filename = os.path.join(self.tmp_dir, f"blender/{os.getpid()}.log")
        os.makedirs(os.path.dirname(tmp_filename), exist_ok=True)
        tmp_file = open(tmp_filename, "w")
        subp = subprocess.Popen(
            f"CUDA_VISIBLE_DEVICES={cuda_id} "
            + f"{self._env_cfg.render_blender_command} {blender_scene_path} --background -noaudio --python {blender_py_path} --", 
            shell=True, text=True, 
            stdin=subprocess.PIPE, stdout=tmp_file, stderr=tmp_file, 
        )

        enc = json.encoder.JSONEncoder()
        init_message = dict(
            urdf_path=self._env_cfg.robot_cfg.urdf_path, 
            mesh_dir=self._env_cfg.robot_cfg.mesh_dir,
            picker_num=self._picker_num, 
            picker_radius=0.02, 
            use_collision_instead_of_visual=False, 
            seed=self._render_process_random_seed, 
            engine=self._env_cfg.render_blender_engine, 
            camera_type=self._env_cfg.render_blender_camera_type, 
            cloth_obj_path=self._env_cfg.cloth_obj_path,
            hide_picker=self._env_cfg.render_blender_hide_picker,
            camera_size_level=self._env_cfg.render_blender_camera_size_level,
            render_set=self._env_cfg.render_set,
        )
        subp.stdin.write(enc.encode(init_message) + "\n")
        subp.stdin.flush()
        
        init_message_path = os.path.join(".render_process", f"init_message_{process_idx}.json")
        os.makedirs(os.path.dirname(init_message_path), exist_ok=True)
        with open(init_message_path, "w") as f:
            json.dump(init_message, f, indent=4)
        
        import signal
        class SIGUSR1Exception(Exception): 
            pass
        def handle_sigusr1(signum, frame): 
            raise SIGUSR1Exception
        assert signal.getsignal(signal.SIGUSR1) == signal.SIG_DFL
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        
        while True:
            worker_print(f"waiting for message ...")
            message = self._queue.get()
            worker_print(f"got message: {pprint.pformat(message, sort_dicts=False)}")
            if message["action"] == "join":
                subp.stdin.write(enc.encode(message) + "\n")
                worker_print(f"join message sent")
                subp.stdin.flush()
                worker_print(f"stdin flushed")
                try:
                    subp.wait(self._timeout_join_blender_sec)
                except subprocess.TimeoutExpired:
                    subp.kill()
                    print(f"subp killed because of timeout")
                worker_print(f"blender process terminated")
                break
            elif message["action"] == "message":
                # remove exist png file
                output_png = message["message"]["output_png"]
                if os.path.exists(output_png):
                    os.remove(output_png)
                
                try:
                    # send render message
                    message = copy.deepcopy(message)
                    message["message"]["target_pid"] = os.getpid()
                    subp.stdin.write(enc.encode(message) + "\n")
                    subp.stdin.flush()
                    while True:
                        self._sleep()
                        assert subp.poll() is None # assert blender process is still running
                except SIGUSR1Exception as e: # wait for render finish
                    pass
                self._is_running_queue.get()
            else:
                raise ValueError("Invalid message")
            self._sleep()
        
        # close file
        tmp_file.close()
        worker_print(f"worker {process_idx} tmp file closed")
        if self._env_cfg.render_remove_tmp_files:
            os.remove(tmp_filename)
        worker_print(f"worker {process_idx} done")

    @env_timer.timer
    def send_message(self, message):
        for p in self._process_list:
            assert p.is_alive()
        self._is_running_queue.put(None)
        self._queue.put(dict(action="message", message=message))
    
    def join(self):
        for idx in range(self._process_num):
            self._queue.put(dict(action="join"))
        for process_idx, p in enumerate(self._process_list):
            logger.info(f"join worker {process_idx}")
            p.join(timeout=self._timeout_join_worker_sec)
            if p.is_alive():
                p.terminate()
                logger.info(f"worker {process_idx} terminated because of timeout")
    
    def sync(self):
        logger.info("sync start")
        while self._is_running_queue.qsize() > 0:
            self._sleep()
        logger.info("sync end")
    
    def terminate(self):
        pid = os.getpid()
        process = psutil.Process(pid)
        children = process.children(recursive=True)
        for child in children:
            if child.is_running():
                child.terminate()


class FlexCoord:
    FLEX_TO_WORLD = tra.euler_matrix(np.pi / 2, 0, 0)[:3, :3]

    @staticmethod
    def flex_to_world(pos):
        return pos @ FlexCoord.FLEX_TO_WORLD.T

    @staticmethod
    def world_to_flex(pos):
        return pos @ FlexCoord.FLEX_TO_WORLD


def get_xyz_center(xyz: np.ndarray):
    return (xyz.min(axis=0) + xyz.max(axis=0)) / 2.


def load_mesh_raw_sim(path: str, scale: float, name="cloth"):
    tm_mesh_raw: trimesh.Trimesh = trimesh.load_mesh(path, process=False)
    dim_range = tm_mesh_raw.bounds[1] - tm_mesh_raw.bounds[0]
    apply_scale = scale / ((dim_range[0] * dim_range[1]) ** 0.5)
    translation = -get_xyz_center(tm_mesh_raw.vertices)
    logger.info(f"apply_translation:{translation}, then apply_scale:{apply_scale} ")
    tm_mesh_raw.apply_translation(translation)
    tm_mesh_raw.apply_scale(apply_scale)
    logger.info(f"mesh {path}:\nafter transform bounds:{tm_mesh_raw.bounds}")
    tm_mesh_sim = trimesh.Trimesh(vertices=tm_mesh_raw.vertices, faces=tm_mesh_raw.faces, process=True)
    tm_mesh_raw.metadata["name"] = name
    tm_mesh_sim.metadata["name"] = name

    with open(path[:-len(".obj")] + "_info.json", "r") as f:
        info = json.load(f)
        keypoint_idx: dict[str, int] = {
            k: v[0] for k, v in info["triangulation"]["keypoint_idx"].items()
        }
        
    return tm_mesh_raw, tm_mesh_sim, keypoint_idx


class Picker:
    OPEN = 0
    CLOSE = 1
    def __init__(
        self, 
        env: 'FoldEnv', 
        name: str, 
        grasp_threshold: tuple[float, float, float], 
        squeeze_factor: tuple[float, float, float]
    ):
        self._env = env
        self._name = name
        self._tf = np.eye(4)
        self._val = self.OPEN
        self._val_float = float(self._val)
        self._prev_val = self.OPEN

        self._grasp: Optional[dict[str, np.ndarray]] = None
        """
        - vid: [N], int
        - xyz_offset: [N, 3], float
        - old_mass_inv: [N], float
        """
        self._grasp_threshold = np.array(grasp_threshold)
        self._squeeze_factor = np.array(squeeze_factor)
        assert self._squeeze_factor.shape == (3,)
    
    @property
    def tf(self):
        return self._tf
    
    @property
    def val(self):
        return self._val
    
    @property
    def val_float(self):
        return self._val_float

    @staticmethod
    def is_open_action(action: float):
        """return action < 0.5"""
        return action < 0.5
    
    @staticmethod
    def is_close_action(action: float):
        """return action >= 0.5"""
        return action >= 0.5

    def compute_grasp_vertices(self, tf: np.ndarray):
        # get xyz and inverse mass
        xyzm = self._env._get_cloth_xyzm()
        xyz_world_frame, mass_inv = xyzm[:, :3], xyzm[:, 3]
        # grasp vert according to distance
        xyz_picker_frame = (np.concatenate([xyz_world_frame, np.ones((xyz_world_frame.shape[0], 1))], axis=1) @ np.linalg.inv(tf).T)[:, :3]
        vid = np.where(np.linalg.norm(xyz_picker_frame / self._grasp_threshold, axis=1) < 1.)[0]
        return vid, xyz_picker_frame, mass_inv

    def set_action(self, action: float):
        """0 is open and 1 is close"""
        self._prev_val = self._val
        self._val = self.OPEN if self.is_open_action(action) else self.CLOSE
        self._val_float = float(action)
        logger.info(f"picker [{self._name}] set action ({action}) : {self._prev_val} -> {self._val}")

        if (self._prev_val, self._val) == (self.OPEN, self.OPEN):
            pass
        elif (self._prev_val, self._val) == (self.OPEN, self.CLOSE):
            vid, xyz_picker_frame, mass_inv = self.compute_grasp_vertices(self._tf)
            xyz_offset = xyz_picker_frame[vid, :]
            old_mass_inv = mass_inv[vid]
            # store grasp info
            self._grasp = dict(vid=vid, xyz_offset=xyz_offset, old_mass_inv=old_mass_inv)
        elif (self._prev_val, self._val) == (self.CLOSE, self.OPEN):
            xyzm = self._env._get_cloth_xyzm()
            xyzm[self._grasp["vid"], 3] = self._grasp["old_mass_inv"]
            self._env._set_cloth_xyzm(xyzm)
            self._grasp = None
        elif (self._prev_val, self._val) == (self.CLOSE, self.CLOSE):
            pass
        else:
            raise ValueError("Invalid action")
    
    def step(self, tf: Optional[np.ndarray]=None):
        """update xyz and grasped vertices"""
        if tf is not None:
            self._tf[...] = tf
        if self._grasp is not None:
            xyzm = self._env._get_cloth_xyzm()
            vid, xyz_offset = self._grasp["vid"], self._grasp["xyz_offset"] * self._squeeze_factor
            xyzm[vid, :3] = (np.concatenate([xyz_offset, np.ones((xyz_offset.shape[0], 1))], axis=1) @ self._tf.T)[:, :3]
            xyzm[vid, 3] = self._grasp["old_mass_inv"] * (self.CLOSE - self._val_float) # smoothly change mass inverse
            self._env._set_cloth_xyzm(xyzm)
            self._env._set_cloth_vel(np.zeros((xyzm.shape[0], 3))) # robot velocity is approximated as zero
    
    def set_tf(self, tf: np.ndarray):
        """only set tf, do not update grasped vertices"""
        self._tf[...] = tf
    
    def current_grasp_nothing(self) -> bool:
        return self._val == self.CLOSE and self._grasp["vid"].shape == (0, )


class Robot:
    def __init__(self, picker_l: Picker, picker_r: Picker, cfg: RobotCfg):
        self._cfg = copy.deepcopy(cfg)
        self._urdf = urdf = batch_urdf.URDF(
            batch_size=1, urdf_path=cfg.urdf_path, dtype=torch.float32,
            device=cfg.device, mesh_dir=cfg.mesh_dir,
        )
        urdf.update_base_link_transformation(cfg.base_link, urdf.tensor([cfg.base_pos]))
        urdf.update_cfg(urdf.cfg_f2t(cfg.init_qpos))
        self._tensor = self._urdf.tensor
        self._t2f = self._urdf.cfg_t2f
        self._f2t = self._urdf.cfg_f2t

        # to compute leg joints
        self._initial_torso_base_link_tf = self._urdf.link_transform_map["torso_base_link"].clone()[0, ...]

        # picker
        self._picker: dict[Literal["left", "right"], Picker] = dict(left=picker_l, right=picker_r)

        # waypoint caches
        self._waypoints_qpos: deque[dict[str, float]] = deque()
        self._waypoints_picker: deque[dict[str, float]] = deque()

        # status caches
        self._ik_fail_count = 0
    
    @property
    def urdf(self):
        return self._urdf
    
    @property
    def picker(self):
        return self._picker
    
    @property
    def ik_fail_count(self):
        return self._ik_fail_count
    
    @property
    def waypoints_qpos(self):
        return self._waypoints_qpos
    
    def modify_gripper_qpos(self, qpos: dict[str, float], hand: Literal["left", "right"], action: float):
        for j in dict(left=self._cfg.gripper_l_joints, right=self._cfg.gripper_r_joints)[hand]:
            qpos[j] = (
                (action - Picker.OPEN) * self._cfg.gripper_close_val + 
                (Picker.CLOSE - action) * self._cfg.gripper_open_val
            ) / (Picker.CLOSE - Picker.OPEN)
        return qpos
    
    def _set_gripper_qpos(self, hand: Literal["left", "right"], action: float):
        action = np.clip(action, 0., 1.)
        qpos = self._t2f(self._urdf.cfg)
        qpos = self.modify_gripper_qpos(qpos, hand, action)
        self._urdf.update_cfg(self._f2t(qpos))

    @staticmethod
    def _xyz_avg(xyz_l: np.ndarray, xyz_r: np.ndarray):
        xyz = (xyz_l + xyz_r) / 2.
        def max_abs_avg(x, y):
            if (x * y) > 0.:
                return max(abs(x), abs(y)) * np.sign(x)
            else:
                return x + y
        xyz[1] = max_abs_avg(xyz_l[1], xyz_r[1])
        return xyz
    
    def _compute_target_pose_and_leg_joint4_to_solve_leg(self, xyz_l: np.ndarray, xyz_r: np.ndarray):
        targ_pose = self._initial_torso_base_link_tf.clone()
        cfg = self._cfg

        x, y, z = xyz = self._xyz_avg(xyz_l, xyz_r)
        logger.info(f"xyz_avg:{xyz} xyz_l:{xyz_l} xyz_r:{xyz_r}")

        # add threshold, when dy is not needed, we set dy = 0
        y_th_p_1, y_th_p_2 = cfg.ik_move_leg_py_th_pos
        y_th_n_1, y_th_n_2 = cfg.ik_move_leg_py_th_neg
        yn, yp = cfg.ik_move_leg_dy_range
        z_th_1, z_th_2 = cfg.ik_move_leg_pz_th
        if y > 0:
            coeff_y = np.clip((y - y_th_p_1) / (y_th_p_2 - y_th_p_1), 0., 1.) * yp
        else:
            coeff_y = np.clip((y_th_n_2 - y) / (y_th_n_2 - y_th_n_1), 0., 1.) * (-yn)
        coeff_z = np.clip((z_th_2 - z) / (z_th_2 - z_th_1), 0., 1.)
        dy = coeff_y * coeff_z
        targ_pose[1, 3] = self._initial_torso_base_link_tf[1, 3] + dy

        # add threshold, when leg_joint4 is not needed, we set leg_joint4 = 0
        if xyz_l is None or xyz_r is None:
            leg_joint4 = 0.
        else:
            yl, yr = xyz_l[1], xyz_r[1]
            disty_th_1, disty_th_2 = cfg.ik_move_leg_joint4_disty_th
            leg_joint4 = np.clip((abs(yl - yr) - disty_th_1) / (disty_th_2 - disty_th_1), 0., 1.) * cfg.ik_move_leg_joint4_max_val
            if (yl > yr):
                leg_joint4 *= -1.

        logger.info(f"dy:{dy} leg_joint4:{leg_joint4} coeff_y:{coeff_y} coeff_z:{coeff_z} xyz_avg:{xyz}")
        return targ_pose, leg_joint4
    
    def _solve_leg_joints(self, xyz_l: np.ndarray, xyz_r: np.ndarray):
        urdf = self._urdf
        cfg = self._cfg

        target_link = "torso_base_link"
        mask = urdf.tensor([1.] * 16)
        curr_pose = urdf.link_transform_map[target_link].clone()[0, ...]
        targ_pose, leg_joint4 = self._compute_target_pose_and_leg_joint4_to_solve_leg(xyz_l, xyz_r)
        logger.info(f"compute_target_pose_solve_leg\ncurr_pose:{curr_pose}\ntarg_pose:{targ_pose}")
        
        B = 1 # batch size
        def err_func(link_transform_map: dict[str, torch.Tensor]):
            curr_mat4 = link_transform_map[target_link].view(B, 16, 16) # [B, 16, 16]
            curr_mat16 = curr_mat4[:, torch.arange(16), torch.arange(16)] # [B, 16]
            err_mat = curr_mat16 - targ_pose.view(B, 16) # [B, 16]
            return err_mat * mask
        def loss_func(link_transform_map: dict[str, torch.Tensor]):
            curr_mat4 = link_transform_map[target_link] # [B, 4, 4]
            curr_mat16 = curr_mat4.view(B, 16) # [B, 16]
            err_mat = curr_mat16 - targ_pose.view(B, 16)
            return torch.sum(torch.square(err_mat) * mask, dim=1) # [B, ]

        qpos, info = urdf.inverse_kinematics_optimize(
            err_func=err_func, loss_func=loss_func, 
            fix_joint=[j for j in urdf.cfg.keys() if j not in cfg.leg_joints], 
            **cfg.ik_kwargs, 
        )
        qpos["leg_joint4"][...] = leg_joint4
        qpos["head_joint1"][...] = -leg_joint4 # keep head not rotate
        if info["iter_idx"] == self._cfg.ik_kwargs["max_iter"] - 1:
            self._ik_fail_count += 1
        logger.info(f"IK solve leg joints xyz_l:{xyz_l} xyz_r:{xyz_r}\ninfo:{info}")
        return {k: v for k, v in self._t2f(qpos).items()}

    def set_target_xyz(
        self, 
        steps: int, 
        xyz_l: Optional[np.ndarray]=None, 
        xyz_r: Optional[np.ndarray]=None,
    ):
        cfg = self._cfg
        self._waypoints_qpos.clear()
        info_all = {}
        curr_qpos = self._t2f(self._urdf.cfg)

        if xyz_l is None and xyz_r is None:
            targ_qpos = {k: v for k, v in curr_qpos.items()}
        
        else:
            if xyz_l is None:
                xyz_l = utils.torch_to_numpy(self._urdf.link_transform_map[cfg.tcp_l])[0, :3, 3]
            if xyz_r is None:
                xyz_r = utils.torch_to_numpy(self._urdf.link_transform_map[cfg.tcp_r])[0, :3, 3]
            
            # set leg joints
            if cfg.ik_move_leg:
                targ_qpos = self._solve_leg_joints(xyz_l, xyz_r)
            else:
                targ_qpos = {k: v for k, v in curr_qpos.items()}

            # set ik init solution
            smooth_coeff = 0.9
            smooth_joints = set(cfg.arm_l_joints + cfg.arm_r_joints)
            ik_init_cfg = self._f2t(targ_qpos)
            for k in ik_init_cfg.keys():
                if k in smooth_joints:
                    ik_init_cfg[k] = smooth_coeff * ik_init_cfg[k] + (1 - smooth_coeff) * self._cfg.ik_init_qpos[k]
            
            # solve ik
            logger.info(f"IK xyz_l:{xyz_l} xyz_r:{xyz_r}")
            for hand in ["left", "right"]:
                xyz = dict(left=xyz_l, right=xyz_r)[hand]
                pose = np.eye(4)
                pose[:3, 3] = xyz
                mask1 = self._tensor([0., 0., 0., 1.] * 2 + [1., 0., 0., 1.] + [0., 0., 0., 1.]) # e_x' dot e_z = 0
                d1, z1, d2, z2 = cfg.ik_gripper_z_deg_max
                z_deg_max = d1 + (d2 - d1) * np.clip((xyz[2] - z1) / (z2 - z1), 0., 1.)
                mask2_lt_target_tensor = self._tensor([-torch.inf] * 10 + [np.cos(np.deg2rad(z_deg_max))] + [-torch.inf] * 5) # e_z' dot e_z > cos(z_deg_max)
                pose = self._tensor(pose) # [4, 4]
                tcp_str = dict(left=cfg.tcp_l, right=cfg.tcp_r)[hand]
                B = 1 # batch size
                def compute_mask(curr_mat16: torch.Tensor):
                    mask = (mask1 + (curr_mat16 < mask2_lt_target_tensor).float())
                    return mask
                def err_func(link_transform_map: dict[str, torch.Tensor]):
                    curr_mat4 = link_transform_map[tcp_str].view(B, 16, 16) # [B, 16, 16]
                    curr_mat16 = curr_mat4[:, torch.arange(16), torch.arange(16)] # [B, 16]
                    err_mat = curr_mat16 - pose.view(B, 16) # [B, 16]
                    return err_mat * compute_mask(curr_mat16)
                def loss_func(link_transform_map: dict[str, torch.Tensor]):
                    curr_mat4 = link_transform_map[tcp_str] # [B, 4, 4]
                    curr_mat16 = curr_mat4.view(B, 16) # [B, 16]
                    err_mat = curr_mat16 - pose.view(B, 16)
                    return torch.sum(torch.square(err_mat) * compute_mask(curr_mat16), dim=1) # [B, ]

                qpos, info = self._urdf.inverse_kinematics_optimize(
                    err_func=err_func, loss_func=loss_func, init_cfg=ik_init_cfg, 
                    fix_joint=[j for j in self._urdf.cfg.keys() if j not in dict(left=cfg.arm_l_joints, right=cfg.arm_r_joints)[hand]], 
                    **cfg.ik_kwargs,
                )
                if info["iter_idx"] == self._cfg.ik_kwargs["max_iter"] - 1:
                    self._ik_fail_count += 1
                qpos = self._t2f(qpos)
                info_all[hand] = info
                logger.info(f"IK {hand} {pprint.pformat(info, sort_dicts=False)}")

                for j in dict(left=cfg.arm_l_joints, right=cfg.arm_r_joints)[hand]:
                    targ_qpos[j] = qpos[j]
            logger.info(f"IK set_target_xyz target_qpos:\n{pprint.pformat(targ_qpos, sort_dicts=False)}")

        # linear interpolation
        for i in range(steps):
            self._waypoints_qpos.append({
                k: curr_qpos[k] + (targ_qpos[k] - curr_qpos[k]) / steps * (i + 1) for k in targ_qpos.keys()
            })
        
        return info_all
    
    def set_target_picker(
        self, 
        steps: int, 
        picker_l: Optional[float]=None, 
        picker_r: Optional[float]=None,
    ):
        self._waypoints_picker.clear()
        curr_l = self._picker["left"].val_float
        curr_r = self._picker["right"].val_float
        targ_l = curr_l if picker_l is None else picker_l
        targ_r = curr_r if picker_r is None else picker_r
        for i in range(steps):
            self._waypoints_picker.append(dict(
                left=curr_l + (targ_l - curr_l) / steps * (i + 1), 
                right=curr_r + (targ_r - curr_r) / steps * (i + 1)
            ))
    
    @env_timer.timer
    def step(self):
        """update robot qpos, picker xyz, and grasped vertices"""
        if len(self._waypoints_qpos) > 0:
            wp = self._waypoints_qpos.popleft()
            self.set_qpos(wp, False)
        if len(self._waypoints_picker) > 0:
            wp = self._waypoints_picker.popleft()
            self.set_picker(left=wp["left"], right=wp["right"])

        for hand in ["left", "right"]:
            tcp_str = dict(left=self._cfg.tcp_l, right=self._cfg.tcp_r)[hand]
            tcp_tf = utils.torch_to_numpy(self._urdf.link_transform_map[tcp_str])[0, :, :]
            self._picker[hand].step(tcp_tf)
    
    def set_qpos(self, qpos: dict[str, float], exclude_gripper_joints: bool):
        """set robot qpos, picker xyz (using fk), do not update grasped vertices"""
        if not exclude_gripper_joints:
            self._urdf.update_cfg(self._f2t(qpos))
        else:
            curr_qpos = self._t2f(self._urdf.cfg)
            exclude_joints = set(self._cfg.gripper_l_joints + self._cfg.gripper_r_joints)
            next_qpos = {k: v if v not in exclude_joints else curr_qpos[k] for k, v in qpos.items()}
            self._urdf.update_cfg(self._f2t(next_qpos))
        for hand in ["left", "right"]:
            tcp_str = dict(left=self._cfg.tcp_l, right=self._cfg.tcp_r)[hand]
            tcp_tf = utils.torch_to_numpy(self._urdf.link_transform_map[tcp_str])[0, :, :]
            self._picker[hand].set_tf(tcp_tf)

    def set_picker(self, left: Optional[float]=None, right: Optional[float]=None):
        if left is not None:
            self._picker["left"].set_action(left)
            self._set_gripper_qpos("left", left)
        if right is not None:
            self._picker["right"].set_action(right)
            self._set_gripper_qpos("right", right)

    def get_qpos(self):
        return self._t2f(self._urdf.cfg)

    def get_base_pose(self):
        return utils.torch_to_numpy(self._urdf.link_transform_map[self._cfg.base_link][0, ...])

    def get_base_link(self):
        return self._cfg.base_link

    def get_tcp_xyz(self) -> dict[Literal["left", "right"], np.ndarray]:
        tcp_xyz = {}
        for hand in ["left", "right"]:
            tcp_xyz[hand] = utils.torch_to_numpy(self._urdf.link_transform_map[dict(left=self._cfg.tcp_l, right=self._cfg.tcp_r)[hand]][0, :3, 3])
        return tcp_xyz
    
    def get_gripper_state(self) -> dict[Literal["left", "right"], float]:
        return dict(left=self._picker["left"].val_float, right=self._picker["right"].val_float)
    
    def get_gripper_state_int(self) -> dict[Literal["left", "right"], int]:
        return dict(left=self._picker["left"].val, right=self._picker["right"].val)
    
    def get_mesh(self):
        return self._urdf.get_scene(0, True).to_geometry()

    def reset(self):
        self.set_qpos(self._cfg.init_qpos, False)
        self.set_picker(Picker.OPEN, Picker.OPEN)
        self._waypoints_qpos.clear()
        self._waypoints_picker.clear()
        self._ik_fail_count = 0


@dataclass
class FoldEnvState:
    step_idx: int = field(init=False)
    substep_idx: int = field(init=False)
    render_frame_idx: int = field(init=False)

    def __post_init__(self):
        self.step_idx = 0
        self.substep_idx = 0
        self.render_frame_idx = 0
    
    def reset(self):
        self.step_idx = 0
        self.substep_idx = 0
        self.render_frame_idx = 0


class FoldEnv:
    _default_height = 0.3
    def __init__(self, cfg: FoldEnvCfg):
        cfg = copy.deepcopy(cfg)
        self._cfg = cfg
        self._state = FoldEnvState()
        
        self._domain_randomize(cfg)
        self._load_cloth(cfg)
        self._init_pyflex(cfg)
        self._init_env(cfg)
        self._init_cloth(cfg)
        self._init_robot(cfg)
        self._init_cache()
        
        RP = dict(blender=RenderProcessBlender, isaacsim=RenderProcessIsaacsim)[cfg.render_method]
        self._render_process = RP(cfg)

    ### init ###
    def _domain_randomize(self, cfg: FoldEnvCfg):
        logger.info(f"Before Domain Randomization:\n{pprint.pformat(asdict(cfg), sort_dicts=False)}")
        dr = cfg.dr_args
        if dr is None:
            dr = dict()
        def apply_dr(v: float, a: float, b: float):
            return v * random.uniform(a, b)
        for k, (a, b) in dr.items():
            old_v = getattr(cfg, k)
            if isinstance(old_v, (list, tuple, np.ndarray)):
                new_v = tuple(apply_dr(v, a, b) for v in old_v)
            else:
                new_v = apply_dr(float(old_v), a, b)
            setattr(cfg, k, new_v)
        logger.info(f"After Domain Randomization:\n{pprint.pformat(asdict(cfg), sort_dicts=False)}")

    def _load_cloth(self, cfg: FoldEnvCfg):
        # load cloth mesh to simulate
        tm_mesh_raw, tm_mesh_sim, keypoint_idx = load_mesh_raw_sim(cfg.cloth_obj_path, cfg.cloth_scale)
        self._tm_mesh_raw = tm_mesh_raw # volatile
        self._tm_mesh_sim = tm_mesh_sim # volatile
        self._tm_mesh_raw_rest = copy.deepcopy(tm_mesh_raw) # keep constant
        self._tm_mesh_sim_rest = copy.deepcopy(tm_mesh_sim) # keep constant
        self._keypoint_idx = keypoint_idx
        logger.info(f"Cloth mesh loaded, vert num: {tm_mesh_raw.vertices.shape[0]} {tm_mesh_sim.vertices.shape[0]}")

    def _init_pyflex(self, cfg: FoldEnvCfg):
        tm_mesh_sim_rest = self._tm_mesh_sim_rest
        vert_num = tm_mesh_sim_rest.vertices.shape[0]
        if cfg.collision_radius is None:
            edge_len = np.linalg.norm(
                tm_mesh_sim_rest.vertices[tm_mesh_sim_rest.edges[:, 1]] - 
                tm_mesh_sim_rest.vertices[tm_mesh_sim_rest.edges[:, 0]], axis=1
            )
            collision_radius = edge_len.mean() * 1.5
            logger.info(f"collision_radius is set to {collision_radius} based on cloth mesh (mean: {edge_len.mean()}, max: {edge_len.max()}, min: {edge_len.min()})")
        else:
            collision_radius = cfg.collision_radius

        scene_cfg = dict(
            numExtraParticles=vert_num,
            radius=collision_radius,
            dynamicFriction=cfg.dynamicFriction,
            particleFriction=cfg.particleFriction,
            dt=cfg.sim_dt_pyflex, 
            numIterations=cfg.numSolverIterations, 
            numSubsteps=cfg.numSolverSubsteps, 
            gravity=9.8,
        )
        scene_id = 0

        pyflex.init(True, False, 0, 0, 0)
        pyflex.set_scene(scene_id, scene_cfg)

    def _init_cloth(self, cfg: FoldEnvCfg):
        tm_mesh_raw_rest, tm_mesh_sim_rest = self._tm_mesh_raw_rest, self._tm_mesh_sim_rest

        # deal with repeated vertices
        vert_xyz_to_idx = {}
        for i, v in enumerate(tm_mesh_sim_rest.vertices):
            vert_xyz_to_idx[tuple(v)] = i
        vert_ren_to_sim = []
        for i, v in enumerate(tm_mesh_raw_rest.vertices):
            vert_ren_to_sim.append(vert_xyz_to_idx[tuple(v)])
        self._vert_ren_to_sim = np.array(vert_ren_to_sim)
        """[NV_R] -> [0, NV_S)"""

        # add edges
        stretch_edges, bend_edges = set(), set()
    
        ## Stretch & Shear
        for face in tm_mesh_sim_rest.faces:
            stretch_edges.add(tuple(sorted([face[0], face[1]])))
            stretch_edges.add(tuple(sorted([face[1], face[2]])))
            stretch_edges.add(tuple(sorted([face[2], face[0]])))
        
        ## Bend
        neighbours: dict[int, set[int]] = dict()
        for vid in range(tm_mesh_sim_rest.vertices.shape[0]):
            neighbours[vid] = set()
        for edge in stretch_edges:
            neighbours[edge[0]].add(edge[1])
            neighbours[edge[1]].add(edge[0])
        for vid in range(tm_mesh_sim_rest.vertices.shape[0]):
            neighbour_list = list(neighbours[vid])
            N = len(neighbour_list)
            for i in range(N - 1):
                for j in range(i + 1, N):
                    bend_edge = tuple(sorted([neighbour_list[i], neighbour_list[j]]))
                    if bend_edge not in stretch_edges:
                        bend_edges.add(bend_edge)

        # add cloth to pyflex
        pyflex.add_cloth_mesh(
            position=np.array([0., 0., 0.]), 
            verts=FlexCoord.world_to_flex(tm_mesh_sim_rest.vertices).reshape(-1), 
            faces=tm_mesh_sim_rest.faces.reshape(-1), 
            stretch_edges=np.array(list(stretch_edges)).reshape(-1), 
            bend_edges=np.array(list(bend_edges)).reshape(-1), 
            shear_edges=np.array([]).reshape(-1), 
            stiffness=np.array([cfg.cloth_stiff_stretch, cfg.cloth_stiff_bend, 0.0]), # no shear edges
            mass=cfg.cloth_mass, 
            uvs=np.array([[1.0, 1.0, 1.0]]).repeat(tm_mesh_sim_rest.vertices.shape[0], axis=0),
        )

        # meta info
        self._cloth_xyz_init = tm_mesh_sim_rest.vertices + np.array([0., 0., self._default_height])
    
    def _init_robot(self, cfg: FoldEnvCfg):
        self._robot = Robot(self._new_picker("left"), self._new_picker("right"), cfg.robot_cfg)
    
    def _init_env(self, cfg: FoldEnvCfg):
        self._picker_dict: dict[str, Picker] = {}
        self._picker_init_cloth = self._new_picker("init_cloth")
        if self._cfg.use_tqdm:
            self._tqdmer = tqdm.tqdm(desc="Fold Env Step", dynamic_ncols=True)
    
    def _new_picker(self, name: str):
        picker = Picker(self, name, grasp_threshold=self._cfg.grasp_threshold[name], squeeze_factor=self._cfg.grasp_squeeze_factor[name])
        if name in self._picker_dict:
            raise ValueError(f"Picker with name {name} already exists")
        self._picker_dict[name] = picker
        return picker

    def _init_cache(self):
        self._cache = dict(
            before_step=self._robot.get_gripper_state_int(),
            after_step=self._robot.get_gripper_state_int(),
        )
    
    @property
    def eps(self):
        return 1e-6
    
    ### pyflex helper ###
    def _get_cloth_xyzm(self) -> np.ndarray:
        """[NV, 4]"""
        xyzm = pyflex.get_positions().reshape((-1, 4))
        xyzm[:, :3] = FlexCoord.flex_to_world(xyzm[:, :3])
        return xyzm
    
    def _set_cloth_xyzm(self, xyzm: np.ndarray):
        xyzm[:, :3] = FlexCoord.world_to_flex(xyzm[:, :3])
        pyflex.set_positions(xyzm.reshape((-1, 4)))

    def _get_cloth_xyz(self) -> np.ndarray:
        """[NV, 3]"""
        return self._get_cloth_xyzm()[:, :3]
    
    def _set_cloth_xyz(self, xyz: np.ndarray):
        xyzm = pyflex.get_positions().reshape((-1, 4))
        xyzm[:, :3] = xyz
        self._set_cloth_xyzm(xyzm)
    
    def _set_cloth_vel(self, vel: np.ndarray):
        pyflex.set_velocities(FlexCoord.world_to_flex(vel.reshape((-1, 3))))
    
    def _get_cloth_vel(self) -> np.ndarray:
        """[NV, 3]"""
        return FlexCoord.flex_to_world(pyflex.get_velocities().reshape((-1, 3)))

    ### render ###
    def _get_curr_render_mesh(self) -> trimesh.Trimesh:
        vert_sim = self._get_cloth_xyz()
        self._tm_mesh_sim.vertices = vert_sim
        tm_mesh_raw = copy.deepcopy(self._tm_mesh_raw)
        tm_mesh_raw.vertices = self._tm_mesh_sim.vertices[self._vert_ren_to_sim]
        tm_mesh_raw.vertex_normals = self._tm_mesh_sim.vertex_normals[self._vert_ren_to_sim]
        return tm_mesh_raw

    def set_render_output(self, output_dir: str):
        self._cfg.render_output_dir = output_dir
    
    def get_render_output(self):
        return self._cfg.render_output_dir
    
    def get_render_mode(self):
        return self._cfg.render_mode
    
    def get_render_fps(self):
        return 1. / (self._cfg.sim_dt_robot * self._cfg.render_per_n_substep)
    
    def sync(self):
        self._render_process.sync()

    ### policy ###
    @property
    def PICKER_OPEN(self):
        return Picker.OPEN
    
    @property
    def PICKER_CLOSE(self):
        return Picker.CLOSE
    
    def get_tcp_xyz(self):
        return self._robot.get_tcp_xyz()
    
    def get_gripper_state(self):
        return self._robot.get_gripper_state()
    
    def get_keypoint_idx(self) -> dict[str, int]:
        return copy.deepcopy(self._keypoint_idx)
    
    def get_raw_mesh_curr(self):
        return self._get_curr_render_mesh()

    def get_raw_mesh_rest(self):
        return copy.deepcopy(self._tm_mesh_raw)
    
    def get_robot_mesh(self):
        return self._robot.get_mesh()

    @env_timer.timer
    def _export_v(self, output_dir: str):
        """only export vertices, the implemetation is efficient"""
        vert_sim = self._get_cloth_xyz()
        np.save(os.path.join(output_dir, "cloth_v_sim.npy"), vert_sim)

    @env_timer.timer
    def _step_pyflex(self):
        pyflex.step()
    
    @env_timer.timer
    def _step_robot(self):
        self._robot.step()
    
    @env_timer.timer
    def _step_render_isaacsim(self, frame_idx: int):
        cfg = self._cfg
        input_folder = os.path.join(cfg.render_output_dir, ".tmp", f"{frame_idx:04d}")
        os.makedirs(input_folder, exist_ok=True)

        with open(os.path.join(input_folder, "cfg.json"), "w") as f:
            json.dump(dict(
                qpos=self._robot.get_qpos(),
                base_pose=self._robot.get_base_pose().tolist(),
                base_link=self._robot.get_base_link(),
            ), f, indent=4)
        self._export_v(input_folder)

        for mode in cfg.render_mode:
            if mode == "mesh":
                mesh = self._get_curr_render_mesh()
                vertex_colors = np.ones_like(mesh.vertices, dtype=np.uint8) * 255
                vertex_colors[np.array([v for v in self._keypoint_idx.values()])] = [255, 0, 0]
                output_path = os.path.join(cfg.render_output_dir, mode, f"{frame_idx:04d}.obj")
                trimesh.util.concatenate([
                    trimesh.Trimesh(mesh.vertices, mesh.faces, vertex_colors=vertex_colors),
                    self._robot.get_mesh(), 
                    trimesh.primitives.Box(extents=[0.8, 0.8, 0.1], transform=tra.translation_matrix([0., 0., -0.05]))
                ]).export(output_path)
            elif mode in ["head", "side"]:
                continue
            else:
                raise NotImplementedError(f"Unsupported render mode {mode}")
        self._render_process.send_message(dict(
            input_folder=RenderProcessIsaacsim.relpath(os.path.abspath(input_folder)), 
            output_dir=RenderProcessIsaacsim.relpath(os.path.abspath(cfg.render_output_dir)), 
            step_idx=frame_idx,
        ))
    
    @env_timer.timer
    def _step_render_blender(self, frame_idx: int):
        cfg = self._cfg
        input_folder = os.path.join(cfg.render_output_dir, ".tmp", f"{frame_idx:04d}")
        os.makedirs(input_folder, exist_ok=True)

        with open(os.path.join(input_folder, "cfg.json"), "w") as f:
            json.dump(dict(
                qpos=self._robot.get_qpos(),
                base_pose=self._robot.get_base_pose().tolist(),
                base_link=self._robot.get_base_link(),
            ), f, indent=4)
        self._export_v(input_folder)

        for mode in cfg.render_mode:
            os.makedirs(os.path.join(cfg.render_output_dir, mode), exist_ok=True)
            output_png = os.path.join(cfg.render_output_dir, mode, f"{frame_idx:04d}.png")
            if mode == "head":
                picker_xyz = [None for p in self._picker_dict.values()]
            elif mode == "side":
                picker_xyz = [p.tf[:3, 3].tolist() for p in self._picker_dict.values()]
            elif mode == "mesh":
                mesh = self._get_curr_render_mesh()
                vertex_colors = np.ones_like(mesh.vertices, dtype=np.uint8) * 255
                vertex_colors[np.array([v for v in self._keypoint_idx.values()])] = [255, 0, 0]
                trimesh.util.concatenate([
                    trimesh.Trimesh(mesh.vertices, mesh.faces, vertex_colors=vertex_colors),
                    self._robot.get_mesh(), 
                    trimesh.primitives.Box(extents=[0.8, 0.8, 0.1], transform=tra.translation_matrix([0., 0., -0.05]))
                ]).export(output_png[:-4] + ".obj")
            else:
                raise NotImplementedError(f"Unsupported render mode {mode}")
            if mode in ["head", "side"]:
                self._render_process.send_message((dict(
                    input_folder=os.path.abspath(input_folder), 
                    output_png=os.path.abspath(output_png), 
                    mode=mode, 
                    picker_xyz=picker_xyz,
                    save_npy_img=True, 
                    save_blend_file=cfg.render_blender_save_blend and frame_idx == 0, 
                )))
    
    def _step_render(self, *args, **kwargs):
        return dict(blender=self._step_render_blender, isaacsim=self._step_render_isaacsim)[self._cfg.render_method](*args, **kwargs)

    def render(self):
        cfg, state = self._cfg, self._state
        if cfg.render:
            self._step_render(state.render_frame_idx)
            self._state.render_frame_idx += 1
    
    def step(
        self, 
        xyz_l: Optional[np.ndarray]=None, xyz_r: Optional[np.ndarray]=None, 
        picker_l: Optional[float]=None, picker_r: Optional[float]=None,
        overwrite_render: Optional[bool]=None,
        callback: Optional[Callable[[], None]]=None,
    ):
        """
        set robot target, call pyflex.step 'n_substeps' times, update robot and its picker, render if needed.
        """
        cfg, state = self._cfg, self._state
        logger.info(f"current_step:{state.step_idx}")
        logger.info(f"execute action: xyz_l={xyz_l}, xyz_r={xyz_r}, picker_l={picker_l}, picker_r={picker_r}")
        self._cache["before_step"] = self._robot.get_gripper_state_int()

        self._robot.set_target_xyz(steps=cfg.n_substep, xyz_l=xyz_l, xyz_r=xyz_r)
        self._robot.set_target_picker(steps=cfg.n_substep, picker_l=picker_l, picker_r=picker_r)
        for _ in range(cfg.n_substep):
            self._step_robot()
            for __ in range(cfg.n_pyflex_steps_per_substep):
                self._step_pyflex()
            if callback is not None:
                callback()
            state.substep_idx += 1
            if state.substep_idx % cfg.render_per_n_substep == 0:
                if (cfg.render if overwrite_render is None else overwrite_render):
                    if not cfg.render and bool(overwrite_render):
                        raise ValueError("cfg.render is set to False but overwrite_render is True")
                    self._step_render(state.render_frame_idx)
                state.render_frame_idx += 1
        
        state.step_idx += 1
        if self._cfg.use_tqdm:
            self._tqdmer.update()
        self._cache["after_step"] = self._robot.get_gripper_state_int()
    
    def _random_init_cloth_generate_random_velocities(self, xyz: np.ndarray):
        xyz = (xyz - xyz.min(axis=0)) / (xyz.max(axis=0) - xyz.min(axis=0))

        DIM = 3
        n_freq = 100
        k_vec_std = np.random.uniform(10., 20.)

        coeff = np.random.randn(DIM, n_freq) * np.array(self._cfg.init_cloth_vel_coeff)[:, None] # small z-axis component
        phase = np.random.rand(n_freq) * 2 * np.pi
        k_vec = np.random.randn(n_freq, DIM) * k_vec_std
        phi = phase[None, None, :] + np.sum(k_vec[None, None, :, :] * xyz[:, None, None, :], axis=3)
        vel = np.sum(coeff[None, :, :] * np.cos(phi), axis=2)

        # normalize vel
        vel_scale = np.random.uniform(*self._cfg.init_cloth_vel_range)
        vel /= max(self.eps, np.max(np.abs(vel)))
        vel *= vel_scale

        logger.info(f"init_cloth_vel: k_vec_std={k_vec_std}, vel_scale={vel_scale}")
        return vel

    def random_init_cloth(self, overwrite_render: Optional[bool]=None, callback: Optional[Callable[[], None]]=None):
        cfg = self._cfg
        STEP = cfg.init_cloth_step
        picker = self._picker_init_cloth

        # set init xyz
        xyz = self._tm_mesh_sim_rest.vertices.copy()
        for i in range(cfg.init_cloth_rot_resample_max_try):
            euler_p = np.random.binomial(1, cfg.init_cloth_rot_y_pi_prob) * np.pi # random flip
            euler_y = np.random.uniform(*cfg.init_cloth_rot_z_range) # random rotation
            logger.info(f"euler rpy:(0, {euler_p}, {euler_y})")
            xyz_new = xyz @ tra.euler_matrix(0., euler_p, euler_y)[:3, :3].T
            xyz_new = xyz_new - get_xyz_center(xyz_new) + np.array([0., 0., self._default_height])
            ymin, ymax = cfg.init_cloth_rot_resample_when_y_out_of_range
            if np.max(xyz_new[:, 1]) < max(ymin, ymax) and np.min(xyz_new[:, 1]) > min(ymin, ymax):
                logger.info(f"valid sample, break")
                break
        xyz = xyz_new
        self._set_cloth_xyz(xyz)

        for i in range(STEP):
            self.step(overwrite_render=overwrite_render, callback=callback)
        
        # generate random velocity
        vel = self._random_init_cloth_generate_random_velocities(xyz)
        self._set_cloth_vel(vel)

        for i in range(STEP):
            self.step(overwrite_render=overwrite_render, callback=callback)
        
        # random pick and place positions
        if cfg.init_cloth_move_random_pick_place: 
            xyz = self._get_cloth_xyz()
            x, y, z = xyz[np.random.choice(xyz.shape[0])]
            angle = np.random.uniform(0., 2. * np.pi)
            r = np.random.uniform(*cfg.init_cloth_move_r_range)
            h = np.random.uniform(*cfg.init_cloth_move_h_range)
            h0 = self._cfg.init_cloth_picker_z
            p0 = np.array([x, y, h0])
            p1 = np.array([x, y, h0 + h])
            p2 = np.array([x + r * np.cos(angle), y + r * np.sin(angle), h0 + h])
            p3 = np.array([x + r * np.cos(angle), y + r * np.sin(angle), h0])

            # directly set picker action and grasp some vertices
            picker.set_action(self.PICKER_OPEN)
            picker.step(tra.translation_matrix(np.array([x, y, z])))
            picker.set_action(self.PICKER_CLOSE)

            class Callback:
                def __init__(self, total_cnt: int, picker: Picker, xyz_s: np.ndarray, xyz_e: np.ndarray):
                    self.cnt = 0
                    self.total_cnt = total_cnt
                    self.picker = picker
                    self.xyz_s = xyz_s
                    self.xyz_e = xyz_e
                
                def __call__(self):
                    self.cnt += 1
                    assert self.cnt <= self.total_cnt
                    self.picker.step(tra.translation_matrix(self.xyz_s + (self.xyz_e - self.xyz_s) * self.cnt / self.total_cnt))
                    if callable(callback):
                        callback()

            # move picker
            move_picker_callback = Callback(STEP * cfg.n_substep, picker, p0, p1)
            for i in range(STEP):
                self.step(overwrite_render=overwrite_render, callback=move_picker_callback)
            
            move_picker_callback = Callback(STEP * cfg.n_substep, picker, p1, p2)
            for i in range(STEP):
                self.step(overwrite_render=overwrite_render, callback=move_picker_callback)
            
            move_picker_callback = Callback(STEP * cfg.n_substep, picker, p2, p3)
            for i in range(STEP):
                self.step(overwrite_render=overwrite_render, callback=move_picker_callback)
            
            picker.set_action(self.PICKER_OPEN)
            for i in range(STEP):
                self.step(overwrite_render=overwrite_render, callback=callback)
        
        # move cloth to center
        xyz = self._get_cloth_xyz()
        xyz[:, :2] -= get_xyz_center(xyz)[:2]
        xyz[:, :2] += np.random.uniform(-cfg.init_cloth_xy_random, +cfg.init_cloth_xy_random, 2)
        xyz[:, :2] += np.array(cfg.init_cloth_xy_offset)
        self._set_cloth_xyz(xyz)
    
    def perfect_init_cloth(self, rot_z_deg: float, flip_y: bool, overwrite_render: Optional[bool]=None, callback: Optional[Callable[[], None]]=None):
        cfg = self._cfg
        STEP = cfg.init_cloth_step

        # set init xyz
        xyz = self._tm_mesh_sim_rest.vertices.copy()
        xyz = xyz @ tra.euler_matrix(0., int(bool(flip_y)) * np.pi, np.deg2rad(float(rot_z_deg)))[:3, :3].T
        xyz = xyz - get_xyz_center(xyz) + np.array([*cfg.init_cloth_xy_offset, self._default_height])
        self._set_cloth_xyz(xyz)

        for i in range(STEP):
            self.step(overwrite_render=overwrite_render, callback=callback)
    
    def post_fold(self, overwrite_render: Optional[bool]=None, callback: Optional[Callable[[], None]]=None):
        for i in range(self._cfg.post_fold_step):
            self.step(overwrite_render=overwrite_render, callback=callback)
    
    def load_gt_mesh(self, cloth_fold_gt_dir: str):
        # load ground truth
        tm_mesh_gts: list[tuple[str, trimesh.Trimesh]] = []
        for dirpath, dirnames, filenames in os.walk(cloth_fold_gt_dir):
            for filename in filenames:
                if filename.endswith(".obj"):
                    path = os.path.join(dirpath, filename)
                    tm_mesh_gts.append((path, trimesh.load_mesh(path, process=False)))
        self._tm_mesh_gts = tm_mesh_gts

    def compute_error(self):
        def shape_match(xyz1: np.ndarray, xyz2: np.ndarray, sample_num: int = 1000):
            assert xyz1.shape == xyz2.shape and xyz1.shape[1] == 3 and xyz2.shape[1] == 3, f"{xyz1.shape}, {xyz2.shape}"

            angle = np.arange(sample_num) * np.pi * 2 / sample_num
            rotation = np.array([
                [np.cos(angle), -np.sin(angle), np.zeros(sample_num)],
                [np.sin(angle), np.cos(angle), np.zeros(sample_num)],
                [np.zeros(sample_num), np.zeros(sample_num), np.ones(sample_num)],
            ])
            rotation = rotation.transpose(2, 0, 1) # [N, 3, 3]
            xy2_rot = np.einsum('ilk,jl->ijk', rotation, xyz2) # [N, P, 3]
            translation = np.mean(xy2_rot - xyz1, axis=1) # [N, 3]
            sqrerr = (xy2_rot - xyz1 - translation.reshape(sample_num, 1, 3)) ** 2 # [N, P, 3]
            sqrerr = np.mean(sqrerr.reshape(sample_num, -1), axis=1) # [N]
            best_sample = np.argmin(sqrerr)
            
            rot: np.ndarray = rotation[best_sample]
            trans: np.ndarray = np.einsum('ij,j->i', rot, translation[best_sample])
            err: float = sqrerr[best_sample]
            return rot, trans, err

        curr_mesh = self._get_curr_render_mesh()
        min_err = float('inf')
        assert hasattr(self, "_tm_mesh_gts"), "load_gt_mesh() should be called before compute_error()"
        for path, mesh in self._tm_mesh_gts:
            rot, trans, err = shape_match(curr_mesh.vertices, mesh.vertices)
            logger.info(f"{path}: err={err:.4f}")

            min_err = min(min_err, err)
        
        return min_err

    def is_grasp_fail(self):
        """determine if the current grasp is already failed, only if current action is close and previous action is open"""
        return {
            k: (
                self._picker_dict[k].current_grasp_nothing() and 
                Picker.is_open_action(self._cache["before_step"][k]) and 
                Picker.is_close_action(self._cache["after_step"][k])
            ) for k in ["left", "right"]
        }
    
    def will_grasp_fail(self, xyz_l: np.ndarray, xyz_r: np.ndarray, picker_l: float, picker_r: float):
        """determin if the next step will fail (the action has not been executed yet) only if current is open"""
        will_grasp_fail = {}
        for k, xyz, p in zip(["left", "right"], [xyz_l, xyz_r], [picker_l, picker_r]):
            if Picker.is_close_action(p) and Picker.is_open_action(self._cache["after_step"][k]):
                tf = self._picker_dict[k].tf.copy()
                tf[:3, 3] = xyz # this is an approximation, the rotation may change
                will_grasp_fail[k] = self._picker_dict[k].compute_grasp_vertices(tf)[0].shape == (0, )
            else:
                will_grasp_fail[k] = False
        return will_grasp_fail
    
    # misc
    @property
    def current_step_idx(self):
        return self._state.step_idx
    
    @property
    def last_render_frame_idx(self):
        return self._state.render_frame_idx - 1

    @property
    def cloth_scale(self):
        return self._cfg.cloth_scale
    
    @property
    def ik_fail_count(self):
        return self._robot.ik_fail_count
    
    def reset(self):
        """set cloth to reset shape"""
        self._robot.reset()
        self._state.reset()
        self._set_cloth_xyz(self._cloth_xyz_init)
        self._init_cache()
    
    def join(self):
        self._render_process.join()

    def close(self):
        self._render_process.join()
        pyflex.clean()