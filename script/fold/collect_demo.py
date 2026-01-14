import subprocess
import os
import multiprocessing
from dataclasses import dataclass, field
import copy
from typing import Optional, Callable
import random
import datetime
import shutil
import json


@dataclass
class Cfg:
    difficulty: str = "easy"
    category: str = "tshirt"
    style: int = 0

    not_correct_action_prob: float = 0.2

    num_running: int = 1000
    cudas: list[int] = field(default_factory=lambda: list(range(0, 8)))
    traj_num_per_running: int = 5
    render_process_num: int = 1

    cloth_obj_path_fn: Callable[[int], str] = field(
        init=False,
        default=lambda self, i: f"data/fold/mesh/gen_proc/tshirt_sp_20250314/{i}/mesh.obj",
    )
    cloth_scale_fn: dict[str, Callable[[], float]] = field(
        init=False,
        default_factory=lambda: {
            ("tshirt", 0): (lambda: random.uniform(0.45, 0.65)),
            ("tshirt", 1): (lambda: random.uniform(0.45, 0.60)),
            ("tshirt", 2): (lambda: random.uniform(0.45, 0.60)),
            ("trousers", 0): (lambda: random.uniform(0.40, 0.50)),
        }, 
    )
    cloth_xy_offset: dict[str, str] = field(
        init=False,
        default_factory=lambda: dict(tshirt='[0.0, 0.0]', trousers='[0.0, -0.05]'), 
    )
    cloth_rot_y_range: dict[str, str] = field(
        init=False,
        default_factory=lambda: dict(tshirt='[-1., +1.]', trousers='[-0.3, +0.3]'), 
    )


@dataclass
class Job:
    job_id: int
    output_dir: str
    tmp_dir: str
    cloth_obj_path: str
    cloth_scale: float


class Main:
    def __init__(self, cfg: Cfg):
        self.cfg = copy.deepcopy(cfg)
        self.job_queue: multiprocessing.Queue[Optional[Job]] = multiprocessing.Queue()
        self.init_vel_range = dict(easy='[1.0, 2.0]', hard='[5.0, 10.0]')[self.cfg.difficulty]
    
    def worker(self, worker_id: int):
        cuda = self.cfg.cudas[worker_id]
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda}"

        while True:
            job = self.job_queue.get()
            if job is None:
                break
            self.run_job(job, worker_id=worker_id, cuda=cuda)

    def run_job(self, job: Job, **kwargs):
        cmd = (
            f"python run/fold_multi_cat.py "
            + f"run.category={self.cfg.category} "
            + f"run.style={self.cfg.style} "
            + f"env.cloth_obj_path={job.cloth_obj_path} "
            + f"env.cloth_scale={job.cloth_scale} "
            + f"hydra.run.dir={job.output_dir} "
            + f"run.traj_num={self.cfg.traj_num_per_running} "
            + f"env.render_process_num={self.cfg.render_process_num} "
            + f"'+env.init_cloth_vel_range={self.init_vel_range}' "
            + f"'+env.init_cloth_xy_offset={self.cfg.cloth_xy_offset[self.cfg.category]}' "
            + f"'+env.init_cloth_rot_resample_when_y_out_of_range={self.cfg.cloth_rot_y_range[self.cfg.category]}' "
            + f"policy.run.state.not_correct_action_prob={self.cfg.not_correct_action_prob} "
        )
        print(f"run command:\n{cmd}")
        fn = os.path.join(job.tmp_dir, f"{job.job_id}_{kwargs['worker_id']}_{kwargs['cuda']}.log")
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        with open(fn, "w") as f:
            subprocess.run(cmd, shell=True, stdout=f, stderr=f)

    def run(self):
        process_list: list[multiprocessing.Process] = []
        now_str = datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        dir_prefix = f"data/fold/demo_{self.cfg.difficulty}/{self.cfg.category}{self.cfg.style}/{now_str}"
        os.makedirs(os.path.join(dir_prefix, ".tmp"), exist_ok=False)
        shutil.copy(__file__, os.path.join(dir_prefix, ".tmp", "script.py"))
        with open(os.path.join(dir_prefix, ".tmp", "info.json"), "w") as f:
            json.dump(dict(pid=os.getpid()), f, indent=4)

        for i in range(len(self.cfg.cudas)):
            p = multiprocessing.Process(target=self.worker, args=(i,), daemon=True)
            process_list.append(p)
            p.start()

        # append jobs
        for i in range(self.cfg.num_running):
            self.job_queue.put(Job(
                i, 
                output_dir=os.path.join(dir_prefix, f"{i}"),
                tmp_dir=os.path.join(dir_prefix, ".tmp"),
                cloth_obj_path=self.cfg.cloth_obj_path_fn(i),
                cloth_scale=self.cfg.cloth_scale_fn[(self.cfg.category, self.cfg.style)](),
            ))
        for i in range(len(self.cfg.cudas)):
            self.job_queue.put(None)

        for p in process_list:
            p.join()


def main():
    Main(Cfg()).run()


if __name__ == '__main__':
    main()