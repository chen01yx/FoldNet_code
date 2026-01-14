from typing import Optional
import os

import open3d as o3d
import numpy as np
import torch
import trimesh.transformations as tra

import garmentds.common.utils as utils
import batch_urdf

class O3dVisualizer:
    def apply_world_tf(self, xyz: np.ndarray):
        return xyz.copy()
        camera_tf = tra.translation_matrix([0.0, 0.3, 0.3]) @ tra.euler_matrix(1.0, 0.0, np.pi)
        camera_tf_inv = np.linalg.inv(camera_tf)
        return xyz @ camera_tf_inv[:3, :3].T + camera_tf_inv[:3, 3]
    
    def __init__(self, output_dir="vis_output", vis_robot=False):
        self._output_dir = os.path.abspath(output_dir)
        os.makedirs(self._output_dir)
        self._vis_robot = bool(vis_robot)
        
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(width=1280, height=960)
        
        if not self._vis_robot:
            # dummy box, open3d bug
            box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
            box.translate(np.array([0., 0., 10.]))
            self._vis.add_geometry(box)
        else:
            device = "cpu"
            self._urdf = batch_urdf.URDF(
                1, utils.get_path_handler()("asset/galbot_one_charlie/urdf_nomtl.urdf"), device=device, 
            )
            self._all_o3d_mesh = []
            for name, tf in self._urdf.link_transform_map.items():
                mesh = self._urdf.get_link_scene(name).to_geometry()
                if mesh.vertices.shape[0] > 0:
                    vertices = o3d.utility.Vector3dVector(mesh.vertices)
                    faces = o3d.utility.Vector3iVector(mesh.faces)
                    o3d_mesh = o3d.geometry.TriangleMesh(vertices, faces)
                    self._vis.add_geometry(o3d_mesh)
                    self._all_o3d_mesh.append((o3d_mesh, name, mesh.vertices.copy()))
            self._urdf.update_base_link_transformation("base_link", torch.tensor([[0.0, -0.75, -0.455, 0.70710678, 0., 0., 0.70710678]], device=device))
                    
        self._scene_pc = o3d.geometry.PointCloud()
        self._scene_pc.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        self._vis.add_geometry(self._scene_pc)
        self._action_pc = o3d.geometry.PointCloud()
        self._action_pc.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        self._vis.add_geometry(self._action_pc)
        
        self._current_step = 0
        self.render(qpos={k: 0. for k in self._urdf.cfg.keys()} if self._vis_robot else None)
    
    def close(self):
        self._vis.destroy_window()
    
    def render(
        self, 
        qpos: Optional[dict[str, float]]=None, 
        scene_pc: Optional[np.ndarray]=None, 
        action_pc: Optional[np.ndarray]=None,
        cam_ext: Optional[np.ndarray]=None,
        cam_int: Optional[np.ndarray]=None,
    ):
        if scene_pc is not None:
            idx, = np.where(scene_pc[:, 2] > -0.1)
            self._scene_pc.points = o3d.utility.Vector3dVector(self.apply_world_tf(scene_pc[idx, 0:3]))
            self._scene_pc.colors = o3d.utility.Vector3dVector(scene_pc[idx, 3:6])
            self._vis.update_geometry(self._scene_pc)
        if action_pc is not None:
            self._action_pc.points = o3d.utility.Vector3dVector(self.apply_world_tf(action_pc[:, 0:3]))
            self._action_pc.colors = o3d.utility.Vector3dVector(action_pc[:, 3:6])
            self._vis.update_geometry(self._action_pc)
        if qpos is not None and self._vis_robot:
            self._urdf.update_cfg(self._urdf.cfg_f2t(qpos))
            for o3d_mesh, name, vert_old in self._all_o3d_mesh:
                tf = utils.torch_to_numpy(self._urdf.link_transform_map[name])[0]
                o3d_mesh.vertices = o3d.utility.Vector3dVector(self.apply_world_tf(vert_old @ tf[:3, :3].T + tf[:3, 3]))
                self._vis.update_geometry(o3d_mesh)
        
        # ctr = self._vis.get_view_control()
        # param = ctr.convert_to_pinhole_camera_parameters()
        # ext = param.extrinsic
        # print(ext)
        # print(tra.euler_from_matrix(ext))
        # print(tra.translation_from_matrix(ext))
        
        # param.extrinsic = tra.translation_matrix([0., 0.2, 2.]) @ tra.euler_matrix(-np.pi * 0.75, 0., np.pi)
        # if cam_ext is not None:
        #     ext = cam_ext @ np.diag([1., -1., -1., 1.]) @ tra.translation_matrix([0., 0., 0.1]) @ tra.euler_matrix(np.pi * 1.5, 0., 0.)
        #     param.extrinsic = ext
        
        ctr = self._vis.get_view_control()
        param = o3d.camera.PinholeCameraParameters()
        param.extrinsic = tra.translation_matrix([0., -0.1, 0.8]) @ tra.euler_matrix(np.pi - 0.3, 0., 0.)
        param.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=1280, height=960, fx=600, fy=600, cx=640. - 0.5, cy=480 - 0.5
        )
        ctr.convert_from_pinhole_camera_parameters(param)
        
        self._vis.poll_events()
        self._vis.update_renderer()
        
        self._vis.capture_screen_image(os.path.join(self._output_dir, f"{self._current_step:03d}.png"))
        self._current_step += 1


if __name__ == "__main__":
    vis = O3dVisualizer()
    import tqdm
    for i in tqdm.tqdm(range(1000)):
        qpos = vis._urdf.cfg_t2f(vis._urdf.cfg)
        # for k, v in qpos.items():
        #     qpos[k] += np.random.randn()
        point_n = np.random.randint(1000, 2000)
        scene_pc = np.random.rand(point_n, 6)
        action_pc = np.random.rand(point_n, 6)
        vis.render(qpos, scene_pc, action_pc)