import os
import sys
import shutil
import subprocess

import json
import PIL.Image
import trimesh
import trimesh.transformations as tra
import pyflex

from garmentds.gentexture.template.clothes import *
from garmentds.gentexture.utils.flex_utils import *
from garmentds.gentexture.utils.mesh_utils import detect_obstruction

class SimEnv:
    def __init__(self, gui : bool = False, 
                 dump_visualizations : bool = False,
                 output_dir : str = 'outputs/flex_imgs'):
        pyflex.init(not gui, True, 480, 480, 0)
        
        self.config = get_default_config()
        self.grasp_height = 0.03
        self.particle_radius = 0.00625
        self.default_speed = 5e-3

        self.action_tool = PickerPickPlace(
            num_picker=2,
            particle_radius=self.particle_radius,
            picker_radius=self.particle_radius,
            picker_low=(-5, 0, -5),
            picker_high=(5, 5, 5))
        self.grasp_states = [0, 0] # true if it's gripping something
        self.step_simulation = pyflex.step

        # visualization
        self.dump_visualizations = dump_visualizations
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        if self.dump_visualizations:
            self.action_tool.moved_steps = 0

        self.action_handlers = {
            "simple_drop": self.simple_drop_primitive,
            "pick_and_place": self.pick_and_place_primitive,
            "random_velocity": self.random_velocity_primitive,
            "random_velocity_and_move_sleeve": self.random_velocity_and_move_sleeve_primitive,
        }

        self.cloth_handlers: dict[str, Base] = {
            "tshirt": TShirtSim(),
            "tshirt_sp": TShirtSPSim(),
            "trousers": TrousersSim(),
            "vest": VestCloseSPSim(),
            "hooded": HoodedCloseSim(),
        }

    def set_scene(self,
        category : str,
        cloth_obj_path : str,
        cloth_pos : tuple = (0, 0.1, 0),
        cloth_stiff : tuple = (0.75, .02, .02),
        cloth_mass : float = 100,
        cam_position : tuple = (0, 1, 1),
        cam_angle : tuple = (0, -np.pi/4, 0),
    ):
        flipped, dist_min, mesh_processed, mesh_faces, mesh_stretch_edges, \
            mesh_bend_edges, mesh_shear_edges = load_cloth(cloth_obj_path)

        config = deepcopy(self.config)
        config.update({
            'cloth_pos': cloth_pos,
            'cloth_stiff': cloth_stiff,
            'cloth_mass': cloth_mass,
            'mesh_verts': mesh_processed.vertices.reshape(-1),
            'mesh_stretch_edges': mesh_stretch_edges.reshape(-1),
            'mesh_bend_edges': mesh_bend_edges.reshape(-1),
            'mesh_shear_edges': mesh_shear_edges.reshape(-1),
            'mesh_faces': mesh_faces.reshape(-1),
            'mesh_nocs_verts': np.array([[1.0, 1.0, 1.0]]).repeat(
                mesh_processed.vertices.shape[0], axis=0),        
        })

        config['camera_params']['default_camera']['cam_position'] = cam_position
        config['camera_params']['default_camera']['cam_angle'] = cam_angle

        pyflex.set_scene(0, config['scene_config'])
        pyflex.set_camera_params(config['camera_params'][config['camera_name']])

        pyflex.add_cloth_mesh(
                position=config['cloth_pos'], 
                verts=config['mesh_verts'], 
                faces=config['mesh_faces'], 
                stretch_edges=config['mesh_stretch_edges'], 
                bend_edges=config['mesh_bend_edges'], 
                shear_edges=config['mesh_shear_edges'], 
                stiffness=config['cloth_stiff'], 
                uvs=config['mesh_nocs_verts'],
                mass=config['cloth_mass'])
        
        self.action_tool.reset(np.array([0., 0.5, 0.]))

        # start simulation
        pyflex.step()
        
        self.cloth_config = {
            "flipped" : flipped,
            "category" : category,
            "dist_min" : dist_min,
            "mesh_processed" : mesh_processed,
            "cloth_obj_path" : cloth_obj_path,
        }
        self.cloth_handlers[category].set_flipped(flipped)

    def deform_cloth(self, output_dir : str = ".", **kwargs):
        strategy = kwargs.get("strategy", "pick_and_place")
        if strategy not in self.action_handlers:
            raise Exception(f"Invalid strategy: {strategy}")
        self.action_handlers[strategy](**kwargs[strategy])

        # save mesh
        os.makedirs(output_dir, exist_ok=True)

        cloth_obj_path = self.cloth_config["cloth_obj_path"]
        mesh_raw = trimesh.load_mesh(cloth_obj_path, process=False)
        mesh_processed = self.cloth_config["mesh_processed"]
        verts_reposed = pyflex.get_positions().reshape((-1, 4))[:, :3]
        mesh_raw.vertices = verts_reposed[self.cloth_config["dist_min"], :]
        mesh_processed.vertices = verts_reposed[:mesh_processed.vertices.shape[0], :]
        mesh_raw.vertex_normals = mesh_processed.vertex_normals[self.cloth_config["dist_min"], :]
        mesh_raw.export(os.path.join(output_dir, "mesh_deformed.obj"))
        
        cloth_obj_dir = os.path.split(cloth_obj_path)[0]
        if os.path.exists(os.path.join(cloth_obj_dir, "material_0.png")):
            shutil.copy(os.path.join(cloth_obj_dir, "material_0.png"), 
                        os.path.join(output_dir, "material_0.png"))
        elif os.path.exists(os.path.join(cloth_obj_dir, "material_0.json")):
            shutil.copy(os.path.join(cloth_obj_dir, "material_0.json"), 
                        os.path.join(output_dir, "material_0.json"))
        else:
            os.remove(os.path.join(output_dir, "material_0.png"))
            shutil.copy(os.path.join(cloth_obj_dir, "material.mtl"),
                        os.path.join(output_dir, "material.mtl"))
            for txt in ["texture_kd.png", "texture_ks.png", "texture_n.png"]:
                shutil.copy(os.path.join(cloth_obj_dir, txt),
                            os.path.join(output_dir, txt))

        # reset
        self.post_action(output_dir)
        self.reset()

    def simple_drop_primitive(self, drop_height : float = 0.1):
        num_verts = self.cloth_config["mesh_processed"].vertices.shape[0]
        xyzm = pyflex.get_positions().reshape((-1, 4))        
        mean_y = np.mean(xyzm[:num_verts, 1])
        xyzm[:num_verts, 1] += (drop_height-mean_y)
        pyflex.set_positions(xyzm)

        _,_ = wait_until_stable(
            output_dir=self.output_dir, max_steps=600,
            dump_visualizations=self.dump_visualizations, 
            visualize_start_step=self.action_tool.moved_steps,)

    def pick_and_place_primitive(
        self, 
        p1:np.ndarray = np.array([0.0, 0.0, 0.0]), 
        p2:np.ndarray = np.array([0.0, 0.0, 0.0]), 
        coverage_ratio : float = 1.0,
        lift_height : float = 0.2,
        random_pick_place : bool = False,
    ):
        if not random_pick_place:
            pick_pos, place_pos = p1, p2
        else:
            mesh_verts = self.cloth_config["mesh_processed"].vertices
            pick_pos = mesh_verts[np.random.randint(mesh_verts.shape[0])]

            direction = np.random.uniform(0, np.pi*2)
            #distance = np.random.uniform(0.1, 0.3)
            distance = 0.2
            place_pos = pick_pos + np.array([np.cos(direction)*distance, 0, np.sin(direction)*distance])
            coverage_ratio = np.random.uniform(0.8, 1.0)
            print("Random pick-place: ", pick_pos, place_pos, coverage_ratio)

        pick_pos[1] = self.grasp_height
        place_pos[1] = self.grasp_height

        prepick_pos = pick_pos.copy()
        prepick_pos[1] = lift_height
        preplace_pos = place_pos.copy()
        preplace_pos[1] = lift_height

        # execute action
        self.set_grasp_states(False)
        self.movep([prepick_pos, [-0.2, 0.3, -0.2]], speed=5e-3)
        self.movep([pick_pos, [-0.2, 0.3, -0.2]], speed=5e-3)

        _, step = wait_until_stable(
                    dump_visualizations=self.dump_visualizations, 
                    visualize_start_step=self.action_tool.moved_steps, 
                    output_dir=self.output_dir, max_steps=100)
        self.action_tool.moved_steps = step

        self.set_grasp_states(True)
        self.movep([prepick_pos, [-0.2, 0.3, -0.2]], speed=5e-3)        

        # get the true position reached by the end effector
        preplace_pos, _ = self.movep([preplace_pos, [-0.2, 0.3, -0.2]], 
                                     speed=5e-3, coverage_ratio=coverage_ratio)

        place_pos = preplace_pos.copy()
        place_pos[1] = self.grasp_height
        self.movep([place_pos, [-0.2, 0.3, -0.2]], speed=5e-3)

        self.set_grasp_states(False)
        self.movep([preplace_pos, [-0.2, 0.3, -0.2]], speed=5e-3)

        _, step = wait_until_stable(
                    dump_visualizations=self.dump_visualizations, 
                    visualize_start_step=self.action_tool.moved_steps, 
                    output_dir=self.output_dir, max_steps=600)
        self.action_tool.moved_steps = step

    def random_velocity_primitive(
        self, n_freq = 100,
        init_cloth_vel_coeff:tuple = (1.,1.,.5),
        init_cloth_vel_range:tuple = (3.0, 8.0),
        gather_vel_scale:tuple = 5,
    ):
        # wait for drop
        _,_ = wait_until_stable(
            output_dir=self.output_dir, max_steps=200,
            dump_visualizations=self.dump_visualizations, 
            visualize_start_step=self.action_tool.moved_steps,)

        # give random velocities
        num_verts = self.cloth_config["mesh_processed"].vertices.shape[0]
        xyz = pyflex.get_positions().reshape((-1, 4))[:num_verts, :3]
        vel = self._random_init_cloth_generate_random_velocities(
            xyz, n_freq, init_cloth_vel_coeff, init_cloth_vel_range)

        ## add velocity toward center of mass
        center_of_mass = np.mean(xyz, axis=0)
        dist = center_of_mass - xyz
        vel += dist * gather_vel_scale

        pyflex.set_velocities(vel.reshape(-1,3))

        # wait for stable
        _,_ = wait_until_stable(
            output_dir=self.output_dir, max_steps=600,
            dump_visualizations=self.dump_visualizations, 
            visualize_start_step=self.action_tool.moved_steps,)

    def _random_init_cloth_generate_random_velocities(
        self, xyz: np.ndarray, n_freq: int, 
        init_cloth_vel_coeff: tuple, init_cloth_vel_range: tuple,
    ):
        xyz = (xyz - xyz.min(axis=0)) / (xyz.max(axis=0) - xyz.min(axis=0))

        DIM = 3
        EPS = 1e-6
        k_vec_std = np.random.uniform(10., 20.)

        coeff = np.random.randn(DIM, n_freq) * np.array(init_cloth_vel_coeff)[:, None] # small z-axis component
        phase = np.random.rand(n_freq) * 2 * np.pi
        k_vec = np.random.randn(n_freq, DIM) * k_vec_std
        phi = phase[None, None, :] + np.sum(k_vec[None, None, :, :] * xyz[:, None, None, :], axis=3)
        vel = np.sum(coeff[None, :, :] * np.cos(phi), axis=2)

        # normalize vel
        vel_scale = np.random.uniform(*init_cloth_vel_range)
        vel /= max(EPS, np.max(np.abs(vel)))
        vel *= vel_scale

        return vel

    def random_velocity_and_move_sleeve_primitive(
        self, n_freq = 100,
        init_cloth_vel_coeff:tuple = (1.,1.,.5),
        init_cloth_vel_range:tuple = (5.0, 10.0),
        lift_height : float = 0.2,
    ):
        """
        This primitive use random velocity frist and then move the sleeve.
        It is only appliable to t-shirt category.
        """

        # get mesh information
        dist_min = self.cloth_config["dist_min"]
        mesh_verts = self.cloth_config["mesh_processed"].vertices
        num_verts = mesh_verts.shape[0]

        mesh_info_path = os.path.join(
            os.path.split(self.cloth_config["cloth_obj_path"])[0], "mesh_info.json")
        with open(mesh_info_path, "r") as f:
            mesh_info = json.load(f)
        keypoint_idxs = mesh_info["triangulation"]["keypoint_idx"]

        r_sleeve_top_idx = dist_min[keypoint_idxs["r_sleeve_top"][0]]
        r_sleeve_btm_idx = dist_min[keypoint_idxs["r_sleeve_bottom"][0]]
        l_sleeve_top_idx = dist_min[keypoint_idxs["l_sleeve_top"][0]]
        l_sleeve_btm_idx = dist_min[keypoint_idxs["l_sleeve_bottom"][0]]

        r_corner_idx = dist_min[keypoint_idxs["r_corner"][0]]
        l_corner_idx = dist_min[keypoint_idxs["l_corner"][0]]

        # random velocity
        xyz = pyflex.get_positions().reshape((-1, 4))[:num_verts, :3]
        vel = self._random_init_cloth_generate_random_velocities(
            xyz, n_freq, init_cloth_vel_coeff, init_cloth_vel_range)
        pyflex.set_velocities(vel.reshape(-1,3))

        _, step = wait_until_stable(
            output_dir=self.output_dir, max_steps=600,
            dump_visualizations=self.dump_visualizations, 
            visualize_start_step=self.action_tool.moved_steps,)
        self.action_tool.moved_steps = step

        cloth_radius = 0.5*(np.linalg.norm(mesh_verts[r_sleeve_top_idx])+
                            np.linalg.norm(mesh_verts[l_sleeve_top_idx]))

        # move sleeve
        ## move right sleeve
        move_r_sleeve = np.random.rand()
        distance = np.random.uniform(0.4*cloth_radius, 0.6*cloth_radius)
        xyz = pyflex.get_positions().reshape((-1, 4))[:num_verts, :3]
        if move_r_sleeve < 0.2:
            pick_pose = xyz[r_sleeve_top_idx]
            direction = xyz[r_corner_idx] - pick_pose
            direction[1] = 0.0
            direction /= np.linalg.norm(direction)
            place_pose = pick_pose + direction * distance
        elif move_r_sleeve < 0.4:
            pick_pose = xyz[r_sleeve_top_idx]
            direction = xyz[l_corner_idx] - pick_pose
            direction[1] = 0.0
            direction /= np.linalg.norm(direction)
            place_pose = pick_pose + direction * distance
        elif move_r_sleeve < 0.6:
            pick_pose = xyz[r_sleeve_btm_idx]
            direction = xyz[r_corner_idx] - pick_pose
            direction[1] = 0.0
            direction /= np.linalg.norm(direction)
            place_pose = pick_pose + direction * distance
        elif move_r_sleeve < 0.8:
            pick_pose = xyz[r_sleeve_btm_idx]
            direction = xyz[l_corner_idx] - pick_pose
            direction[1] = 0.0
            direction /= np.linalg.norm(direction)
            place_pose = pick_pose + direction * distance

        if move_r_sleeve < 0.8:
            self.pick_and_place_primitive(pick_pose, place_pose, coverage_ratio=0.0, lift_height=lift_height)

        ## move left sleeve
        move_l_sleeve = np.random.rand()
        distance = np.random.uniform(0.7*cloth_radius, 1.1*cloth_radius)
        xyz = pyflex.get_positions().reshape((-1, 4))[:num_verts, :3]
        if move_l_sleeve < 0.2:
            pick_pose = xyz[l_sleeve_top_idx]
            direction = xyz[l_corner_idx] - pick_pose
            direction[1] = 0.0
            direction /= np.linalg.norm(direction)
            place_pose = pick_pose + direction * distance
        elif move_l_sleeve < 0.4:
            pick_pose = xyz[l_sleeve_top_idx]
            direction = xyz[r_corner_idx] - pick_pose
            direction[1] = 0.0
            direction /= np.linalg.norm(direction)
            place_pose = pick_pose + direction * distance
        elif move_l_sleeve < 0.6:
            pick_pose = xyz[l_sleeve_btm_idx]
            direction = xyz[l_corner_idx] - pick_pose
            direction[1] = 0.0
            direction /= np.linalg.norm(direction)
            place_pose = pick_pose + direction * distance
        elif move_l_sleeve < 0.8:
            pick_pose = xyz[l_sleeve_btm_idx]
            direction = xyz[r_corner_idx] - pick_pose
            direction[1] = 0.0
            direction /= np.linalg.norm(direction)
            place_pose = pick_pose + direction * distance

        if move_l_sleeve < 0.8:
            self.pick_and_place_primitive(pick_pose, place_pose, coverage_ratio=0.0, lift_height=lift_height)

    def movep(
        self, pos, speed=None, limit=1000, min_steps=None, 
        coverage_ratio = None, eps=1e-4
    ):
        if speed is None:
            if self.dump_visualizations:
                speed = self.default_speed
            else:
                speed = 0.1
        target_pos = np.array(pos)
        target_coverage = None
        if coverage_ratio is not None:
            target_coverage = get_current_covered_area() * coverage_ratio
        for step in range(limit):
            curr_pos = self.action_tool._get_pos()[0]
            deltas = [(targ - curr)
                      for targ, curr in zip(target_pos, curr_pos)]
            dists = [np.linalg.norm(delta) for delta in deltas]
            if all([dist < eps for dist in dists]) and\
                    (min_steps is None or step > min_steps):
                return [pos for pos in curr_pos]
            action = []
            for targ, curr, delta, dist, gs in zip(
                    target_pos, curr_pos, deltas, dists, self.grasp_states):
                if dist < speed:
                    action.extend([*targ, float(gs)])
                else:
                    delta = delta/dist
                    action.extend([*(curr+delta*speed), float(gs)])
            action = np.array(action)
            _, coverage_reached = self.action_tool.step(
                action, target_coverage, step_sim_fn=self.step_simulation,
                dump_visualizations=self.dump_visualizations, output_dir=self.output_dir)
            
            curr_pos = self.action_tool._get_pos()[0]
            if coverage_ratio is not None and coverage_reached:
                return [pos for pos in curr_pos]

        curr_pos = self.action_tool._get_pos()[0]
        return [pos for pos in curr_pos]     

    def set_grasp_states(self, grasp_states):
        if type(grasp_states) == bool:
            self.grasp_states = [grasp_states] * len(self.grasp_states)
        elif len(grasp_states) == len(self.grasp_states):
            self.grasp_states = grasp_states
        else:
            raise Exception()

    def reset_end_effectors(self):
        pyflex.clear_shapes()

    def reset(self):
        self.reset_end_effectors()
        pyflex.set_scene(0, self.config['scene_config'])

    def post_action(self, output_dir : str = "."):
        """
            process mesh_info, filter out obstructed keypoints 
        """
        mesh_info_path = os.path.join(
            os.path.split(self.cloth_config["cloth_obj_path"])[0], "mesh_info.json")
        shutil.copy(mesh_info_path, os.path.join(output_dir, "mesh_info.json"))

        if os.path.exists(mesh_info_path):
            with open(mesh_info_path, "r") as f:
                mesh_info = json.load(f)
            keypoint_idxs = mesh_info["triangulation"]["keypoint_idx"]

            idxs = []
            for key in self.cloth_handlers[self.cloth_config["category"]].generate_keys():
                idxs.append(keypoint_idxs[key][0])

            mesh_deformed_path = os.path.join(output_dir, "mesh_deformed.obj")
            mesh = trimesh.load_mesh(mesh_deformed_path, process=False)
            keypoints = mesh.vertices[idxs]
            mask = detect_obstruction(mesh_deformed_path, idxs).reshape(-1, 1)
            keypoints_unobstructed = np.hstack((keypoints, mask), dtype=np.float32)
            np.save(os.path.join(output_dir, "keypoints_3D.npy"), keypoints_unobstructed)

        else:
            raise Exception("mesh_info.json not found")
    
    def render(self, blender_script: str, output_dir: str, 
               need_mask: bool, need_keypoints_2D: bool,
               cloth_use_polyhaven_textures: bool = False):
        """
            render 'mesh_deformed.obj' and generate 'keypoints_2D'. 
            'mesh_deformed.obj' and 'keypoints_3D.npy' are supposed 
            to be found under 'output_dir'.
        """
        command_line = ['blender', '-noaudio', '--background', '--python', blender_script, 
                         '--', f'--base_dir={output_dir}', f"--cloth_rotation_euler=[{np.pi/2},0.0,0.0]"]

        if need_mask:
            command_line.append("--need_mask")
        if need_keypoints_2D:
            command_line.append("--need_keypoints_2D")
        if cloth_use_polyhaven_textures:
            command_line.append("--cloth_use_polyhaven_textures")

        subprocess.call(command_line)


if __name__ == '__main__':
    """
        Use this script to test the simulation environment.
    """

    import argparse

    parser = argparse.ArgumentParser(description='Test the simulation environment.')
    parser.add_argument('--category', type=str, default='hooded', help='cloth category')
    parser.add_argument('--strategy', type=str, default='random_velocity', help='deformation strategy')
    parser.add_argument('--cloth_obj_path', type=str, help='path to cloth obj file')
    parser.add_argument('--output_dir', type=str, help='output directory')

    args = parser.parse_args()
    sim = SimEnv(gui=False)

    sim.set_scene(args.category, args.cloth_obj_path)
    kwargs = {
        "strategy": args.strategy,
        "pick_and_place": {
            "p1": [0.0, 0.0, 0.0],
            "p2": [0.0, 0.0, 0.0],
            "coverage_ratio": 1.0,
            "random_pick_place": True},
        "random_velocity": {
            "n_freq": 100,
            "init_cloth_vel_coeff": [1.0, 1.0, 0.5],
            "init_cloth_vel_range": [3.0, 8.0],
            "gather_vel_scale": 5.0},
    }
    sim.deform_cloth(args.output_dir, **kwargs)
    sim.reset()