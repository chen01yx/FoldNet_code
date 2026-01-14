from dataclasses import dataclass, field
import math

import tqdm
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import KDTree
import trimesh

import taichi as ti
from garmentds.common.taichi import Triplet, GLOBAL_CREATER


@dataclass
class HoodOptimizerCfg:
    final_z: float = -0.04
    y_offset: float = 0.00
    z_count: int = 200
    rc: float = 1e0
    kc: float = 1e1
    cutoff_r: float = 0.14
    damping: float = 1e0
    xyzc: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 2.0]))


@ti.data_oriented
class HoodOptimizer:
    def __init__(
        self, 
        hood_vert_type_np: np.ndarray, 
        hood_vert_xyz_np: np.ndarray, 
        hood_edge_np: np.ndarray, 
        hood_face_np: np.ndarray,
    ) -> None:
        self.hood_vert_type_np = hood_vert_type_np.copy()
        self.hood_vert_xyz_np = hood_vert_xyz_np.copy()
        self.hood_edge_np = hood_edge_np.copy()

        self.nv = nv = hood_vert_type_np.shape[0]
        self.ne = ne = hood_edge_np.shape[0]

        self.pos = GLOBAL_CREATER.VectorField(3, dtype=float, shape=(nv, ))
        self.pos.from_numpy(hood_vert_xyz_np.astype(np.float32))

        self.pos0 = GLOBAL_CREATER.VectorField(3, dtype=float, shape=(nv, ))
        self.pos0.from_numpy(hood_vert_xyz_np.astype(np.float32))

        self.vert_type = GLOBAL_CREATER.ScalarField(dtype=int, shape=(nv, ))
        self.vert_type.from_numpy(hood_vert_type_np.astype(np.int32))

        self.edge = GLOBAL_CREATER.VectorField(2, dtype=int, shape=(ne, ))
        self.edge.from_numpy(hood_edge_np.astype(np.int32))

        self.dE = GLOBAL_CREATER.ScalarField(dtype=float, shape=(nv * 3, ))
        self.ddE = GLOBAL_CREATER.StructField(Triplet, shape=(nv * 9 + ne * 36))

        self.curr_E = float("inf")
        self.face = hood_face_np.copy()
    
    @ti.kernel
    def _compute_energy_kernel(
        self, 
        rc: float, kc: float, cutoff_r: float, damping: float, 
        xyz0: ti.types.vector(n=3, dtype=float), 
        xyzc: ti.types.vector(n=3, dtype=float),
        # theta_phi_value_std: ti.types.ndarray(dtype=ti.math.vec4),
    ) -> float:
        E = 0.
        EPS = 1e-7
        xc, yc, zc = xyzc
        x0, y0, z0 = xyz0
        cst = rc * ti.math.length(xyzc) / self.nv

        for i in self.dE:
            self.dE[i] = 0

        for vid in range(self.nv):
            # compute coefficient
            # For simplicity, we ignore all gradients of coeff. This is an approximation. 
            coeff = cst
            x, y, z = self.pos[vid]

            # compute energy
            r = ti.max(ti.math.length(ti.Vector([xc * (x - x0), yc * (y - y0), zc * (z - z0)], dt=float)), EPS)
            if r > cutoff_r:
                coeff = 0.
            if self.vert_type[vid] != 2:
                E += coeff / r 

            # compute dE / dx
            dE_dr = - coeff / (r ** 2)
            dr_dx = ti.Vector([xc * xc * x, yc * yc * y, zc * zc * z], dt=float) / r
            dE_dx = dE_dr * dr_dx
            for i in ti.static(range(3)):
                if self.vert_type[vid] != 2:
                    self.dE[vid * 3 + i] += dE_dx[i]
            
            # compute d2E / dx2
            d2E_dr2 = 2. * coeff / (r ** 3)
            d2r_dx2 = (ti.Matrix([
                [xc * xc, 0, 0],
                [0, yc * yc, 0],
                [0, 0, zc * zc],
            ], dt=float) - dr_dx.outer_product(dr_dx)) / r
            # d2E_dx2 = d2E_dr2 * dr_dx.outer_product(dr_dx) + dE_dr * d2r_dx2
            d2E_dx2 = d2E_dr2 * dr_dx.outer_product(dr_dx) # s.p.d
            if self.vert_type[vid] == 2:
                d2E_dx2[:, :] = ti.Matrix.identity(dt=float, n=3)
            for i, j in ti.static(ti.ndrange(3, 3)):
                value = d2E_dx2[i, j]
                if i == j: value += damping
                self.ddE[vid * 9 + i * 3 + j] = Triplet(value, vid * 3 + i, vid * 3 + j)
        
        ddE_offset = self.nv * 9
        exp_factor = 4.
        for eid in range(self.ne):
            # compute energy
            v1id, v2id = self.edge[eid]
            p, q = self.pos[v1id], self.pos[v2id]
            pq, r = p - q, ti.max(ti.math.length(p - q), EPS)
            l = ti.max(ti.math.length(self.pos0[v1id] - self.pos0[v2id]), EPS)
            E += 1. / exp_factor * kc * (ti.abs(r / l - 1.) ** exp_factor) * l

            # compute dE / dx
            dE_dr = kc * (ti.abs(r / l - 1.) ** (exp_factor - 1.)) * ti.math.sign(r - l)
            dr_dp = pq / r
            dE_dp = dE_dr * dr_dp
            dE_dq = -dE_dp
            for i in ti.static(range(3)):
                self.dE[v1id * 3 + i] += dE_dp[i]
                self.dE[v2id * 3 + i] += dE_dq[i]
            
            # compute d2E / dx2
            d2E_dr2 = kc * (exp_factor - 1.) * (ti.abs(r / l - 1.) ** (exp_factor - 2.)) / l
            dr2_dp2 = (ti.Matrix.identity(dt=float, n=3) - dr_dp.outer_product(dr_dp)) / r
            # d2E_dp2 = d2E_dr2 * dr_dp.outer_product(dr_dp) + dE_dr * dr2_dp2
            d2E_dp2 = d2E_dr2 * dr_dp.outer_product(dr_dp) # s.p.d
            d2E_dq2 = d2E_dp2
            d2E_dpdq = -d2E_dp2
            if self.vert_type[v1id] == 2:
                d2E_dp2[:, :] = 0. # ti.Matrix.identity(dt=float, n=3)
                d2E_dpdq[:, :] = 0.
            if self.vert_type[v2id] == 2:
                d2E_dq2[:, :] = 0. # ti.Matrix.identity(dt=float, n=3)
                d2E_dpdq[:, :] = 0.
            for i, j in ti.static(ti.ndrange(3, 3)):
                self.ddE[ddE_offset + eid * 36 + i * 3 + j + 0 ] = Triplet(d2E_dp2[i, j], v1id * 3 + i, v1id * 3 + j)
                self.ddE[ddE_offset + eid * 36 + i * 3 + j + 9 ] = Triplet(d2E_dq2[i, j], v2id * 3 + i, v2id * 3 + j)
                self.ddE[ddE_offset + eid * 36 + i * 3 + j + 18] = Triplet(d2E_dpdq[i, j], v1id * 3 + i, v2id * 3 + j)
                self.ddE[ddE_offset + eid * 36 + i * 3 + j + 27] = Triplet(d2E_dpdq[i, j], v2id * 3 + i, v1id * 3 + j)
    
        return E
    
    @ti.kernel
    def _update_kernel(self, dx: ti.types.ndarray(dtype=ti.math.vec3)):
        for vid in range(self.nv):
            if self.vert_type[vid] == 0:
                self.pos[vid] += dx[vid]
    
    def step(self, xyz0: np.ndarray, cfg: HoodOptimizerCfg, damping: float):
        E = self._compute_energy_kernel(
            cfg.rc, cfg.kc, cfg.cutoff_r, damping,
            ti.Vector(xyz0),
            ti.Vector(cfg.xyzc),
        )

        ddE = self.ddE.to_numpy()
        A = coo_matrix((ddE["value"], (ddE["row"], ddE["column"])), shape=(self.nv * 3, self.nv * 3))
        B = -self.dE.to_numpy()
        dx = spsolve(A.tocsc(), B)
        self._update_kernel(dx.reshape(self.nv, 3))

        if E > self.curr_E:
            damping *= 2.
        self.curr_E = E
        return damping
    
    def optimize(self, cfg: HoodOptimizerCfg) -> np.ndarray:
        damping = cfg.damping
        for z in tqdm.tqdm(np.linspace(0., cfg.final_z, cfg.z_count // 2).tolist() + [cfg.final_z] * (cfg.z_count // 2)):
            xyz0 = np.array([0., cfg.y_offset, z])
            damping = self.step(xyz0, cfg, damping)
        return self.pos.to_numpy()


class HoodSmoother:
    def __init__(self, mesh: trimesh.Trimesh, target_vertex_idx: list[int], vertex_exclude_idx: list[int]):
        self.smooth_threshold = 0.03 # all vertices within this distance (distance to the target vertex) will be smoothed
        self.smooth_iterations = 20 # number of smoothing iterations
        self.smooth_neighbor_weight = 0.5

        self.mesh = mesh
        self.neighbor: list[list[int]] = [set() for _ in range(mesh.vertices.shape[0])]
        for edge in mesh.edges:
            self.neighbor[edge[0]].add(edge[1])
            self.neighbor[edge[1]].add(edge[0])
        for vid in range(mesh.vertices.shape[0]):
            self.neighbor[vid] = list(self.neighbor[vid])
        
        kdtree = KDTree(mesh.vertices[target_vertex_idx].copy())
        self.vertex_exclude_idx = set(vertex_exclude_idx)
        self.smooth_vertex_idx = []
        for vid, v in enumerate(mesh.vertices):
            dist, idx = kdtree.query(v)
            if dist < self.smooth_threshold and vid not in self.vertex_exclude_idx:
                self.smooth_vertex_idx.append(vid)
    
    def smooth(self):
        prev_vert = self.mesh.vertices.copy()
        curr_vert = self.mesh.vertices.copy()
        for i in tqdm.tqdm(range(self.smooth_iterations)):
            for vid in self.smooth_vertex_idx:
                neighbor_xyz_mean = prev_vert[self.neighbor[vid]].mean(axis=0)
                curr_vert[vid] = neighbor_xyz_mean * self.smooth_neighbor_weight + prev_vert[vid] * (1 - self.smooth_neighbor_weight)
            prev_vert = curr_vert.copy()
        return curr_vert