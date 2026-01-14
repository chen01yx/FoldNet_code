import taichi as ti

import tqdm
import numpy as np
import trimesh

from garmentds.common.taichi import Triplet, GLOBAL_CREATER
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


@ti.data_oriented
class CollarOptimizer:
    def __init__(
        self,
        xy: np.ndarray,
        uv: np.ndarray,
        o: np.ndarray, face: np.ndarray, 
    ) -> None:
        assert xy.shape == uv.shape == (*o.shape, 2), f"{xy.shape} {uv.shape} {o.shape}"
        assert len(face.shape) == 2 and face.shape[1] == 3, face.shape

        edge_set: set[tuple[int, int]] = set()
        def put_in_edge_set(v1: int, v2: int):
            if v1 > v2: v1, v2 = v2, v1
            edge_set.add((v1, v2))
        for v1, v2, v3 in face:
            put_in_edge_set(v1, v2)
            put_in_edge_set(v2, v3)
            put_in_edge_set(v3, v1)
        
        edge = np.array(list(edge_set), dtype=np.int32)
        self.nv = nv = len(xy)
        self.ne = ne = len(edge)

        self.xy = GLOBAL_CREATER.VectorField(2, dtype=float, shape=(nv, ))
        self.xy.from_numpy(xy.astype(np.float32))

        self.uv = GLOBAL_CREATER.VectorField(2, dtype=float, shape=(nv, ))
        self.uv.from_numpy(uv.astype(np.float32))

        self.o = GLOBAL_CREATER.ScalarField(dtype=int, shape=(nv, ))
        self.o.from_numpy(o.astype(np.int32)) # if o == 1, then this vertice can be optimized, otherwise fixed.

        self.edge = GLOBAL_CREATER.VectorField(2, dtype=int, shape=(ne, ))
        self.edge.from_numpy(edge)

        self.dE = GLOBAL_CREATER.ScalarField(dtype=float, shape=(nv * 2, ))
        self.ddE = GLOBAL_CREATER.StructField(Triplet, shape=(nv * 2 + ne * 16))

        self.kc = np.mean(np.linalg.norm(xy[edge[0]] - xy[edge[1]], axis=1))
        self.face = face.copy()
        self.curr_E = float("inf")

    @ti.kernel
    def _compute_energy_kernel(
        self,
        damping: float,
        kc: float, 
    ) -> float:
        E = 0.
        EPS = 1e-7

        for i in self.dE:
            self.dE[i] = 0
        
        for vid in range(self.nv):
            for i in ti.static(range(2)):
                self.ddE[vid * 2 + i] = Triplet(damping, vid * 2 + i, vid * 2 + i)
        
        ddE_offset = self.nv * 2
        for eid in range(self.ne):
            # compute energy
            v1id, v2id = self.edge[eid]
            p, q = self.uv[v1id], self.uv[v2id]
            pq, r = p - q, ti.max(ti.math.length(p - q), EPS)
            l = ti.max(ti.math.length(self.xy[v1id] - self.xy[v2id]), EPS)
            E += 0.5 * kc * ((r / l) ** 2) * l

            # compute dE / dx
            dE_dr = kc * (r / l)
            dr_dp = pq / r
            dE_dp = dE_dr * dr_dp
            dE_dq = -dE_dp
            for i in ti.static(range(2)):
                if self.o[v1id] == 1:
                    self.dE[v1id * 2 + i] += dE_dp[i]
                if self.o[v2id] == 1:
                    self.dE[v2id * 2 + i] += dE_dq[i]
            
            # compute d2E / dx2
            d2E_dr2 = kc / l
            dr2_dp2 = (ti.Matrix.identity(dt=float, n=2) - dr_dp.outer_product(dr_dp)) / r
            # d2E_dp2 = d2E_dr2 * dr_dp.outer_product(dr_dp) + dE_dr * dr2_dp2
            d2E_dp2 = d2E_dr2 * dr_dp.outer_product(dr_dp) # s.p.d
            d2E_dq2 = d2E_dp2
            d2E_dpdq = -d2E_dp2
            if self.o[v1id] == 0:
                d2E_dp2[:, :] = 0. # ti.Matrix.identity(dt=float, n=3)
                d2E_dpdq[:, :] = 0.
            if self.o[v2id] == 0:
                d2E_dq2[:, :] = 0. # ti.Matrix.identity(dt=float, n=3)
                d2E_dpdq[:, :] = 0.
            for i, j in ti.static(ti.ndrange(2, 2)):
                self.ddE[ddE_offset + eid * 16 + i * 2 + j + 0 ] = Triplet(d2E_dp2[i, j], v1id * 2 + i, v1id * 2 + j)
                self.ddE[ddE_offset + eid * 16 + i * 2 + j + 4 ] = Triplet(d2E_dq2[i, j], v2id * 2 + i, v2id * 2 + j)
                self.ddE[ddE_offset + eid * 16 + i * 2 + j + 8 ] = Triplet(d2E_dpdq[i, j], v1id * 2 + i, v2id * 2 + j)
                self.ddE[ddE_offset + eid * 16 + i * 2 + j + 12] = Triplet(d2E_dpdq[i, j], v2id * 2 + i, v1id * 2 + j)
        
        return E
    
    @ti.kernel
    def _update_kernel(
        self, 
        duv: ti.types.ndarray(dtype=ti.math.vec2), 
    ):
        for vid in range(self.nv):
            if self.o[vid] == 1:
                self.uv[vid] += duv[vid]
    
    def step(self, damping: float):
        E = self._compute_energy_kernel(damping, self.kc)

        ddE = self.ddE.to_numpy()
        A = coo_matrix((ddE["value"], (ddE["row"], ddE["column"])), shape=(self.nv * 2, self.nv * 2))
        B = -self.dE.to_numpy()
        duv = spsolve(A.tocsc(), B)
        self._update_kernel(duv.reshape(self.nv, 2))

        if E > self.curr_E:
            damping *= 2.
        self.curr_E = E
        return damping

    def optimize(self, opt_step=100, damping=1e1):
        for step_idx in tqdm.tqdm(range(opt_step)):
            damping = self.step(damping)
        
        return self.uv.to_numpy()