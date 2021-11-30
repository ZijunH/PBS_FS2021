import sys
import numpy as np
import taichi as ti
import open3d as o3d
import utils
import functools 
import time

ti.init(arch=ti.cuda, device_memory_fraction=0.8)


@ti.data_oriented
class ImplicitCompressiblePressureSolver:

    def __init__(self, particle: "Particle", to_store):
        self.particle = particle
        self.N = self.particle.N

        self.v_star = ti.Vector.field(3, ti.f32, self.N)
        self.rho_star = ti.field(ti.f32, self.N)
        self.diag = ti.field(ti.f32, self.N)
        self.omega = 0.5
        self.conv_threshold = 0.001
        self.conv_iters = 730
        self.p_buf = ti.field(ti.f32, self.N)

        self.to_store = ti.static(to_store)

    @ti.func
    def disc_div(self, p_i, do_div):
        k_sum = 0.0
        for neighbor_i in range(self.particle.neighbors_num[p_i]):
            n_i = self.particle.neighbors[p_i, neighbor_i]
            n_volume = self.particle.ret_volume(n_i)
            n_diff = do_div[n_i] - do_div[p_i]
            n_kernel_d = self.particle.kernel_cubic_d(self.particle.x[p_i] - self.particle.x[n_i], self.particle.cutoff_radius)
            k_sum += n_volume * n_diff.dot(n_kernel_d)
        return k_sum

    def icps(self):
        print("icps begin")
        self.icps_prepare()

        num_iters = 0
        residual = self.conv_threshold - 1
        while num_iters < self.conv_iters and residual < self.conv_threshold:
            # this is actually the residual of hte previous iteration, 
            # so an extra iteration is run
            residual = self.icps_iter()
            print(residual)
            num_iters += 1

    @ti.kernel
    def icps_prepare(self):
        for p_i in range(self.N):
            # Alg 2, line 2
            self.v_star[p_i] = self.particle.v[p_i] + self.particle.dt * (self.particle.a_friction[p_i] + self.particle.a_other[p_i])

        for p_i in range(self.N):
            p_volume = self.particle.ret_volume(p_i)
            div_v = self.disc_div(p_i, self.v_star)
            self.rho_star[p_i] = self.particle.rho[p_i] - self.particle.dt * self.particle.rho[p_i] * div_v

            # Alg 2, line 3
            j_first_sum = ti.cast(0.0, ti.f32)
            j_second_sum = ti.Vector([0.0, 0.0, 0.0])
            b_sum = ti.Vector([0.0, 0.0, 0.0])
            k_sum = ti.Vector([0.0, 0.0, 0.0])

            for neighbor_i in range(self.particle.neighbors_num[p_i]):
                n_i = self.particle.neighbors[p_i, neighbor_i]
                n_volume = self.particle.ret_volume(n_i)
                n_kernel_d = self.particle.kernel_cubic_d(self.particle.x[p_i] - self.particle.x[n_i], self.particle.cutoff_radius)
                if(self.particle.material_type[n_i] == Particle.Materials.FLUID):
                    j_first_sum += p_volume * n_volume * n_kernel_d.norm_sqr()
                    j_second_sum += n_volume * n_kernel_d
                if(self.particle.material_type[n_i] == Particle.Materials.BOUNDARY):
                    b_sum += n_volume * n_kernel_d
                k_sum += n_volume * n_kernel_d

            self.diag[p_i] = -self.particle.rho_rest[p_i] / self.particle.lame_lambda[p_i] - self.particle.dt ** 2 * j_first_sum \
                             - self.particle.dt ** 2 * (j_second_sum + self.particle.psi * b_sum).dot(k_sum) 

    @ti.kernel
    def icps_iter(self) -> ti.f32:
        # Alg 2, line 8
        for p_i in range(self.N):
            self.particle.calc_d_p(p_i)
        residual = 0.0
        # TODO: verify semi-implicit or explicit update
        for p_i in range(self.N):
            # Eq. 6 for laplace
            laplace_p = self.disc_div(p_i, self.particle.d_p)
            # Alg 2, line 10
            lhs = -self.particle.rho_rest[p_i] / self.particle.lame_lambda[p_i] * self.particle.p[p_i] + self.particle.dt ** 2 * laplace_p
            # Alg 2, line 11
            residual += (self.particle.rho_rest[p_i] - self.rho_star[p_i] - lhs) / (self.particle.rho_rest[p_i] - self.rho_star[p_i])
            self.p_buf[p_i] = self.to_store[p_i] + self.omega / self.diag[p_i] * (self.particle.rho_rest[p_i] - self.rho_star[p_i] - lhs)
        # TODO: this is a stupid way to swap
        for p_i in range(self.N):
            self.to_store[p_i], self.p_buf[p_i] = self.p_buf[p_i], self.to_store[p_i]
        return residual / self.N


@ti.data_oriented
class ImplicitShearSolver:

    I_3 = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    EPS = 7./3 - 4./3 - 1
    EPS2 = EPS * EPS

    class ToCalcLHS:

        X = 0
        P = 1
        S = 2


    def __init__(self, particle: "Particle", to_store):
        self.particle = particle
        self.N = self.particle.N
        self.v_star = ti.Vector.field(3, float, self.N)

        self.temp_stress_tensor = ti.Matrix.field(3, 3, ti.f32, self.N)

        # result
        self.rhs = ti.Vector.field(3, float, self.N)
        self.x = ti.static(to_store)
        self.lhs_in = ti.Vector.field(3, float, self.N)
        self.lhs_out = ti.Vector.field(3, float, self.N)

        self.max_iter = 3 * 10 * self.N
        self.tol = 1e-5

        self.r = ti.Vector.field(3, float, self.N)
        self.r0 = ti.Vector.field(3, float, self.N)
        self.v = ti.Vector.field(3, float, self.N)
        self.p = ti.Vector.field(3, float, self.N)
        self.y = ti.Vector.field(3, float, self.N)
        self.s = ti.Vector.field(3, float, self.N)
        self.rho = ti.field(float, ())
        self.a = ti.field(float, ())
        self.w = ti.field(float, ())
        self.b = ti.field(float, ())
        self.r0_sqr_norm = ti.field(float, ())
        self.rhs_sqr_norm = ti.field(float, ())
        self.tol2 = ti.field(float, ())

    def iss(self):
        print("iss begin")
        self.rhs_func()
        self.BiCGSTAB()

    @ti.func
    def lhs_func(self, to_calc_v: ti.i32):
        if(to_calc_v == self.ToCalcLHS.X):
            to_calc = ti.static(self.x)
        elif(to_calc_v == self.ToCalcLHS.P):
            to_calc = ti.static(self.p)
        elif(to_calc_v == self.ToCalcLHS.S):
            to_calc = ti.static(self.s)

        for p_i in range(self.N):
            b_d = self.disc_grad(p_i, to_calc)
            F_res = b_d @ self.particle.F[p_i]
            self.temp_stress_tensor[p_i] = self.particle.lame_G[p_i] * (F_res + F_res.transpose())
        for p_i in range(self.N):
            pre_sub = self.disc_div_stress(p_i) * self.particle.dt / self.particle.rho[p_i]
            self.lhs_out[p_i] = to_calc[p_i] - pre_sub

    @ti.kernel
    def rhs_func(self):
        for p_i in range(self.N):
            self.v_star[p_i] = self.particle.v[p_i] + self.particle.dt * (self.particle.a_friction[p_i] + self.particle.a_other[p_i] + self.particle.a_lambda[p_i])
        for p_i in range(self.N):
            v_d = self.disc_grad(p_i, self.v_star)
            F_star = self.particle.F[p_i] + self.particle.dt * v_d @ self.particle.F[p_i]
            F_star_T = F_star.transpose()
            self.temp_stress_tensor[p_i] = self.particle.lame_G[p_i] * (F_star + F_star_T - 2 * self.I_3) 
        for p_i in range(self.N):
            self.rhs[p_i] = self.disc_div_stress(p_i) / self.particle.rho[p_i]

    @ti.func
    def disc_div_stress(self, p_i):
        j_sum = ti.Vector([0.0, 0.0, 0.0])
        b_sum = ti.Vector([0.0, 0.0, 0.0])
        for neighbor_i in range(self.particle.neighbors_num[p_i]):
            n_i = self.particle.neighbors[p_i, neighbor_i]
            n_volume = self.particle.ret_volume(n_i)
            if(self.particle.material_type[n_i] == Particle.Materials.FLUID):
                n_kernel_d_ij = self.particle.kernel_cubic_d(self.particle.x[p_i] - self.particle.x[n_i], self.particle.cutoff_radius)
                n_kernel_d_ji = self.particle.kernel_cubic_d(self.particle.x[n_i] - self.particle.x[p_i], self.particle.cutoff_radius)
                term_1 = self.temp_stress_tensor[n_i] @ (-n_volume * self.particle.L_i[n_i] @ n_kernel_d_ji)
                term_2 = self.temp_stress_tensor[p_i] @ (n_volume * self.particle.L_i[p_i] @ n_kernel_d_ij)
                j_sum += term_1 + term_2
            if(self.particle.material_type[n_i] == Particle.Materials.BOUNDARY):
                n_kernel_d = self.particle.kernel_cubic_d(self.particle.x[p_i] - self.particle.x[n_i], self.particle.cutoff_radius)
                b_sum += n_volume * self.particle.L_i[p_i] @ n_kernel_d
        mirrored_cauchy = 1 / 3 * self.temp_stress_tensor[p_i].trace() * self.I_3
        return j_sum + mirrored_cauchy @ b_sum 

    @ti.func
    def disc_grad(self, p_i, to_grad):
        L_i_T = self.particle.L_i[p_i].transpose()

        # Eq. 17
        d_v_j = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        d_v_b = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        for neighbor_i in range(self.particle.neighbors_num[p_i]):
            n_i = self.particle.neighbors[p_i, neighbor_i]
            n_volume = self.particle.ret_volume(n_i)
            n_diff = to_grad[n_i] - to_grad[p_i]
            n_kernel_d = self.particle.kernel_cubic_d(self.particle.x[p_i] - self.particle.x[n_i], self.particle.cutoff_radius)
            if(self.particle.material_type[n_i] == Particle.Materials.FLUID):
                d_v_j += n_diff.outer_product(n_volume * n_kernel_d)
            if(self.particle.material_type[n_i] == Particle.Materials.BOUNDARY):
                d_v_b += n_diff.outer_product(n_volume * n_kernel_d)

        # after Eq. 17
        tild_d_v = d_v_j @ L_i_T + 1 / 3 * (d_v_b @ L_i_T).trace() * self.I_3
        tild_d_v_T = tild_d_v.transpose()

        # Eq. 18
        tild_R = 1 / 2 * (tild_d_v - tild_d_v_T)
        prim_V = 1 / 3 * (d_v_j + d_v_b).trace() * self.I_3
        tild_S = 1 / 2 * (tild_d_v + tild_d_v_T) - 1 / 3 * tild_d_v.trace() * self.I_3

        return tild_R + prim_V + tild_S

    @ti.kernel
    def BiCGSTAB_prepare(self):
        self.rho[None] = 1.0
        self.a[None] = 1.0
        self.w[None] = 1.0
        self.b[None] = 1.0
        self.r0_sqr_norm[None] = 0.0
        self.rhs_sqr_norm[None] = 0.0
        for p_i in range(self.N):
            self.v[p_i] = ti.Vector([0, 0, 0])
            self.p[p_i] = ti.Vector([0, 0, 0])
            self.x[p_i] = ti.Vector([0, 0, 0])
            self.r[p_i] = self.rhs[p_i] - self.lhs_out[p_i]
            self.r0[p_i] = self.r[p_i]
            self.r0_sqr_norm[None] += self.r0[p_i].norm_sqr()
            self.rhs_sqr_norm[None] += self.rhs[p_i].norm_sqr()
        self.tol2[None] = self.tol * self.tol * self.rhs_sqr_norm[None]
        
        for p_i in range(self.N):
            b_d = self.disc_grad(p_i, self.x)
            F_res = b_d @ self.particle.F[p_i]
            self.temp_stress_tensor[p_i] = self.particle.lame_G[p_i] * (F_res + F_res.transpose())
        for p_i in range(self.N):
            pre_sub = self.disc_div_stress(p_i) * self.particle.dt / self.particle.rho[p_i]
            self.lhs_out[p_i] = self.x[p_i] - pre_sub


    @ti.kernel
    def BiCGSTAB_iter(self, i: ti.i32) -> ti.i32: 
        r_sqr_norm = 0.0
        for p_i in range(self.N):
            r_sqr_norm += self.r[p_i].norm_sqr()
        res = 0
        if(r_sqr_norm > self.tol2[None] and i < self.max_iter):
            res = 1
        else:
            rho_old = self.rho[None]
            self.rho[None] = 0.0
            for p_i in range(self.N):
                self.rho[None] += self.r0[p_i].dot(self.r[p_i])
            if(ti.abs(self.rho[None]) < self.EPS2):

                for p_i in range(self.N):
                    b_d = self.disc_grad(p_i, self.x)
                    F_res = b_d @ self.particle.F[p_i]
                    self.temp_stress_tensor[p_i] = self.particle.lame_G[p_i] * (F_res + F_res.transpose())
                for p_i in range(self.N):
                    pre_sub = self.disc_div_stress(p_i) * self.particle.dt / self.particle.rho[p_i]
                    self.lhs_out[p_i] = self.x[p_i] - pre_sub
                
                self.r0_sqr_norm[None] = 0.0
                for p_i in range(self.N):
                    self.r[p_i] = self.rhs[p_i] - self.lhs_out[p_i]
                    self.r0[p_i] = self.r[p_i]
                    self.r0_sqr_norm[None] += self.r[p_i].norm_sqr()
                self.rho[None] = self.r0_sqr_norm[None]
            self.b[None] = (self.rho[None] - rho_old) * (self.a[None] / self.w[None])
            for p_i in range(self.N):
                self.p[p_i] = self.r[p_i] + self.b[None] * (self.p[p_i] - self.w[None] * self.v[p_i])

            for p_i in range(self.N):
                b_d = self.disc_grad(p_i, self.p)
                F_res = b_d @ self.particle.F[p_i]
                self.temp_stress_tensor[p_i] = self.particle.lame_G[p_i] * (F_res + F_res.transpose())
            for p_i in range(self.N):
                pre_sub = self.disc_div_stress(p_i) * self.particle.dt / self.particle.rho[p_i]
                self.lhs_out[p_i] = self.p[p_i] - pre_sub

            dot_sum = 0.0
            for p_i in range(self.N):
                dot_sum += self.r0[p_i].dot(self.lhs_out[p_i])
            self.a[None] = self.rho[None] / dot_sum
            for p_i in range(self.N):
                self.s[p_i] = self.r[p_i] - self.a[None] * self.lhs_out[p_i]

            for p_i in range(self.N):
                b_d = self.disc_grad(p_i, self.s)
                F_res = b_d @ self.particle.F[p_i]
                self.temp_stress_tensor[p_i] = self.particle.lame_G[p_i] * (F_res + F_res.transpose())
            for p_i in range(self.N):
                pre_sub = self.disc_div_stress(p_i) * self.particle.dt / self.particle.rho[p_i]
                self.lhs_out[p_i] = self.s[p_i] - pre_sub

            tmp = 0.0
            for p_i in range(self.N):
                tmp += self.lhs_out[p_i].norm_sqr()
            if(tmp > 0):
                self.w[None] = 0.0
                for p_i in range(self.N):
                    self.w[None] += self.lhs_out[p_i].dot(self.s[p_i])
                self.w[None] /= tmp
            else:
                self.w[None] = 0.0
            for p_i in range(self.N):
                self.x[p_i] += self.a[None] * self.y[p_i] + self.w[None] * self.s[p_i]
                self.r[p_i] = self.s[p_i] - self.w[None] * self.lhs_out[p_i]
        return res

    # https://eigen.tuxfamily.org/dox/BiCGSTAB_8h_source.html
    def BiCGSTAB(self):
        self.BiCGSTAB_prepare()
        
        i = 0
        restarts = 0
        while True:
            if(self.BiCGSTAB_iter(i) == 1):
                break
            i += 1


@ti.data_oriented
class Particle:

    class Materials:
        FLUID = 0
        BOUNDARY = 1

    def __init__(self, N, v_range, paused = False, substep = 1, dt = 1e-3):
        self.N = N
        self.range = v_range
        self.paused = paused
        self.substep_num = substep
        self.dt = dt

        # material properties
        self.young = 140
        self.poisson = 0.2
        self.harden = 10
        self.rho_0 = 400
        self.psi = 1.5 # motivated by Akinci et al.
        self.clamp_l = 1 - 0.025
        self.clamp_h = 1 + 0.0075
        self.friction = 1
        self.volume_scaling = 0.8

        # particle properties
        self.m = 0.1
        self.x = ti.Vector.field(3, ti.f32, self.N)
        self.p = ti.field(ti.f32, self.N)
        self.d_p = ti.Vector.field(3, ti.f32, self.N)
        self.material_type = ti.field(ti.i32, self.N)
        self.v = ti.Vector.field(3, ti.f32, self.N)
        self.rho = ti.field(ti.f32, self.N)
        self.rho_rest = ti.field(ti.f32, self.N)
        self.F = ti.Matrix.field(3, 3, ti.f32, self.N) # deformation gradient
        self.L_i = ti.Matrix.field(3, 3, float, self.N)
        self.lame_lambda = ti.field(float, self.N)
        self.lame_G = ti.field(float, self.N)

        # accelerations
        self.a_other = ti.Vector.field(3, ti.f32, self.N)
        self.a_friction = ti.Vector.field(3, ti.f32, self.N)
        self.a_lambda = ti.Vector.field(3, ti.f32, self.N)
        self.a_G = ti.Vector.field(3, ti.f32, self.N)

        # solvers
        self.icps_solver = ImplicitCompressiblePressureSolver(self, self.p)
        self.iss_solver = ImplicitShearSolver(self, self.a_G)

        # neighbor search functionality
        self.max_neighbors = 100
        self.neighbors = ti.field(ti.i32, (self.N, self.max_neighbors))
        self.neighbors_num = ti.field(ti.i32, self.N)
        self.max_particles_per_cell = 100 # should be a upper bound somewhere but we will leave it this way
        self.grid_side = 0.025 # must divide exactly
        self.grid_size = ((np.array(self.range)[:, 1] - np.array(self.range)[:, 0]) / self.grid_side).astype(int)
        self.grid = ti.field(ti.i32, (*self.grid_size, self.max_particles_per_cell))
        self.grid_num = ti.field(ti.i32, (*self.grid_size, ))
        self.grid_search_offset_range = (-1, 1 + 1)
        self.cutoff_radius = min(self.grid_side * -self.grid_search_offset_range[0], self.grid_side * (self.grid_search_offset_range[1] - 1))
        self.cutoff_radius_sqr = self.cutoff_radius * self.cutoff_radius

    @ti.func
    def compute_grid_index(self, pos):
        return (pos / self.grid_side).cast(int)

    @ti.func
    def is_in_grid(self, grid_cell):
        res = True
        for i in ti.static(range(3)):
            res &= 0 <= grid_cell[i] < self.grid_size[i]
        return res

    @ti.kernel
    def neighbor_search(self):
        for p_i in range(self.N):
            pos_i = self.x[p_i]
            num_neighbors = 0
            g_i = self.compute_grid_index(pos_i)
            for offs in ti.static(ti.grouped(ti.ndrange(*(self.grid_search_offset_range, ) * 3))):
                g_j = g_i + offs
                if self.is_in_grid(g_j):
                    for g_j_i in range(self.grid_num[g_j]):
                        p_j = self.grid[g_j, g_j_i]
                        if num_neighbors < self.max_neighbors and p_i != p_j and (pos_i - self.x[p_j]).norm_sqr() < self.cutoff_radius_sqr:
                            self.neighbors[p_i, num_neighbors] = p_j
                            num_neighbors += 1
            self.neighbors_num[p_i] = num_neighbors

    @ti.kernel
    def allocate_grid(self):
        for p_i in range(self.N):
            pos_i = self.x[p_i]
            g_i = self.compute_grid_index(pos_i)
            # atomical write and return
            num_grid_i = self.grid_num[g_i].atomic_add(1)
            self.grid[g_i, num_grid_i] = p_i

    @ti.func
    def kernel_cubic(self, v, h):
        # https://pysph.readthedocs.io/en/latest/reference/kernels.html
        # http://web.cse.ohio-state.edu/~wang.3602/courses/cse3541-2017-fall/08-SPH.pdf
        q = v.norm() / h
        smoothing = 1. / (np.pi * h ** 3)
        q = v.norm() / h
        res = 0.0
        if q <= 1.0:
            res = smoothing * (1 - 1.5 * q ** 2 + 0.75 * q ** 3)
        elif q < 2.0:
            res = smoothing * 0.25 * (2 - q) ** 3
        return res

    @ti.func
    def kernel_cubic_d(self, v, h):
        smoothing = 1. / (np.pi * h ** 3)
        v_norm = v.norm()
        q = v_norm / h
        res = ti.cast(0.0, ti.f32)
        if q < 1.0:
            res = (smoothing / h) * (-3 * q + 2.25 * q ** 2)
        elif q < 2.0:
            res = -0.75 * (smoothing / h) * (2 - q) ** 2
        else:
            res = 0.0
        return res * v / (v_norm * h)

    @ti.func
    def ret_volume(self, p_i):
        ret_v = 0.0
        if(self.material_type[p_i] == self.Materials.FLUID):
            ret_v = self.m / self.rho[p_i]
        if(self.material_type[p_i] == self.Materials.BOUNDARY):
            kernel_sum = 0.0
            for neighbor_i in range(self.neighbors_num[p_i]):
                n_i = self.neighbors[p_i, neighbor_i]
                if(self.material_type[n_i] == self.Materials.BOUNDARY):
                    kernel_sum += self.kernel_cubic(self.x[p_i] - self.x[n_i], self.cutoff_radius)
            ret_v = self.volume_scaling / kernel_sum
        return ret_v

    # Eq. 7
    @ti.func
    def calc_d_p(self, p_i):
        j_sum = ti.Vector([0.0, 0.0, 0.0])
        b_sum = ti.Vector([0.0, 0.0, 0.0])
        for neighbor_i in range(self.neighbors_num[p_i]):
            n_i = self.neighbors[p_i, neighbor_i]
            n_volume = self.ret_volume(n_i)
            n_kernel_d = self.kernel_cubic_d(self.x[p_i] - self.x[n_i], self.cutoff_radius)
            if(self.material_type[n_i] == self.Materials.FLUID):
                j_sum += (self.p[p_i] + self.p[n_i]) * n_volume * n_kernel_d
            if(self.material_type[n_i] == self.Materials.BOUNDARY):
                b_sum += n_volume * n_kernel_d
        self.d_p[p_i] = j_sum + self.psi * self.p[p_i] * b_sum

    @ti.func
    def calc_L_i(self, p_i):
        # Eq. 15
        inv_L_i = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        for neighbor_i in range(self.neighbors_num[p_i]):
            n_i = self.neighbors[p_i, neighbor_i]
            n_volume = self.ret_volume(n_i)
            n_diff = self.x[n_i] - self.x[p_i]
            n_kernel_d = self.kernel_cubic_d(self.x[p_i] - self.x[n_i], self.cutoff_radius)
            inv_L_i += n_volume * n_kernel_d.outer_product(n_diff)
        self.L_i[p_i] = inv_L_i.inverse()

    @ti.func
    def calc_lame(self, p_i):
        # Eq. 22, 23
        exp_val = ti.exp(self.harden * (self.rho_rest[p_i] - self.rho_0) / self.rho_rest[p_i])
        self.lame_lambda[p_i] = self.young * self.poisson / ((1 + self.poisson) * (1 - 2 * self.poisson)) * exp_val
        self.lame_G[p_i] = self.young / (2 * (1 + self.poisson)) * exp_val

    @ti.func
    def calc_rhos(self, p_i):
        # Eq. 20, 21
        cur_rho = 0.0
        print("n", p_i, self.neighbors_num[p_i])
        for neighbor_i in range(self.neighbors_num[p_i]):
            n_i = self.neighbors[p_i, neighbor_i]
            n_kernel_d = self.kernel_cubic(self.x[p_i] - self.x[n_i], self.cutoff_radius)
            cur_rho += self.m * n_kernel_d
        self.rho[p_i] = cur_rho
        self.rho_rest[p_i] = cur_rho * ti.abs(self.F[p_i].determinant())

    @ti.func
    def sig_clamp(self, sigma, loc):
        sigma[loc, loc] = ti.max(self.clamp_l, min(sigma[loc, loc], self.clamp_h))

    @ti.func
    def calc_F(self, p_i):
        d_v = self.a_other[p_i] + self.a_friction[p_i] + self.a_lambda[p_i] + self.a_G[p_i]
        prim_F = self.F[p_i] + self.dt * d_v * self.F[p_i]
        U, sigma, V = ti.svd(prim_F, float)
        self.sig_clamp(sigma, 0)
        self.sig_clamp(sigma, 1)
        self.sig_clamp(sigma, 2)
        self.F[p_i] = V @ sigma @ V.transpose()

    @ti.func
    def calc_a_other(self, p_i):
        self.a_other[p_i] = ti.Vector([0, 0, -9.81])

    @ti.func
    def calc_a_friction(self, p_i):
        # Eq. 24, 25
        d_ii_b_sum = 0.0
        a_b_sum = ti.Vector([0.0, 0.0, 0.0])
        for neighbor_i in range(self.neighbors_num[p_i]):
            n_i = self.neighbors[p_i, neighbor_i]
            n_volume = self.ret_volume(n_i)
            n_diff = self.x[p_i] - self.x[n_i]
            n_kernel_d = self.kernel_cubic_d(self.x[p_i] - self.x[n_i], self.cutoff_radius)
            common = n_volume * n_diff.dot(n_kernel_d) / (n_diff.norm_sqr() * 0.01 * self.grid_side ** 2)
            d_ii_b_sum += common
            a_b_sum += common * self.v[n_i]
        d_ii = 1 - self.dt * self.friction * d_ii_b_sum
        self.a_friction[p_i] = 1 / (d_ii * self.dt) * (self.v[p_i] + self.dt * self.a_other[p_i] - self.dt * self.friction * a_b_sum)

    @ti.kernel
    def init(self):
        side_len = (int)(self.N ** (1. / 3))
        for p_i in range(self.N):
            i = (int)(p_i / (side_len * side_len))
            j = (int)((p_i % (side_len * side_len)) / side_len)
            k = (int)(p_i % side_len)
            self.x[p_i] = ti.Vector([i / side_len, j / side_len, k / side_len])
            self.p[p_i] = 10
            self.v[p_i] = ti.Vector([0, 0, 0])
            self.F[p_i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

            if(i == 0 or j == 0 or j == side_len - 1 or k == 0 or k == side_len - 1):
                self.material_type[p_i] = self.Materials.BOUNDARY
            else:
                self.material_type[p_i] = self.Materials.FLUID

    @ti.kernel
    def prologue(self):
        for i, j, k in ti.ndrange(*self.grid_size):
            self.grid_num[i, j, k] = 0

    @ti.kernel
    def prereq_update(self):
        for p_i in range(self.N):
            self.calc_rhos(p_i)
            self.calc_lame(p_i)
            self.calc_L_i(p_i)
            self.calc_a_other(p_i)
            self.calc_a_friction(p_i)
        print(self.rho[35555], self.rho_rest[35555])
        print(self.lame_G[35555], self.lame_lambda[35555])
        print(self.L_i[35555])
        print(self.a_other[35555])
        print(self.a_friction[35555])

    @ti.kernel
    def a_lambda_update(self):
        for p_i in range(self.N):
            self.calc_d_p(p_i)
            self.a_lambda[p_i] = - 1 / self.rho[p_i] * self.d_p[p_i]

    @ti.kernel
    def v_x_update(self):
        for p_i in range(self.N):
            self.v[p_i] += self.dt * (self.a_other[p_i] + self.a_friction[p_i] + self.a_lambda[p_i] + self.a_G[p_i])
            self.calc_F(p_i)
            self.x[p_i] += self.dt * self.v[p_i]

    @ti.kernel
    def print_neighbor_info(self):
        max_neighbor = -1
        total_neighbor = 0
        for p_i in range(self.N):
            ti.atomic_max(max_neighbor, self.neighbors_num[p_i])
            total_neighbor += self.neighbors_num[p_i]
        print("max neighbors", max_neighbor)
        print("avg neighbors", total_neighbor / self.N)

        max_grid = -1
        total_grid = 0
        for i, j, k in ti.ndrange(*self.grid_size):
            ti.atomic_max(max_grid, self.grid_num[i, j, k])
            total_grid += self.grid_num[i, j, k]
        print("max grid", max_grid)
        print("avg grid", total_grid / (self.grid_size[0] * self.grid_size[1] * self.grid_size[2]))


    def substep(self):
        self.prologue()
        self.allocate_grid()
        self.neighbor_search()
        self.print_neighbor_info()

        self.prereq_update()

        # Alg 1, line 7
        self.icps_solver.icps()
        self.a_lambda_update()

        # Alg 1, line 8
        self.iss_solver.iss()
        self.v_x_update()

    def step(self):
        if self.paused:
            return
        for _ in range(self.substep_num):
            self.substep()

    def toggle_paused(self):
        self.paused = not self.paused


class Render():
    def __init__(self, N, obj):
        # https://github.com/isl-org/Open3D/issues/572
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.obj = obj

    def init(self):
        # converted to flaot64 for speed
        self.point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.obj.x.to_numpy(dtype=np.float64)))
        self.point_cloud.paint_uniform_color([1, 0.706, 0])
        self.vis.add_geometry(self.point_cloud)

    def update(self):
        self.point_cloud.points = o3d.utility.Vector3dVector(self.obj.x.to_numpy(dtype=np.float64))
        self.point_cloud.paint_uniform_color([1, 0.706, 0])
        self.vis.update_geometry(self.point_cloud)

    def create_window(self, *args):
        return self.vis.create_window(*args)
    
    def update_renderer(self, *args):
        return self.vis.update_renderer(*args)

    def register_key_callback(self, *args):
        return self.vis.register_key_callback(*args)

    def get_view_control(self, *args):
        return self.vis.get_view_control(*args)

    def poll_events(self, *args):
        return self.vis.poll_events(*args)



print("Starting")
N  = 1000000
v_range = ((0, 1), (0, 1), (0, 1))
print("\tParticle")
particle = Particle(N, v_range)
print("\tRenderer")
vis = Render(N, particle)


def init():
    particle.init()
    vis.init()

def reset_sim(R_vis):
    init()

def pause_sim(R_vis):
    particle.toggle_paused()

vis.create_window()
vis.register_key_callback(ord("R"), reset_sim)
vis.register_key_callback(ord(" "), pause_sim)

ctr = vis.get_view_control()
ctr.set_lookat([0.0, 0.5, 0.0])
ctr.set_up([0.0, 1.0, 0.0])
reset_sim(vis)

while True:
    particle.step()
    vis.update()
    if not vis.poll_events():
        break
    vis.update_renderer()
