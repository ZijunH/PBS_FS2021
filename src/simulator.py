import numpy as np
import taichi as ti
import open3d as o3d
import time
import os
import json

ti.init(arch=ti.cuda, device_memory_fraction=0.9)


@ti.data_oriented
class ImplicitCompressiblePressureSolver:

    def __init__(self, particle: "Particle", to_store):
        self.particle = particle
        self.N_snow = particle.N_snow
        self.N_bound = particle.N_bound
        self.N_tot = particle.N_snow + particle.N_bound

        self.v_star = ti.Vector.field(3, float, self.N_tot)
        self.rho_star = ti.field(float, self.N_snow)
        self.diag = ti.field(float, self.N_snow)
        self.omega = 0.5
        self.conv_threshold = 1e-3
        self.conv_iters = 500000
        self.p_buf = ti.field(float, self.N_snow)

        self.to_store = ti.static(to_store)

    @ti.func
    def disc_div(self, p_i, do_div):
        # Eq. 6
        k_sum = 0.0
        for neighbor_i in range(self.particle.neighbors_num[p_i]):
            n_i = self.particle.neighbors[p_i, neighbor_i]
            n_volume = self.particle.ret_volume(n_i)
            n_diff = ti.Vector([0.0, 0.0, 0.0])
            if(self.particle.material_type[n_i] == Particle.Materials.FLUID):
                n_diff = do_div[n_i] - do_div[p_i]
            elif(self.particle.material_type[n_i] == Particle.Materials.BOUNDARY):
                if(do_div.shape[0] <= p_i):
                    n_diff = -do_div[p_i]
                else:
                    n_diff = do_div[n_i] - do_div[p_i]
            n_kernel_d = self.particle.kernel_cubic_d(self.particle.x[p_i] - self.particle.x[n_i], self.particle.cutoff_radius)
            k_sum += n_volume * n_diff.dot(n_kernel_d)
        return k_sum


    def icps(self):
        print("icps begin")
        self.icps_prepare()

        num_iter = 0
        residual = self.conv_threshold + 1
        while num_iter < self.conv_iters and residual >= self.conv_threshold:
            # this is actually the residual of hte previous iteration, 
            # so an extra iteration is run
            residual = self.icps_iter()
            num_iter += 1
        print("    ran iter", num_iter, "final residual%", residual)

    @ti.kernel
    def icps_prepare(self):
        for p_i in range(self.N_tot):
            # Alg 2, line 2
            if(self.particle.material_type[p_i] == Particle.Materials.FLUID):
                self.v_star[p_i] = self.particle.v[p_i] + self.particle.dt * (self.particle.a_friction[p_i] + self.particle.a_other[p_i])
            elif(self.particle.material_type[p_i] == Particle.Materials.BOUNDARY):
                self.v_star[p_i] = self.particle.v[p_i]
        for p_i in range(self.N_snow):
            # Alg 2, line 3
            p_volume = self.particle.ret_volume(p_i)
            div_v = self.disc_div(p_i, self.v_star)
            self.rho_star[p_i] = self.particle.rho[p_i] - self.particle.dt * self.particle.rho[p_i] * div_v
            # Eq. 8
            j_first_sum = ti.cast(0.0, float)
            j_second_sum = ti.Vector([0.0, 0.0, 0.0])
            b_sum = ti.Vector([0.0, 0.0, 0.0])
            for neighbor_i in range(self.particle.neighbors_num[p_i]):
                n_i = self.particle.neighbors[p_i, neighbor_i]
                n_volume = self.particle.ret_volume(n_i)
                n_kernel_d = self.particle.kernel_cubic_d(self.particle.x[p_i] - self.particle.x[n_i], self.particle.cutoff_radius)
                if(self.particle.material_type[n_i] == Particle.Materials.FLUID):
                    j_first_sum += p_volume * n_volume * n_kernel_d.norm_sqr()
                    j_second_sum += n_volume * n_kernel_d
                elif(self.particle.material_type[n_i] == Particle.Materials.BOUNDARY):
                    b_sum += n_volume * n_kernel_d
            self.diag[p_i] = -self.particle.rho_rest[p_i] / self.particle.lame_lambda[p_i] - self.particle.dt ** 2 * j_first_sum \
                             - self.particle.dt ** 2 * (j_second_sum + self.particle.psi * b_sum).dot(j_second_sum + b_sum) 

    @ti.kernel
    def icps_iter(self) -> float:
        # Alg 2, line 8
        for p_i in range(self.N_snow):
            self.particle.calc_d_p(p_i)
        lhs_sum = 0.0
        rhs_sum = 0.0
        for p_i in range(self.N_snow):
            # Eq. 6 for laplace
            laplace_p = self.disc_div(p_i, self.particle.d_p)
            # Alg 2, line 10
            lhs = -self.particle.rho_rest[p_i] / self.particle.lame_lambda[p_i] * self.particle.p[p_i] + self.particle.dt ** 2 * laplace_p
            # Alg 2, line 11
            lhs_sum += (self.particle.rho_rest[p_i] - self.rho_star[p_i] - lhs) ** 2
            rhs_sum += (self.particle.rho_rest[p_i] - self.rho_star[p_i]) ** 2
            self.p_buf[p_i] = self.to_store[p_i] + self.omega / self.diag[p_i] * (self.particle.rho_rest[p_i] - self.rho_star[p_i] - lhs)
        for p_i in range(self.N_snow):
            self.to_store[p_i], self.p_buf[p_i] = self.p_buf[p_i], self.to_store[p_i]
        return (lhs_sum / (rhs_sum + 0.00001)) ** (1. / 2.)


# https://eigen.tuxfamily.org/dox/BiCGSTAB_8h_source.html
@ti.data_oriented
class BiCGSTAB:

    EPS = 7./3 - 4./3 - 1
    EPS2 = EPS * EPS

    def __init__(self, N, lhs_func, rhs, x, max_iter, tol, with_guess=False):
        self.lhs_func = lhs_func
        self.rhs = ti.static(rhs)
        self.max_iter = max_iter
        self.x = ti.static(x)
        self.tol = tol
        self.N = N
        self.with_guess = with_guess

        self.r = ti.Vector.field(3, float, self.N)
        self.r0 = ti.Vector.field(3, float, self.N)
        self.v = ti.Vector.field(3, float, self.N)
        self.p = ti.Vector.field(3, float, self.N)
        self.s = ti.Vector.field(3, float, self.N)
        self.t = ti.Vector.field(3, float, self.N)
        self.temp = ti.Vector.field(3, float, self.N)

    @ti.kernel
    def prepare(self):
        for i in range(self.N):
            if(not self.with_guess):
                self.x[i] = ti.Vector([0.0, 0.0, 0.0])

            self.r[i] = ti.Vector([0.0, 0.0, 0.0])
            self.r0[i] = ti.Vector([0.0, 0.0, 0.0])
            self.v[i] = ti.Vector([0.0, 0.0, 0.0])
            self.p[i] = ti.Vector([0.0, 0.0, 0.0])
            self.s[i] = ti.Vector([0.0, 0.0, 0.0])
            self.t[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def sqnorm(self, to_calc: ti.template()) -> float:
        res = 0.0
        for i in range(self.N):
            res += to_calc[i].norm_sqr()
        return res

    @ti.kernel
    def dot(self, to_calc1: ti.template(), to_calc2: ti.template()) -> float:
        res = 0.0
        for i in range(self.N):
            res += to_calc1[i].dot(to_calc2[i])
        return res

    def calc_res(self):
        self.lhs_func(self.x, self.temp)
        self.sub(self.rhs, self.temp, self.r)

    @ti.kernel
    def sub(self, sub_from: ti.template(), sub_to: ti.template(), store: ti.template()) -> float:
        for i in range(self.N):
            store[i] = sub_from[i] - sub_to[i]

    @ti.kernel
    def copy_over(self, from_v: ti.template(), to_v: ti.template()) -> float:
        for i in range(self.N):
            to_v[i] = from_v[i]

    @ti.kernel
    def sub_scale_vec(self, to_sub: ti.template(), scale: float, vec: ti.template(), store: ti.template()) -> float:
        for i in range(self.N):
            store[i] = to_sub[i] - scale * vec[i]

    @ti.kernel
    def update_x(self, a: float, w: float):
        for i in range(self.N):
            self.x[i] += a * self.p[i] + w * self.s[i]

    def solve(self):
        self.prepare()
        rhs_sqnorm = self.sqnorm(self.rhs)

        rho = 1.0
        a = 1.0
        w = 1.0

        self.calc_res()
        self.copy_over(self.r, self.r0)
        r0_sqnorm = self.sqnorm(self.r0)
        residual = r0_sqnorm / (rhs_sqnorm + 0.00001)
        
        num_iter = 0
        while(num_iter < self.max_iter and residual >= self.tol * self.tol):
            rho_old = rho
            rho = self.dot(self.r0, self.r)
            if(abs(rho) < self.EPS2 * r0_sqnorm):
                self.calc_res()
                self.copy_over(self.r, self.r0)
                r0_sqnorm = self.sqnorm(self.r0)
                rho = r0_sqnorm
            if(w != 0):
                b = (rho / rho_old) * (a / w)
            else:
                b = 0.0
            self.sub_scale_vec(self.p, w, self.v, self.temp)
            self.sub_scale_vec(self.r, -b, self.temp, self.p)
            self.lhs_func(self.p, self.v)
            a = rho / self.dot(self.r0, self.v)
            self.sub_scale_vec(self.r, a, self.v, self.s)
            self.lhs_func(self.s, self.t)

            t_sqnorm = self.sqnorm(self.t)
            if(t_sqnorm > 0):
                w = self.dot(self.t, self.s) / t_sqnorm
            else:
                w = 0.0
            self.update_x(a, w)
            self.sub_scale_vec(self.s, w, self.t, self.r)
            residual = self.sqnorm(self.r) / (rhs_sqnorm + 0.00001)
            num_iter += 1
            if(num_iter % 1000 == 0):
                print(num_iter, residual ** (1. / 2.))
        print("    ran iter", num_iter, "final residual%", residual ** (1. / 2.))


@ti.data_oriented
class ImplicitShearSolver:

    I_3 = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def __init__(self, particle: "Particle", to_store):
        self.particle = particle
        self.N_snow = particle.N_snow
        self.N_bound = particle.N_bound
        self.N_tot = particle.N_snow + particle.N_bound
        self.v_star = ti.Vector.field(3, float, self.N_tot)
        self.temp_stress_tensor = ti.Matrix.field(3, 3, float, self.N_snow)

        # result
        self.rhs = ti.Vector.field(3, float, self.N_snow)
        self.x = ti.static(to_store)

    def iss(self):
        print("iss begin")
        self.rhs_func()
        self.BiCGSTAB()

    @ti.kernel
    def rhs_func(self):
        for p_i in range(self.N_tot):
            if(self.particle.material_type[p_i] == Particle.Materials.FLUID):
                self.v_star[p_i] = self.particle.v[p_i] + self.particle.dt * (self.particle.a_friction[p_i] + self.particle.a_other[p_i] + self.particle.a_lambda[p_i])
            elif(self.particle.material_type[p_i] == Particle.Materials.BOUNDARY):
                self.v_star[p_i] = self.particle.v[p_i]
        for p_i in range(self.N_snow):
            v_d = self.disc_grad(p_i, self.v_star)
            F_star = self.particle.F[p_i] + self.particle.dt * v_d @ self.particle.F[p_i]
            F_star_T = F_star.transpose()
            self.temp_stress_tensor[p_i] = 2.0 * self.particle.lame_G[p_i] * (0.5 * (F_star + F_star_T) - self.I_3) 
        for p_i in range(self.N_snow):
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
        mirrored_cauchy = 1.0 / 3.0 * self.temp_stress_tensor[p_i].trace() * self.I_3
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
            n_kernel_d = self.particle.kernel_cubic_d(self.particle.x[p_i] - self.particle.x[n_i], self.particle.cutoff_radius)
            if(self.particle.material_type[n_i] == Particle.Materials.FLUID):
                n_diff = to_grad[n_i] - to_grad[p_i]
                d_v_j += n_diff.outer_product(n_volume * n_kernel_d)
            elif(self.particle.material_type[n_i] == Particle.Materials.BOUNDARY):
                n_diff = ti.Vector([0.0, 0.0, 0.0])
                if(to_grad.shape[0] <= p_i):
                    n_diff = -to_grad[p_i]
                else:
                    n_diff = to_grad[n_i] - to_grad[p_i]
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
    def lhs_func(self, inp_v: ti.template(), out_v: ti.template()):
        for p_i in range(self.N_snow):
            b_d = self.disc_grad(p_i, inp_v)
            F_res = b_d @ self.particle.F[p_i]
            self.temp_stress_tensor[p_i] = self.particle.lame_G[p_i] * (F_res + F_res.transpose())
        for p_i in range(self.N_snow):
            pre_sub = self.disc_div_stress(p_i) * self.particle.dt / self.particle.rho[p_i]
            out_v[p_i] = inp_v[p_i] - pre_sub

    def BiCGSTAB(self):
        max_iter = 50000
        tol = 1e-3
        solver = BiCGSTAB(self.N_snow, self.lhs_func, self.rhs, self.x, max_iter, tol, with_guess=True)
        solver.solve()


@ti.data_oriented
class Particle:

    class Materials:
        FLUID = 0
        BOUNDARY = 1

    def __init__(self, N_snow, N_bound, v_range, mass=0.02, cutoff_radius=0.075, substep=1, dt=1e-3, boundary_call_back=None, young=140000, friction=1.0, enable_shear=True):
        self.N_snow = N_snow
        self.N_bound = N_bound
        self.N_tot = N_snow + N_bound
        self.range = np.array(v_range, dtype=np.float32)
        self.range_start = ti.Vector([self.range[0, 0], self.range[1, 0], self.range[2, 0]])
        self.range_end = ti.Vector([self.range[0, 1], self.range[1, 1], self.range[2, 1]])
        self.substep_num = substep
        self.dt = dt
        self.boundary_call_back = boundary_call_back
        self.total_t = ti.field(float, ())
        self.enable_shear = enable_shear

        # material properties
        self.young = young
        self.poisson = 0.2
        self.harden = 10
        self.rho_0 = 400
        self.psi = 1.5 # motivated by Akinci et al.
        self.clamp_l = 1 - 0.025
        self.clamp_h = 1 + 0.0075
        self.friction = friction
        self.volume_scaling = 8.0

        # all particle properties
        self.m = mass
        self.x = ti.Vector.field(3, float, self.N_tot)
        self.material_type = ti.field(ti.i32, self.N_tot)
        self.v = ti.Vector.field(3, float, self.N_tot)

        # snow particle properties
        self.p = ti.field(float, self.N_snow)
        self.d_p = ti.Vector.field(3, float, self.N_snow)
        self.rho = ti.field(float, self.N_snow)
        self.rho_rest = ti.field(float, self.N_snow)
        self.F = ti.Matrix.field(3, 3, float, self.N_snow) # deformation gradient
        self.L_i = ti.Matrix.field(3, 3, float, self.N_snow)
        self.lame_lambda = ti.field(float, self.N_snow)
        self.lame_G = ti.field(float, self.N_snow)

        # accelerations
        self.a_other = ti.Vector.field(3, float, self.N_snow)
        self.a_friction = ti.Vector.field(3, float, self.N_snow)
        self.a_lambda = ti.Vector.field(3, float, self.N_snow)
        self.a_G = ti.Vector.field(3, float, self.N_snow)

        # solvers
        self.icps_solver = ImplicitCompressiblePressureSolver(self, self.p)
        self.iss_solver = ImplicitShearSolver(self, self.a_G)

        # kernels
        self.cutoff_radius = cutoff_radius
        self.cutoff_radius_sqr = self.cutoff_radius * self.cutoff_radius

        # neighbor search functionality
        self.max_neighbors = 100
        self.neighbors = ti.field(ti.i32, (self.N_tot, self.max_neighbors))
        self.neighbors_num = ti.field(ti.i32, self.N_tot)
        self.max_particles_per_cell = 100 # should be a upper bound somewhere but we will leave it this way
        self.grid_side = 2 * self.cutoff_radius
        self.grid_size = (np.ceil((self.range[:, 1] - self.range[:, 0]) / self.grid_side)).astype(int)
        self.grid = ti.field(ti.i32, (*self.grid_size, self.max_particles_per_cell))
        self.grid_num = ti.field(ti.i32, (*self.grid_size, ))
        self.grid_search_offset_range = (-1, 1 + 1)

    @ti.kernel
    def init(self):
        for p_i in range(self.N_snow):
            self.a_G = ti.Vector([0.0, 0.0, 0.0])

    @ti.func
    def compute_grid_index(self, pos):
        return ((pos - self.range_start) / self.grid_side).cast(int)

    @ti.func
    def is_in_grid(self, grid_cell):
        res = True
        for i in ti.static(range(3)):
            res &= 0 <= grid_cell[i] < self.grid_size[i]
        return res

    @ti.kernel
    def neighbor_search(self):
        for p_i in range(self.N_tot):
            pos_i = self.x[p_i]
            num_neighbors = 0
            g_i = self.compute_grid_index(pos_i)
            for offs in ti.static(ti.grouped(ti.ndrange(*(self.grid_search_offset_range, ) * 3))):
                g_j = g_i + offs
                if self.is_in_grid(g_j):
                    for g_j_i in range(self.grid_num[g_j]):
                        p_j = self.grid[g_j, g_j_i]
                        if num_neighbors < self.max_neighbors and (pos_i - self.x[p_j]).norm_sqr() < 4.0 * self.cutoff_radius_sqr:
                            self.neighbors[p_i, num_neighbors] = p_j
                            num_neighbors += 1
            self.neighbors_num[p_i] = num_neighbors

    @ti.kernel
    def allocate_grid(self):
        for p_i in range(self.N_tot):
            pos_i = self.x[p_i]
            g_i = self.compute_grid_index(pos_i)
            if self.is_in_grid(g_i):
                # atomical write and return
                num_grid_i = ti.atomic_add(self.grid_num[g_i], 1)
                self.grid[g_i, num_grid_i] = p_i

    @ti.func
    def kernel_cubic(self, v, h):
        # https://pysph.readthedocs.io/en/latest/reference/kernels.html
        # http://web.cse.ohio-state.edu/~wang.3602/courses/cse3541-2017-fall/08-SPH.pdf
        smoothing = 1. / (np.pi * h ** 3)
        q = v.norm() / h
        res = 0.0
        if q <= 1.0:
            res = smoothing * (1 - 1.5 * q ** 2 + 0.75 * q ** 3)
        elif q <= 2.0:
            res = smoothing * 0.25 * (2.0 - q) ** 3
        return res

    @ti.func
    def kernel_cubic_d(self, v, h):
        smoothing = 1. / (np.pi * h ** 3)
        v_norm = v.norm()
        q = v_norm / h
        res = ti.cast(0.0, float)
        if q <= 1.0:
            res = (smoothing / h) * (-3.0 * q + 2.25 * q ** 2.0)
        elif q <= 2.0:
            res = -0.75 * (smoothing / h) * (2.0 - q) ** 2
        else:
            res = 0.0
        ret = ti.Vector([0.0, 0.0, 0.0])
        if v_norm != 0:
            ret = res * v / (v_norm * h)
        return ret

    @ti.func
    def ret_volume(self, p_i):
        ret_v = 0.0
        if(self.material_type[p_i] == self.Materials.FLUID):
            ret_v = self.m / self.rho[p_i]
        elif(self.material_type[p_i] == self.Materials.BOUNDARY):
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
            elif(self.material_type[n_i] == self.Materials.BOUNDARY):
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
        for neighbor_i in range(self.neighbors_num[p_i]):
            n_i = self.neighbors[p_i, neighbor_i]
            if(self.material_type[n_i] == self.Materials.FLUID):
                n_kernel_d = self.kernel_cubic(self.x[p_i] - self.x[n_i], self.cutoff_radius)
                cur_rho += self.m * n_kernel_d
        self.rho[p_i] = cur_rho
        self.rho_rest[p_i] = cur_rho * ti.abs(self.F[p_i].determinant())

    @ti.func
    def calc_F(self, p_i):
        # https://www.math.ucla.edu/~jteran/papers/SSCTS13.pdf
        d_v = self.iss_solver.disc_grad(p_i, self.v)
        prim_F = self.F[p_i] + self.dt * d_v @ self.F[p_i]
        U, sigma, V = ti.svd(prim_F, ti.f32)
        sigma[0, 0] = ti.max(self.clamp_l, ti.min(sigma[0, 0], self.clamp_h))
        sigma[1, 1] = ti.max(self.clamp_l, ti.min(sigma[1, 1], self.clamp_h))
        sigma[2, 2] = ti.max(self.clamp_l, ti.min(sigma[2, 2], self.clamp_h))
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
            if(self.material_type[n_i] == self.Materials.BOUNDARY):
                n_volume = self.ret_volume(n_i)
                n_diff = self.x[p_i] - self.x[n_i]
                n_kernel_d = self.kernel_cubic_d(self.x[p_i] - self.x[n_i], self.cutoff_radius)
                common = n_volume * n_diff.dot(n_kernel_d) / (n_diff.norm_sqr() + 0.01 * self.grid_side ** 2)
                d_ii_b_sum += common
                a_b_sum += common * self.v[n_i]
        d_ii = 1 - self.dt * self.friction * d_ii_b_sum
        new_v = 1 / d_ii * (self.v[p_i] + self.dt * self.a_other[p_i] - self.dt * self.friction * a_b_sum)
        self.a_friction[p_i] = (new_v - (self.v[p_i] + self.dt * self.a_other[p_i])) / self.dt

    @ti.kernel
    def prologue(self):
        for i, j, k in ti.ndrange(*self.grid_size):
            self.grid_num[i, j, k] = 0

    @ti.kernel
    def prereq_update(self):
        for p_i in range(self.N_snow):
            self.calc_rhos(p_i)
            self.calc_lame(p_i)
            self.calc_a_other(p_i)
        for p_i in range(self.N_snow):
            self.calc_L_i(p_i)
            self.calc_a_friction(p_i)

    def print_init(self):
        for p_i in range(self.N_snow):
            print(p_i)
            print(self.rho[p_i], self.rho_rest[p_i], self.lame_G[p_i], self.lame_lambda[p_i])
            print(self.L_i[p_i])
            print(self.a_other[p_i], self.a_friction[p_i])

    @ti.kernel
    def a_lambda_update(self):
        for p_i in range(self.N_snow):
            self.calc_d_p(p_i)
            self.a_lambda[p_i] = - 1.0 / self.rho[p_i] * self.d_p[p_i]

    @ti.kernel
    def v_x_update(self):
        for p_i in range(self.N_snow):
            self.v[p_i] += self.dt * (self.a_other[p_i] + self.a_friction[p_i] + self.a_lambda[p_i] + self.a_G[p_i])
        for p_i in range(self.N_snow):
            self.calc_F(p_i)
        for p_i in range(self.N_snow):
            self.x[p_i] += self.dt * self.v[p_i]

    @ti.kernel
    def print_neighbor_info(self):
        min_neighbor = self.max_neighbors
        max_neighbor = -1
        total_neighbor = 0
        for p_i in range(self.N_tot):
            ti.atomic_max(max_neighbor, self.neighbors_num[p_i])
            ti.atomic_min(min_neighbor, self.neighbors_num[p_i])
            total_neighbor += self.neighbors_num[p_i]
        print("max neighbors", max_neighbor)
        print("min neighbors", min_neighbor)
        print("avg neighbors", total_neighbor / self.N_tot)

        min_grid = self.max_particles_per_cell
        max_grid = -1
        total_grid = 0
        for i, j, k in ti.ndrange(*self.grid_size):
            ti.atomic_max(max_grid, self.grid_num[i, j, k])
            ti.atomic_min(min_grid, self.grid_num[i, j, k])
            total_grid += self.grid_num[i, j, k]
        print("max grid", max_grid)
        print("min grid", min_grid)
        print("avg grid", total_grid / (self.grid_size[0] * self.grid_size[1] * self.grid_size[2]))

    def substep(self):
        self.total_t[None] += self.dt
        self.prologue()
        self.allocate_grid()
        self.neighbor_search()
        self.print_neighbor_info()
        ti.sync()

        self.prereq_update()
        # self.print_init()

        # Alg 1, line 7
        self.icps_solver.icps()
        self.a_lambda_update()

        # Alg 1, line 8
        if(self.enable_shear):
            self.iss_solver.iss()
        self.v_x_update()
        if(self.boundary_call_back != None):
            self.boundary_call_back()

    def step(self):
        for _ in range(self.substep_num):
            self.substep()

    def export_frame(self, folder, file):
        to_store = os.path.join(folder, file)
        with open(to_store, 'wb') as f:
            np.savez(f, x=self.x.to_numpy(dtype=np.float64), 
                        material_type=self.material_type.to_numpy(dtype=np.float64), 
                        v=self.v.to_numpy(dtype=np.float64),
                        p=self.p.to_numpy(dtype=np.float64),
                        neighbors_num=self.neighbors_num.to_numpy(dtype=np.float64),
                        rho=self.rho.to_numpy(dtype=np.float64),
                        rho_rest=self.rho_rest.to_numpy(dtype=np.float64))


@ti.data_oriented
class Compression:
    def __init__(self, cube_size):
        self.init_i = ti.field(int, ())
        self.cube_size = cube_size
        self.init()

    def get_particle(self):
        return self.particle

    @ti.kernel
    def gen_bound(self, side_len: ti.int32):
        for p_i in range(side_len * side_len * side_len):
            i = (int)(p_i // (side_len * side_len))
            j = (int)((p_i % (side_len * side_len)) // side_len)
            k = (int)(p_i % side_len)
            if (i == 0 or i == side_len - 1 or j == 0 or j == side_len - 1 or k == 0):
                old = ti.atomic_add(self.init_i[None], 1) 
                self.particle.material_type[old] = self.particle.Materials.BOUNDARY
                self.particle.x[old] = ti.Vector([i / side_len,
                                                  j / side_len, 
                                                  k / side_len])
                self.particle.v[old] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def gen_lid(self, side_len: ti.int32):
        for p_i in range(side_len * side_len * side_len):
            i = (int)(p_i // (side_len * side_len))
            j = (int)((p_i % (side_len * side_len)) // side_len)
            k = (int)(p_i % side_len)
            if (k == side_len - 1 and not (i == 0 or i == side_len - 1 or j == 0 or j == side_len - 1 or k == 0)):
                old = ti.atomic_add(self.init_i[None], 1) 
                self.particle.material_type[old] = self.particle.Materials.BOUNDARY
                self.particle.x[old] = ti.Vector([i / side_len,
                                                  j / side_len, 
                                                  0.2 + k / side_len])
                self.particle.v[old] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def gen_snow(self, side_len: ti.int32):
        for p_i in range(side_len * side_len * side_len):
            i = (int)(p_i // (side_len * side_len))
            j = (int)((p_i % (side_len * side_len)) // side_len)
            k = (int)(p_i % side_len)
            if not(i == 0 or i == side_len - 1 or j == 0 or j == side_len - 1 or k == 0 or k >= side_len - 6):
                old = ti.atomic_add(self.init_i[None], 1) 
                self.particle.material_type[old] = self.particle.Materials.FLUID
                self.particle.x[old] = ti.Vector([i / side_len, 
                                                  j / side_len, 
                                                  k / side_len])
                self.particle.p[old] = 0.0
                self.particle.a_G[old] = ti.Vector([0.0, 0.0, 0.0])
                self.particle.v[old] = ti.Vector([0.0, 0.0, 0.0])
                self.particle.F[old] = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    @ti.kernel
    def boundary_call_back(self):
        for p_i in range(self.lid_start, self.lid_end):
            v = 5
            a_other = ti.Vector([0.0, 0.0, v / 200.0])
            if (self.particle.total_t[None] // self.particle.dt == 500):
                self.particle.v[p_i] = ti.Vector([0.0, 0.0, -v])

            if (self.particle.total_t[None] // self.particle.dt >= 1000):
                self.particle.v[p_i] = ti.Vector([0.0, 0.0, 0.0])
            elif (self.particle.total_t[None] // self.particle.dt >= 800):
                self.particle.v[p_i] += a_other
            elif(self.particle.total_t[None] // self.particle.dt >= 700):
                self.particle.v[p_i] = ti.Vector([0.0, 0.0, 0.0])
            elif (self.particle.total_t[None] // self.particle.dt >= 500):
                self.particle.v[p_i] += a_other
            self.particle.x[p_i] += self.particle.dt *self.particle.v[p_i]


    def init(self):
        snow = (self.cube_size - 2) * (self.cube_size - 2) * (self.cube_size - 7)
        bounds_density = (int)(self.cube_size * 1.5)
        bounds = bounds_density * bounds_density * bounds_density - (bounds_density - 2) * (bounds_density - 2) * (bounds_density - 2)
        v_range = ((-1, 2), (-1, 2), (-1, 2))
        self.particle = Particle(snow, bounds, v_range, 0.8 / self.cube_size ** 3 * 400, cutoff_radius=1.0 / self.cube_size, boundary_call_back=self.boundary_call_back, enable_shear=False)
        self.init_i[None] = 0
        self.gen_snow(self.cube_size)
        self.gen_bound(bounds_density)
        self.lid_start = self.init_i[None]
        self.gen_lid(bounds_density)
        self.lid_end = self.init_i[None]


@ti.data_oriented
class Slide:
    def __init__(self, cube_size, scale=1.0):
        self.init_i = ti.field(int, ())
        self.cube_size = cube_size
        self.scale = scale
        self.init()

    def get_particle(self):
        return self.particle

    @ti.kernel
    def gen_bound(self, side_len: ti.int32):
        for p_i in range(side_len * side_len):
            j = (int)(p_i // side_len)
            k = (int)(p_i % side_len)
            old = ti.atomic_add(self.init_i[None], 1) 
            self.particle.material_type[old] = self.particle.Materials.BOUNDARY
            self.particle.x[old] = ti.Vector([0.0,
                                              j / side_len, 
                                              k / side_len])
            self.particle.v[old] = ti.Vector([0, 0, 0])

    @ti.kernel
    def gen_snow(self, side_len: ti.int32):
        for p_i in range(side_len * side_len * side_len):
            i = (int)(p_i // (side_len * side_len))
            j = (int)((p_i % (side_len * side_len)) // side_len)
            k = (int)(p_i % side_len)
            if not(i == 0 or i == side_len - 1 or j == 0 or j == side_len - 1 or k == 0):
                old = ti.atomic_add(self.init_i[None], 1) 
                self.particle.material_type[old] = self.particle.Materials.FLUID
                self.particle.x[old] = ti.Vector([i / side_len, 
                                                  j / side_len, 
                                                  k / side_len])
                self.particle.p[old] = 0
                self.particle.v[old] = ti.Vector([0, 0, 0])
                self.particle.F[old] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    @ti.kernel
    def rescale(self):
        for p_i in range(self.particle.N_tot):
            self.particle.x[p_i] /= self.scale

    def init(self):
        snow = (self.cube_size - 2) * (self.cube_size - 2) * (self.cube_size - 1)
        bounds_density = self.cube_size 
        bounds = bounds_density * bounds_density
        v_range = ((-1 / self.scale, 2 / self.scale), (-1 / self.scale, 2 / self.scale), (-10 / self.scale, 2 / self.scale))
        self.particle = Particle(snow, bounds, v_range, 1.0 / (self.cube_size * self.scale) ** 3 * 400, cutoff_radius=1.0 / self.cube_size / self.scale, enable_shear=False)
        self.init_i[None] = 0
        self.gen_snow(self.cube_size)
        self.gen_bound(bounds_density)
        self.rescale()


def scene_1():
    return 2000, Compression(30).get_particle()

def scene_2():
    return 600, Slide(30).get_particle()


to_save_folder = f"simul_{int(time.time())}"
to_save_frame = "frame_{}.npy"
os.makedirs(to_save_folder)

print("Starting")
# REPLACE HERE
timesteps, particle = scene_1()
cur_iter_num = 0
particle.export_frame(to_save_folder, to_save_frame.format(cur_iter_num))
while True:
    print(cur_iter_num)
    cur_iter_num += 1
    particle.step()
    particle.export_frame(to_save_folder, to_save_frame.format(cur_iter_num))
    if(cur_iter_num == timesteps):
        break
