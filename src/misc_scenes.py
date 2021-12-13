
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
            if not(i == 0 or i == side_len - 1 or j == 0 or j == side_len - 1 or k == 0 or k == side_len - 1 ):
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
            v = 7.5
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
        snow = (self.cube_size - 2) * (self.cube_size - 2) * (self.cube_size - 2)
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
class CubeParticleIniter:
    # 30, psi = 8
    def __init__(self, cube_size):
        self.init_i = ti.field(int, ())
        self.cube_size = cube_size
        self.init()

    def get_particle(self):
        return self.particle

    @ti.kernel
    def gen_bound(self, side_len: ti.int32):
        rand_factor = 0.25 / side_len 
        for p_i in range(side_len * side_len * side_len):
            i = (int)(p_i // (side_len * side_len))
            j = (int)((p_i % (side_len * side_len)) // side_len)
            k = (int)(p_i % side_len)
            if (i == 0 or i == side_len - 1 or j == 0 or j == side_len - 1 or k == 0):
                old = ti.atomic_add(self.init_i[None], 1) 
                self.particle.material_type[old] = self.particle.Materials.BOUNDARY
            # self.particle.x[old] = ti.Vector([0.0 ,
            #                                   j / side_len + (ti.random(float) - 0.5) * rand_factor, 
            #                                   k / side_len + (ti.random(float) - 0.5) * rand_factor])
                self.particle.x[old] = ti.Vector([i / side_len,
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

    def init(self):
        snow = (self.cube_size - 2) * (self.cube_size - 2) * (self.cube_size - 1)
        bounds_density = (int)(self.cube_size * 1.5)
        bounds = bounds_density * bounds_density * bounds_density - (bounds_density - 2) * (bounds_density - 2) * (bounds_density - 1)
        v_range = ((-1, 2), (-1, 2), (-1, 2))
        self.particle = Particle(snow, bounds, v_range, 1 / self.cube_size ** 3 * 400, cutoff_radius=1.0 / self.cube_size)
        self.init_i[None] = 0
        self.gen_snow(self.cube_size)
        self.gen_bound(bounds_density)

@ti.data_oriented
class PlaneParticleIniter:

    def __init__(self, cube_size):
        self.init_i = ti.field(int, ())
        self.cube_size = cube_size
        self.init()

    def get_particle(self):
        return self.particle

    @ti.kernel
    def gen_bound(self, side_len: ti.int32):
        rand_factor = 0.25 / side_len 
        for p_i in range(side_len * side_len * side_len):
            i = (int)(p_i // (side_len * side_len))
            j = (int)((p_i % (side_len * side_len)) // side_len)
            k = (int)(p_i % side_len)
            if (i == 0 or i == side_len - 1 or j == 0 or j == side_len - 1 or k == 0):
                old = ti.atomic_add(self.init_i[None], 1) 
                self.particle.material_type[old] = self.particle.Materials.BOUNDARY
            # self.particle.x[old] = ti.Vector([0.0 ,
            #                                   j / side_len + (ti.random(float) - 0.5) * rand_factor, 
            #                                   k / side_len + (ti.random(float) - 0.5) * rand_factor])
                self.particle.x[old] = ti.Vector([i / side_len,
                                                    j / side_len, 
                                                    k / side_len])
                self.particle.v[old] = ti.Vector([0, 0, 0])

    @ti.kernel
    def gen_snow(self, side_len: ti.int32):
        for k in range(3):
            for p_i in range((side_len * side_len) // 4):
                i = (int)(p_i // (side_len // 2))
                j = (int)(p_i % (side_len // 2))
                old = ti.atomic_add(self.init_i[None], 1) 
                self.particle.material_type[old] = self.particle.Materials.FLUID
                self.particle.x[old] = ti.Vector([side_len // 4 * 1 / side_len + i / side_len, 
                                                side_len // 4 * 1 / side_len + j / side_len, 
                                                0.1 + k / side_len])
                self.particle.p[old] = 0
                self.particle.v[old] = ti.Vector([0, 0, 0])
                self.particle.F[old] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def init(self):
        snow = (self.cube_size) * (self.cube_size) // 4 * 3
        bounds_density = self.cube_size
        bounds = bounds_density * bounds_density * bounds_density - (bounds_density - 2) * (bounds_density - 2) * (bounds_density - 1)
        v_range = ((-1, 2), (-1, 2), (-1, 2))
        self.particle = Particle(snow, bounds, v_range, 7.0 / self.cube_size ** 3 * 400, cutoff_radius=1.0 / self.cube_size)
        self.init_i[None] = 0
        self.gen_snow(self.cube_size)
        self.gen_bound(bounds_density)

@ti.data_oriented
class FlatSurfaceParticleIniter:
    # 30, 30.0, psi = 0.8
    def __init__(self, cube_size, scale=1.0):
        self.init_i = ti.field(int, ())
        self.cube_size = cube_size
        self.scale = scale
        self.init()

    def get_particle(self):
        return self.particle

    @ti.kernel
    def gen_bound(self, side_len: ti.int32):
        rand_factor = 0.25 / side_len 
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
        self.particle = Particle(snow, bounds, v_range, 1.0 / (self.cube_size * self.scale) ** 3 * 400, cutoff_radius=1.0 / self.cube_size / self.scale)
        self.init_i[None] = 0
        self.gen_snow(self.cube_size)
        self.gen_bound(bounds_density)
        self.rescale()

@ti.data_oriented
class CubeParticleIniter2:
    # 30, psi = 8
    def __init__(self, cube_size, scale=100.0):
        self.init_i = ti.field(int, ())
        self.cube_size = cube_size
        self.scale = scale
        self.init()

    def get_particle(self):
        return self.particle

    @ti.kernel
    def gen_bound(self, side_len: ti.int32):
        for _ in range(1):
            rand_factor = 0.25 / side_len 
            for p_i in range(side_len * side_len * side_len):
                i = (int)(p_i // (side_len * side_len))
                j = (int)((p_i % (side_len * side_len)) // side_len)
                k = (int)(p_i % side_len)
                if (i == 0 or i == side_len - 1 or j == 0 or j == side_len - 1 or k == 0):
                    old = ti.atomic_add(self.init_i[None], 1) 
                    self.particle.material_type[old] = self.particle.Materials.BOUNDARY
                # self.particle.x[old] = ti.Vector([0.0 ,
                #                                   j / side_len + (ti.random(float) - 0.5) * rand_factor, 
                #                                   k / side_len + (ti.random(float) - 0.5) * rand_factor])
                    self.particle.x[old] = ti.Vector([i / side_len,
                                                        j / side_len, 
                                                        k / side_len])
                    self.particle.v[old] = ti.Vector([0, 0, 0])

    @ti.kernel
    def gen_snow(self, side_len: ti.int32):
        for _ in range(1):
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
        bounds_density = (int)(self.cube_size * 1.5)
        bounds = bounds_density * bounds_density * bounds_density - (bounds_density - 2) * (bounds_density - 2) * (bounds_density - 1)
        v_range = ((-1 / self.scale, 2 / self.scale), (-1 / self.scale, 2 / self.scale), (-1 / self.scale, 2 / self.scale))
        self.particle = Particle(snow, bounds, v_range, 1 / (self.cube_size * self.scale) ** 3 * 400, cutoff_radius=1.0 / (self.cube_size * self.scale))
        self.init_i[None] = 0
        self.gen_snow(self.cube_size)
        self.gen_bound(bounds_density)
        self.rescale()