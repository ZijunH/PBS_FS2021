import numpy as np


class Box(object):
    def __init__(self, vmin, vmax):
        self.vmin = vmin
        self.vmax = vmax
        assert (
            sum([self.vmin[i] < self.vmax[i] for i in range(3)]) == 3
        ), f"invalid vmin, vmax: {vmin}, {vmax}"

    def vertices(self):
        return np.array(
            [
                self.vmin,
                [self.vmax[0], self.vmin[1], self.vmin[2]],
                [self.vmin[0], self.vmax[1], self.vmin[2]],
                [self.vmax[0], self.vmax[1], self.vmin[2]],
                [self.vmin[0], self.vmin[1], self.vmax[2]],
                [self.vmax[0], self.vmin[1], self.vmax[2]],
                [self.vmin[0], self.vmax[1], self.vmax[2]],
                self.vmax,
            ], dtype=np.float32
        )

    def indices(self):
        return np.array(
            [
                0,
                2,
                1,
                2,
                3,
                1,
                1,
                3,
                5,
                3,
                7,
                5,
                2,
                6,
                3,
                6,
                7,
                3,
                5,
                7,
                4,
                7,
                6,
                4,
                4,
                6,
                0,
                6,
                2,
                0,
                0,
                4,
                1,
                4,
                5,
                1,
            ], dtype=np.int32
        )


class Plane(object):
    def __init__(self, center=[0, 0, 0], size=10):
        self.center = center
        self.size = size

    def vertices(self):
        v0 = [
            self.center[0] - self.size * 0.5,
            self.center[1],
            self.center[2] - self.size * 0.5,
        ]
        v1 = [
            self.center[0] + self.size * 0.5,
            self.center[1],
            self.center[2] - self.size * 0.5,
        ]
        v2 = [
            self.center[0] + self.size * 0.5,
            self.center[1],
            self.center[2] + self.size * 0.5,
        ]
        v3 = [
            self.center[0] - self.size * 0.5,
            self.center[1],
            self.center[2] + self.size * 0.5,
        ]
        v = np.array([v0, v1, v2, v3], dtype=np.float32)
        return v

    def indices(self):
        return np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
