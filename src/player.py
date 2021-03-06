import numpy as np
import open3d as o3d
import os
import sys
import time


class NP_Particle:

    to_save_frame = "frame_{}.npy"

    class CurrentRender:
        MAT = "material_type"
        P = "p"
        NEIGHBORS_NUM = "neighbors_num"
        A_LAMBDA = "a_lambda"
        RHO = "rho"

    def __init__(self, folder):
        self.folder = folder
        self.arrays = {}

    def init(self):
        self.load_frame(0)

    def load_frame(self, i):
        to_load = os.path.join(self.folder, self.to_save_frame.format(i))
        with open(to_load, 'rb') as f:
            npzfile = np.load(f)
            for name in npzfile.files:
                self.arrays[name] = npzfile[name]

    def get_pos(self):
        return self.arrays["x"]

    def color_bound(self, arr):
        mat_arr = self.arrays[self.CurrentRender.MAT]
        if(arr.shape[0] < mat_arr.shape[0]):
            arr = np.pad(arr, ((0,mat_arr.shape[0] - arr.shape[0]),(0,0)))
        arr[mat_arr == 1] = np.array([0, 1, 0.706])
        return arr

    def get_attribute(self, attr):
        if(attr == self.CurrentRender.MAT):
            raw_arr = self.arrays[attr]
            res = np.empty((raw_arr.shape[0], 3))
            res[raw_arr == 0] = np.array([1, 0.706, 0])
            res[raw_arr == 1] = np.array([0, 1, 0.706])
            return res
        elif(attr == self.CurrentRender.P):
            raw_arr = self.arrays[attr]
            p_mag = (raw_arr - np.min(raw_arr)) / (np.max(raw_arr) - np.min(raw_arr))
            res = np.stack([p_mag, p_mag, np.ones_like(p_mag)], axis=1)
            res = self.color_bound(res)
            return res
        elif(attr == self.CurrentRender.NEIGHBORS_NUM):
            raw_arr = self.arrays[attr]
            p_mag = (raw_arr - np.min(raw_arr)) / (np.max(raw_arr) - np.min(raw_arr))
            res = np.stack([p_mag, p_mag, np.ones_like(p_mag)], axis=1)
            return res
        elif(attr == self.CurrentRender.A_LAMBDA):
            raw_arr = self.arrays[attr]
            raw_arr_norm = np.linalg.norm(raw_arr, axis=1)
            res = np.ones_like(raw_arr) - raw_arr * np.expand_dims(raw_arr_norm / (np.max(raw_arr_norm) + 0.00001), axis=1)
            res = self.color_bound(res)
            return res
        elif(attr == self.CurrentRender.RHO):
            raw_arr = self.arrays[attr]
            p_mag = np.ones_like(raw_arr) - (raw_arr - np.full_like(raw_arr, 100.0)) / (np.full_like(raw_arr, 1000.0) - np.full_like(raw_arr, 100.0))
            res = np.stack([p_mag, p_mag, np.ones_like(p_mag), ], axis=1)
            res = self.color_bound(res)
            return res

class Render:

    def __init__(self, obj):
        # https://github.com/isl-org/Open3D/issues/572
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.obj = obj
        self.cur_view = NP_Particle.CurrentRender.MAT

    def init(self):
        # converted to flaot64 for speed
        self.point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.obj.get_pos()))
        self.point_cloud.colors = o3d.utility.Vector3dVector(self.obj.get_attribute(self.cur_view))
        self.vis.add_geometry(self.point_cloud)

    def update(self):
        self.point_cloud.points = o3d.utility.Vector3dVector(self.obj.get_pos())
        self.point_cloud.colors = o3d.utility.Vector3dVector(self.obj.get_attribute(self.cur_view))
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


folder_name = sys.argv[1]
particle = NP_Particle(folder_name)
vis = Render(particle)
num_frames = len(next(os.walk(folder_name))[2])
paused = True

def init():
    particle.init()
    vis.init()

def reset_sim(R_vis):
    global i
    i = 0
    particle.init()

def pause_sim(R_vis):
    global paused
    paused = not paused

def adv_frame(R_vis):
    global i
    i = (i + 1) % num_frames
    print("frame", i)


def prev_frame(R_vis):
    global i
    i = (i - 1) % num_frames
    print("frame", i)


vis.create_window()
vis.register_key_callback(ord("R"), reset_sim)
vis.register_key_callback(ord(" "), pause_sim)
vis.register_key_callback(ord("D"), adv_frame)
vis.register_key_callback(ord("A"), prev_frame)

ctr = vis.get_view_control()
ctr.set_lookat([0.0, 0.5, 0.0])
ctr.set_up([0.0, 1.0, 0.0])
init()

i = 0
while True:
    if(not paused):
        i = (i + 1) % num_frames
        print("frame", i)
    particle.load_frame(i)
    vis.update()
    if not vis.poll_events():
        break
    vis.update_renderer()
