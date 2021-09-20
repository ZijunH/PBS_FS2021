import taichi as ti

from primitives import Box, Plane

###############################################################
## Simulation

ti.init(arch=ti.gpu)

box = Box([-0.5, 0.1, -0.5], [0.5, 1.1, 0.5])
bv, bf = box.vertices(), box.indices()
num_vertices, num_faces = bv.shape[0], bf.shape[0]
box_vertices = ti.Vector.field(3, float, num_vertices)
box_indices = ti.field(int, num_faces)
box_vertex_colors = ti.Vector.field(3, float, num_vertices)

box_vertices.from_numpy(bv)
box_indices.from_numpy(bf)

speed = 60

###############################################################
## GUI
window_res = (800, 800)
window = ti.ui.Window("PBS", window_res, vsync=True)
canvas = window.get_canvas()

camera = ti.ui.make_camera()
scene = ti.ui.Scene()

paused = False
frame_id = ti.field(int, ())

base = Plane()
base_v, base_f = base.vertices(), base.indices()
base_vertices = ti.Vector.field(3, float, base_v.shape[0])
base_indices = ti.field(int, base_f.shape[0])
base_vertices.from_numpy(base_v)
base_indices.from_numpy(base_f)


def init():
    global paused
    paused = True
    frame_id[None] = 0

    # reset camera
    camera.position(0, 1, 5)
    camera.lookat(0, 0, 0)
    camera.up(0, 1, 0)
    camera.fov(50)

    # reset canvas background color
    canvas.set_background_color((0.1, 0.25, 0.25))

    #########
    # reset scene
    box_vertex_colors.fill(0)
    for i in range(num_vertices):
        box_vertex_colors[i][0] = 1.0


# @ti.kernel
def step():
    """
    integrate a single step in parallel
    """
    dec_color = (frame_id[None] // speed) % 3
    inc_color = (dec_color + 1) % 3

    for i in range(num_vertices):
        box_vertex_colors[i][dec_color] = (
            box_vertex_colors[i][dec_color] * speed - 1
        ) / speed
        box_vertex_colors[i][inc_color] = (
            box_vertex_colors[i][inc_color] * speed + 1
        ) / speed


# @ti.kernel
def render():
    """
    update content for rendering
    """
    pass


def show_options():
    """
    show gui widgets
    """
    global paused

    window.GUI.begin("Options", 0.05, 0.05, 0.2, 0.4)
    if window.GUI.button("Reset"):
        init()
    if paused:
        if window.GUI.button("Continue"):
            paused = False
    else:
        if window.GUI.button("Pause"):
            paused = True
    window.GUI.end()


def update_canvas():
    """
    update canvas with new camera and scene
    """
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)

    # NOTE: keep light setting in the loop..
    scene.ambient_light((0.25, 0.25, 0.25))
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(1.0, 1.0, 1.0))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1.0, 1.0, 1.0))
    scene.set_camera(camera)

    #################
    # update scene
    scene.mesh(
        vertices=box_vertices, indices=box_indices, per_vertex_color=box_vertex_colors
    )
    #################

    scene.mesh(vertices=base_vertices, indices=base_indices, color=(0.5, 0.5, 0.5))
    canvas.scene(scene)


# start simulation
init()
while window.running:
    for e in window.get_events(ti.ui.PRESS):
        if e.key in ("q", ti.ui.ESCAPE):
            window.running = False
            print("exit")
        elif e.key in ("p", ti.ui.SPACE):
            paused = not paused
            print("paused", paused)
        elif e.key == "r":
            init()
        elif e.key == ti.ui.LEFT:
            print("press left key")
        elif e.key == ti.ui.RIGHT:
            print("press right key")
        elif e.key == ti.ui.DOWN:
            print("press down key")
        elif e.key == ti.ui.UP:
            print("press up key")

    if not paused:
        step()

    render()
    show_options()
    update_canvas()
    window.show()
    frame_id[None] += 1
