
import time
import point_cloud_utils as pcu

import click
import numpy as np
import viser
import viser.transforms as tf
from termcolor import colored

def single_semantic_voxel_to_mesh(voxel_ijk, voxel_size = 0.1, voxel_origin = [0, 0, 0]):
    cube_v, cube_f = pcu.voxel_grid_geometry(voxel_ijk, gap_fraction=0.02, voxel_size=voxel_size, voxel_origin=voxel_origin)
    return cube_v, cube_f

def read_fvdb_grid(grid):

    vox_ijk = grid.ijk.jdata.cpu().numpy()

    return vox_ijk

def set_kill_key_button(server, gui_kill_button):
    @gui_kill_button.on_click
    def _(_) -> None:
        print(f"{colored('Killing the sample.', 'red', attrs=['bold'])}")
        setattr(server, "alive", False)
        time.sleep(0.3)

def render_point_cloud(server, points, colors=None, point_size=0.1, name="/simple_pc", port=8080):
    """
        points: [N, 3]
        colors: [3,] or [N, 3]
    """
    setattr(server, "alive", True)

    if colors is None:
        colors = (90, 200, 255)
    
    server.add_point_cloud(
        name=name,
        points=points,
        colors=colors,
        point_size=point_size,
    )

    while True:
        clients = server.get_clients()
        for id, client in clients.items():
            camera = client.camera
            # ic(id, camera.position, camera.wxyz)
        time.sleep(1/60)

        if not server.alive:
            server.scene.reset()
            break

def render_multiple_polygon_mesh(server,
                                 vertices_list,
                                 face_list,
                                 color_list=None,
                                 name="/simple_mesh",
                                 port=8080):
    """
    Render multiple polygon meshes without texture

    Args:
        vertices_list (List[ndarray]) 
            A list of numpy array of vertex positions. Each array should have shape (V, 3).
        faces_list (List[ndarray]) 
            A list of numpy array of face indices. Each array should have shape (F, 3).
        color_list (List[Tuple[int, int, int] | Tuple[float, float, float] | ndarray]) 
            A list of color of the mesh as an RGB tuple.
    """
    setattr(server, "alive", True)

    for i, vertices in enumerate(vertices_list):
        server.add_mesh_simple(
            name=name + f"_{i}",
            vertices=vertices,
            faces=face_list[i],
            color=None,
            wxyz=tf.SO3.from_x_radians(0.0).wxyz,
            position=(0, 0, 0),
        )

    while True:
        clients = server.get_clients()
        for id, client in clients.items():
            camera = client.camera
            # ic(id, camera.position, camera.wxyz)
        time.sleep(1/60)

        if not server.alive:
            server.scene.reset()
            break

    time.sleep(1)

@click.command()
@click.option('--paths', '-p', multiple=True, help='directories of .pt files')
@click.option('--port', '-o', default=8080, help='port number')
@click.option("--type", '-t', default="voxel", help="voxel or pc. voxel can not be used for 1024 resolution grid.")
@click.option("--size", '-s', default=0.1, help="point size for point cloud")
def visualize_grid_from_grid(grid, port, type, size):
    server = viser.ViserServer(port=port)
    gui_kill_button = server.add_gui_button(
        "Kill", hint="Press this button to kill this sample"
    )
    set_kill_key_button(server, gui_kill_button)

    vox_ijk = read_fvdb_grid(grid)
    vox_ijk_center = np.round(np.mean(vox_ijk, axis=0))
    vox_ijk = vox_ijk - vox_ijk_center
    vox_ijk = vox_ijk.astype(np.int32)

    if type == "voxel":
        cube_v_list = []
        cube_f_list = []
        geometry_list = []

        cube_v_i, cube_f_i = single_semantic_voxel_to_mesh(vox_ijk)

        cube_v_list.append(cube_v_i)
        cube_f_list.append(cube_f_i)

        from pycg import image, render, vis

        geometry = vis.mesh(cube_v_i, cube_f_i)
        geometry_list.append(geometry)

        save_render = False
        if save_render:
            scene: render.Scene = vis.show_3d(geometry_list, show=False, up_axis='+Z', default_camera_kwargs={"pitch_angle": 80.0, "fill_percent": 0.8, "fov": 80.0, 'plane_angle': 270})
            img = scene.render_filament()
            img = scene.render_pyrender()
            image.write(img, 'tmp.png')

        render_multiple_polygon_mesh(server, cube_v_list, cube_f_list)

    elif type == "pc":
        vox_ijk = vox_ijk * 0.1
        render_point_cloud(server, vox_ijk, visualization_color, point_size=size, port=port)
