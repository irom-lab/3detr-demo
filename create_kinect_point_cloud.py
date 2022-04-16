import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d  # noqa: F401

import pyk4a
from pyk4a import Config, PyK4A
from utils.pc_util_demo import write_ply_kinect, write_ply_kinect_rgb
import IPython as ipy

def main():
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            camera_fps=pyk4a.FPS.FPS_5,
            depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED,
            synchronized_images_only=True,
        )
    )
    k4a.start()

    # getters and setters directly get and set on device
    k4a.whitebalance = 4500
    assert k4a.whitebalance == 4500
    k4a.whitebalance = 4510
    assert k4a.whitebalance == 4510
    while True:
        capture = k4a.get_capture()
        if np.any(capture.depth) and np.any(capture.color):
            break
    while True:
        capture = k4a.get_capture()
        if np.any(capture.depth) and np.any(capture.color):
            break
    points = capture.depth_point_cloud.reshape((-1, 3))
    colors = capture.transformed_color[..., (2, 1, 0)].reshape((-1, 3))

    # Convert to m
    points = points/1000.0

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2], s=1, c=colors / 255,
    )


    # Save ply file
    write_ply_kinect(points, "demo_files/input_pc_kinect.ply")
    write_ply_kinect_rgb(points, colors, "demo_files/input_pc_kinect_rgb.obj")


    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 4)
    ax.view_init(elev=-90, azim=-90)
    plt.show()

    k4a.stop()


if __name__ == "__main__":
    main()
