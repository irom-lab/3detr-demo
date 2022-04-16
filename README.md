# 3DETR Demo for 3D Object Localization

This repo provides a simple demo of [`3DETR: An End-to-End Transformer Model for 3D Object Detection`](https://github.com/facebookresearch/3detr). In particular, given a point cloud obtained from a RealSense or Azure Kinect camera, the code outputs bounding boxes found using 3DETR.

# Installation

Follow the installation directions [`here`](https://github.com/facebookresearch/3detr). 

If you are using the Microsoft Azure Kinect camera, install [`Azure-Kinect-Sensor-SDK`](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md) using the Debian packages (for Linux). In addition, install [`pyk4a`](https://github.com/etiennedub/pyk4a).   

If you are using the Intel RealSense camera, install [`librealsense`](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md). 

# Running the demo

## Scannet or SUNRGB-D

The ```demo_files``` folder contains point clouds (in ply format). In order to run 3DETR on a demo point cloud from Scannet or SUNRGB-D, run the following:
```
python demo.py --test_ckpt pretrained/sunrgbd_ep1080.pth --nqueries 128 --data_source [scannet or sunrgbd]

```

The output is saved in ```demo_files/output_bboxes.ply```. You can visualize the input pointcloud (demo_files/input_pc_scannet.ply or demo_files/input_pc_sunrgbd.ply) and output bounding boxes by importing these into [`MeshLab`](https://www.meshlab.net/). 

## Azure Kinect

If you are using the Azure Kinect camera, you can run the following to capture a point cloud and save it in ply format:
```
python create_kinect_point_cloud.py
```
This will save the point cloud in demo_files/input_pc_kinect.ply and demo_files/input_pc_kinect_rgb.obj.  

Then, run the following to generate bounding boxes using 3DETR:
```
python demo.py --test_ckpt pretrained/sunrgbd_ep1080.pth --nqueries 128 --data_source kinect

```

The output is saved in ```demo_files/output_bboxes.ply```. You can visualize the input pointcloud (demo_files/input_pc_kinect.ply) and output bounding boxes by importing these into [`MeshLab`](https://www.meshlab.net/). 


## RealSense

If you are using a RealSense camera, you can export a point cloud in ply format using realsense-viewer (save this in demo_files/input_pc_realsense.ply). Note that this exports a point cloud using a left-handed coordinate system (!). The code here accounts for this fact when the point cloud is read in and also saves a ply file using the correct right-handed system in demo_files/input_pc_realsense_transformed.ply. 

Run the following to generate bounding boxes using 3DETR:
```
python demo.py --test_ckpt pretrained/sunrgbd_ep1080.pth --nqueries 128 --data_source realsense
```

The output is saved in ```demo_files/output_bboxes.ply```. You can visualize the **transformed** input pointcloud (demo_files/input_pc_realsense_transformed.ply) and output bounding boxes by importing these into [`MeshLab`](https://www.meshlab.net/). 
