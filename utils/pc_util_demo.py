import os
import sys
import torch

# Point cloud IO
import numpy as np
from plyfile import PlyData, PlyElement

# Mesh IO
import trimesh

import matplotlib.pyplot as pyplot
from pc_util import write_oriented_bbox

def read_ply_realsense(filename):
    """ read XYZ point cloud from filename PLY file from RealSense """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    # pc_array = np.array([[x, y, z, r, g, b] for x,y,z,r,g,b in pc])
    # pc_array = np.array([[x, y, z] for x,y,z in pc])

    # Change to correct coordinate system:
    # Real-Sense: +X (right), +Y (down),    +Z (forward) [But export to ply is different; correction below]
    # VoteNet:    +X (right),   +Y (forward), +Z (up) 
    pc_array = np.array([[x, -z, y, r, g, b] for x,y,z,r,g,b in pc])

    write_ply_rgb(pc_array[:,0:3], pc_array[:,3:6], "demo_files/input_pc_realsense_transformed.obj")

    pc_array = pc_array[:,0:3] 


    return pc_array

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

def write_ply_kinect(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    # Change to correct coordinate system:
    # Kinect: +X (right), +Y (down),    +Z (forward)
    # 3DETR:    +X (right),   +Y (forward), +Z (up) 
    points = [(points[i,0], points[i,2], -points[i,1]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)

def write_ply_kinect_rgb(points, colors, out_filename, num_classes=None):
    """ Color (N,3) points with RGB colors (N,3) within range [0,255] as OBJ file """
    colors = colors.astype(int)
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        c = colors[i,:]
        fout.write('v %f %f %f %d %d %d\n' % (points[i,0], points[i,2], -points[i,1],c[0],c[1],c[2]))
    fout.close()

def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)

def write_ply_color(points, labels, filename, num_classes=None, colormap=pyplot.cm.jet):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    labels = labels.astype(int)
    N = points.shape[0]
    if num_classes is None:
        num_classes = np.max(labels)+1
    else:
        assert(num_classes>np.max(labels))
    
    vertex = []
    #colors = [pyplot.cm.jet(i/float(num_classes)) for i in range(num_classes)]    
    colors = [colormap(i/float(num_classes)) for i in range(num_classes)]    
    for i in range(N):
        c = colors[labels[i]]
        c = [int(x*255) for x in c]
        vertex.append( (points[i,0],points[i,1],points[i,2],c[0],c[1],c[2]) )
    vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(filename)
   
def write_ply_rgb(points, colors, out_filename, num_classes=None):
    """ Color (N,3) points with RGB colors (N,3) within range [0,255] as OBJ file """
    colors = colors.astype(int)
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        c = colors[i,:]
        fout.write('v %f %f %f %d %d %d\n' % (points[i,0],points[i,1],points[i,2],c[0],c[1],c[2]))
    fout.close()

def point_cloud_to_bbox(points):
    """ Extract the axis aligned box from a pcl or batch of pcls
    Args:
        points: Nx3 points or BxNx3
        output is 6 dim: xyz pos of center and 3 lengths        
    """
    which_dim = len(points.shape) - 2 # first dim if a single cloud and second if batch
    mn, mx = points.min(which_dim), points.max(which_dim)
    lengths = mx - mn
    cntr = 0.5*(mn + mx)
    return np.concatenate([cntr, lengths], axis=which_dim)

def write_bbox_ply_from_outputs(outputs, out_filename, prob_threshold=0.5):
    '''
    Write ply file corresponding to bounding boxes using outputs of 3DETR model.
    Args:
        outputs: outputs from 3DETR model.
        out_filename: name of ply file to write.
        prob_threshold: probability above which we consider something an object.
    '''

    # Parse outputs
    centers = outputs["outputs"]["center_unnormalized"] 
    centers = centers.cpu().detach().numpy()
    lengths = outputs["outputs"]["size_unnormalized"]
    lengths = lengths.cpu().detach().numpy() 

    inds = outputs["outputs"]["objectness_prob"] > prob_threshold
    inds = inds.cpu()
    inds = inds[0,:]
    centers = centers[:,inds,:]
    lengths = lengths[:,inds,:]

    angles = outputs["outputs"]["angle_continuous"]
    angles = angles[:,inds]
    angles = angles.cpu().detach().numpy()

    scene_bbox = np.concatenate((centers, lengths), 2)
    scene_bbox = scene_bbox[0,:,:]

    scene_bbox = np.concatenate((scene_bbox, angles.T), 1)

    write_oriented_bbox(scene_bbox, out_filename)

    # Number of objects (detected above prob_threshold)
    num_objects = inds.sum().item()

    return num_objects