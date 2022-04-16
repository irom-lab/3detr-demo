""" Demo of using 3DETR object detector to detect objects from a point cloud.
"""

import os
import sys
import numpy as np
import argparse
import importlib
import time
import IPython as ipy
from models import build_model
from datasets import build_dataset

import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pretrained'))
from pc_util import random_sampling, write_bbox
from pc_util_demo import read_ply_realsense, read_ply


def make_args_parser():
    parser = argparse.ArgumentParser("3D Detection Using Transformers", add_help=False)

    ##### Model #####
    parser.add_argument(
        "--model_name",
        default="3detr",
        type=str,
        help="Name of the model",
        choices=["3detr"],
    )
    ### Encoder
    parser.add_argument(
        "--enc_type", default="vanilla", choices=["masked", "maskedv2", "vanilla"]
    )
    # Below options are only valid for vanilla encoder
    parser.add_argument("--enc_nlayers", default=3, type=int)
    parser.add_argument("--enc_dim", default=256, type=int)
    parser.add_argument("--enc_ffn_dim", default=128, type=int)
    parser.add_argument("--enc_dropout", default=0.1, type=float)
    parser.add_argument("--enc_nhead", default=4, type=int)
    parser.add_argument("--enc_pos_embed", default=None, type=str)
    parser.add_argument("--enc_activation", default="relu", type=str)

    ### Decoder
    parser.add_argument("--dec_nlayers", default=8, type=int)
    parser.add_argument("--dec_dim", default=256, type=int)
    parser.add_argument("--dec_ffn_dim", default=256, type=int)
    parser.add_argument("--dec_dropout", default=0.1, type=float)
    parser.add_argument("--dec_nhead", default=4, type=int)

    ### MLP heads for predicting bounding boxes
    parser.add_argument("--mlp_dropout", default=0.3, type=float)
    parser.add_argument(
        "--nsemcls",
        default=-1,
        type=int,
        help="Number of semantic object classes. Can be inferred from dataset",
    )

    ### Other model params
    parser.add_argument("--preenc_npoints", default=2048, type=int)
    parser.add_argument(
        "--pos_embed", default="fourier", type=str, choices=["fourier", "sine"]
    )
    parser.add_argument("--nqueries", default=128, type=int) # 256
    parser.add_argument("--use_color", default=False, action="store_true")


    ##### Testing #####
    parser.add_argument("--test_only", default=True, action="store_true")
    parser.add_argument("--test_ckpt", default=None, type=str)


    ##### Number of points #####
    parser.add_argument('--num_point', type=int, default=40000, help='Point Number [default: 40000]')

    ##### Data source #####
    parser.add_argument(
        "--data_source", choices=["realsense", "kinect", "scannet", "sunrgbd"]
    )

    return parser



def preprocess_point_cloud(point_cloud, num_points_to_sample):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''
    point_cloud = point_cloud[:,0:3] # do not use color for now
    # floor_height = np.percentile(point_cloud[:,2],0.99)
    # height = point_cloud[:,2] - floor_height
    # point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)
    point_cloud = random_sampling(point_cloud, num_points_to_sample)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0) # (1,40000,3)
    return pc

if __name__=='__main__':

    parser = make_args_parser()
    args = parser.parse_args()
    
    # Set file paths and dataset config

    if args.data_source == "scannet":
        pc_path = os.path.join(BASE_DIR, 'demo_files/input_pc_scannet.ply')
        point_cloud = read_ply(pc_path)
    elif args.data_source == "sunrgbd":
        pc_path = os.path.join(BASE_DIR, 'demo_files/input_pc_sunrgbd.ply')
        point_cloud = read_ply(pc_path)
    elif args.data_source == "realsense":
        pc_path = os.path.join(BASE_DIR, 'demo_files/input_pc_realsense.ply')
        point_cloud = read_ply_realsense(pc_path)
    elif args.data_source == "kinect":
        pc_path = os.path.join(BASE_DIR, 'demo_files/input_pc_kinect.ply')
        point_cloud = read_ply(pc_path)
    else:
        raise ValueError("Dataset source should be one of scannet, sunrgbd, realsense, or kinect")


    # Preprocess point cloud
    pc = preprocess_point_cloud(point_cloud, args.num_point)
    print('Loaded point cloud data: %s'%(pc_path))

    # Dataset config: use SUNRGB-D
    from datasets.sunrgbd import SunrgbdDatasetConfig as dataset_config
    # from datasets.scannet import ScannetDatasetConfig as dataset_config
    dataset_config = dataset_config()

    # Build model
    model, _ = build_model(args, dataset_config)

    # Load pre-trained weights
    sd = torch.load(args.test_ckpt, map_location=torch.device("cpu")) 
    model.load_state_dict(sd["model"]) 

    model = model.cuda()
    model.eval()

    device = torch.device("cuda")
    pc_torch = torch.from_numpy(pc).to(device)
    inputs = {'point_clouds': pc_torch, 'point_cloud_dims_min': pc_torch.min(1).values, 'point_cloud_dims_max': pc_torch.max(1).values}
    
    t_start = time.time()
    outputs = model(inputs)
    t_end= time.time()
    print("Infence time: ", t_end - t_start)


    # Save bboxes as point cloud (ply)
    centers = outputs["outputs"]["center_unnormalized"] 
    centers = centers.cpu().detach().numpy()
    lengths = outputs["outputs"]["size_unnormalized"]
    lengths = lengths.cpu().detach().numpy() 

    inds = outputs["outputs"]["objectness_prob"] > 0.5
    inds = inds.cpu()
    inds = inds[0,:]
    centers = centers[:,inds,:]
    lengths = lengths[:,inds,:]

    scene_bbox = np.concatenate((centers, lengths), 2)
    scene_bbox = scene_bbox[0,:,:]
    write_bbox(scene_bbox, "demo_files/output_bboxes.ply")

    print(" ")
    print("Number of objects detected: ", inds.sum().item())
    print(" ")

