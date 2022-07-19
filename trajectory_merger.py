import sys
sys.path.append('droid_slam')


import numpy as np
from tqdm import tqdm
import numpy as np
import torch

import lietorch
import cv2
import os
import glob 
import time
import argparse


from torch.multiprocessing import Process
from droid_merger import DroidMerger

import torch.nn.functional as F
import natsort

def load_from_saved_reconstruction( recon_path, to_align=False):
    # load numpy arrays
    tstamps = np.load("{}/tstamps.npy".format(recon_path))
    images = np.load("{}/images.npy".format(recon_path))
    disps = np.load("{}/disps.npy".format(recon_path))
    if to_align:
        poses = np.load("{}/poses_transformed.npy".format(recon_path))
    else:
        poses = np.load("{}/poses.npy".format(recon_path))
    intrinsics = np.load("{}/intrinsics.npy".format(recon_path))
    fmaps = np.load("{}/fmaps.npy".format(recon_path))
    nets = np.load("{}/nets.npy".format(recon_path))
    inps = np.load("{}/inps.npy".format(recon_path))

    return tstamps, images, disps, poses, intrinsics, fmaps, nets, inps

traj1 = "/media/Data/projects/DROID-SLAM/data/yt_gopro_mtb/NH_Youtube/NH_results"
traj2 = "/media/Data/projects/DROID-SLAM/data/yt_gopro_mtb/NH_Youtube/MN_results"

merged_path = "/media/Data/projects/DROID-SLAM/data/yt_gopro_mtb/NH_Youtube/common"

tstamps1, images1, disps1, poses1, intrinsics1, fmaps1, nets1, inps1 = \
    load_from_saved_reconstruction(traj1, True)

tstamps2, images2, disps2, poses2, intrinsics2, fmaps2, nets2, inps2 = \
    load_from_saved_reconstruction(traj2)


if __name__ == '__main__':
    import os
    if not os.path.exists(merged_path):
        os.makedirs(merged_path)
    np.save(os.path.join(merged_path, "tstamps.npy"), np.concatenate([tstamps1, tstamps2],0))
    np.save(os.path.join(merged_path, "images.npy"), np.concatenate([images1, images2],0))
    np.save(os.path.join(merged_path, "disps.npy"), np.concatenate([disps1, disps2],0))
    np.save(os.path.join(merged_path, "intrinsics.npy"), np.concatenate([intrinsics1, intrinsics2],0))
    np.save(os.path.join(merged_path, "poses.npy"), np.concatenate([poses1, poses2],0))
    np.save(os.path.join(merged_path, "fmaps.npy"), np.concatenate([fmaps1, fmaps2],0))
    np.save(os.path.join(merged_path, "nets.npy"), np.concatenate([nets1, nets2],0))
    np.save(os.path.join(merged_path, "inps.npy"), np.concatenate([inps1, inps2],0))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')


    args.stereo = False
    args.do_localization = False
    args.image_size = [270, 480]
    droid = DroidMerger(args)

    droid.load_from_saved_reconstruction(merged_path)
    torch.cuda.empty_cache()
    print("#" * 32)
    droid.merge()
