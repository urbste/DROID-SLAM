import sys
sys.path.append('droid_slam')

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
from droid import Droid

import torch.nn.functional as F
import natsort

def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(1)

def image_stream(imagedir, calib, stride):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = natsort.natsorted(os.listdir(imagedir))[::stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)
        time_ns = int(imfile[:imfile.rfind(".")])
        yield time_ns, image[None], intrinsics

def load_reconstruction(args):

    from pathlib import Path
    import random
    import string    
    images = np.load("{}/images.npy".format(args.reconstruction_path))
    args.image_size = [images.shape[2], images.shape[3]]
    droid = Droid(args)
    droid.load_from_saved_reconstruction(args.reconstruction_path)
    last_timestamp = sorted(np.load("{}/tstamps.npy".format(args.reconstruction_path)).tolist())[-1]
    return droid, last_timestamp

def convert_image(image, intrinsics):

    h0, w0, _ = image.shape
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

    image = cv2.resize(image, (w1, h1))
    image = image[:h1-h1%8, :w1-w1%8]
    image_t = torch.as_tensor(image).permute(2, 0, 1)

    intrinsics_t = torch.as_tensor([fx, fy, cx, cy])

    intrinsics_t[0::2] *= (w1 / w0)
    intrinsics_t[1::2] *= (h1 / h0)

    return image_t[None], intrinsics_t

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="/media/Data/projects/DROID-SLAM/data/yt_gopro_mtb/NH_Youtube/MNYoutube.mp4", type=str, help="path to image directory")
    parser.add_argument("--calib", default="/media/Data/projects/DROID-SLAM/calib/gopro9_wide.txt", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=1, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=20)
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
    parser.add_argument("--reconstruction_path", 
        default="/media/Data/projects/DROID-SLAM/data/yt_gopro_mtb/NH_Youtube/NH_results/", 
        help="path to saved reconstruction")
    parser.add_argument("--do_localization", default=True)
    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')


    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True

    droid, last_timestamp = load_reconstruction(args)

    calib = np.loadtxt(args.calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    tstamps = []

    invalid_images = 0
    had_zero = False
    frame_id = 0
    cap = cv2.VideoCapture(args.video)
    while True:
        ret, I = cap.read()
        ts_ns = int(1e6*cap.get(cv2.CAP_PROP_POS_MSEC))
        if not ret:
            invalid_images += 1
            if invalid_images > 100:
                break
            continue
        if had_zero and ts_ns == 0:
            continue
        I = cv2.resize(I, (480, 270))
        if len(calib) > 4:
            I = cv2.undistort(I, K, calib[4:])

        image_t, intrinsics_t = convert_image(I, K)

        if not args.disable_vis:
            show_image(I)
        
        droid.track(ts_ns + last_timestamp + 1, image_t, intrinsics=intrinsics_t)

