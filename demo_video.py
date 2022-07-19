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


def save_reconstruction(droid, reconstruction_path):

    from pathlib import Path
    import random
    import string

    t = droid.video.counter.value
    tstamps = droid.video.tstamp[:t].cpu().numpy()
    images = droid.video.images[:t].cpu().numpy()
    disps = droid.video.disps_up[:t].cpu().numpy()
    poses = droid.video.poses[:t].cpu().numpy()
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()

    fmaps = droid.video.fmaps[:t].cpu().numpy()
    nets = droid.video.nets[:t].cpu().numpy()
    inps = droid.video.inps[:t].cpu().numpy()

    Path("{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)
    np.save("{}/tstamps.npy".format(reconstruction_path), tstamps)
    np.save("{}/images.npy".format(reconstruction_path), images)
    np.save("{}/disps.npy".format(reconstruction_path), disps)
    np.save("{}/poses.npy".format(reconstruction_path), poses)
    np.save("{}/intrinsics.npy".format(reconstruction_path), intrinsics)
    np.save("{}/fmaps.npy".format(reconstruction_path), fmaps)
    np.save("{}/nets.npy".format(reconstruction_path), nets)
    np.save("{}/inps.npy".format(reconstruction_path), inps)

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
    parser.add_argument("--calib", default="/media/Data/projects/DROID-SLAM/calib/gopro9_linear.txt", type=str, help="path to calibration file")
    parser.add_argument("--stride", default=1, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=150)
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
        default="/media/Data/projects/DROID-SLAM/data/yt_gopro_mtb/NH_Youtube/MN_results", 
        help="path to saved reconstruction")
    parser.add_argument("--do_localization", default=False)
    parser.add_argument("--t0", type=float, default=0.0)
    parser.add_argument("--tend", type=float, default=-1.0)
    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    droid = None

    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True

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
    num_tracked = 0
    while True:
        ret, I = cap.read()
        ts_ns = int(1e6*cap.get(cv2.CAP_PROP_POS_MSEC))
        ts_s = ts_ns*1e-9
        if ts_s < args.t0:
            continue
        if ts_s > args.tend and args.tend > args.t0:
            break

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

        if droid is None:
            args.image_size = [image_t.shape[2], image_t.shape[3]]
            print(args.image_size)
            droid = Droid(args)
        
        num_kfs = droid.track(ts_ns, image_t, intrinsics=intrinsics_t)
        if num_kfs >= args.buffer-2:
            print("Buffer full. Stopping SLAM.")
            break

    if args.reconstruction_path is not None:
        save_reconstruction(droid, args.reconstruction_path)

    traj_est = droid.terminate()

