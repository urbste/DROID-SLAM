import sys
sys.path.append('droid_slam')


import numpy as np
import numpy as np
import torch

from lietorch import SE3

import os
import argparse


from torch.multiprocessing import Process
from droid_merger import DroidMerger

import torch.nn.functional as F
import natsort

def load_from_saved_reconstruction( recon_path, to_align=False):
    # load numpy arrays
    tstamps = np.load("{}/kf_tstamps.npy".format(recon_path))
    all_tstamps = np.load("{}/all_tstamps.npy".format(recon_path))

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

    return tstamps, all_tstamps, images, disps, poses, intrinsics, fmaps, nets, inps

def lerp(positions, position_times, interp_times):
    x_interp = np.interp(interp_times, position_times, positions[:,0])
    y_interp = np.interp(interp_times, position_times, positions[:,1])
    z_interp = np.interp(interp_times, position_times, positions[:,2])
    return np.stack([x_interp,y_interp,z_interp],1)

def save_merged_reconstructions(droid, reconstruction_path, split_timestamp_ns):

    from pathlib import Path

    t = droid.video.counter.value
    tstamps = droid.video.tstamp[:t].cpu().numpy()
    # split the merged trajectories in two again
    split_idx = np.where((tstamps == split_timestamp_ns)==True)[0][0]

    tstamps1_s = tstamps[0:split_idx]*1e-9
    tstamps2_s = tstamps[split_idx:]*1e-9

    all_tstamps = droid.all_tstamps.astype(np.float32)
    split_idx_all_tstamps = np.where((all_tstamps == split_timestamp_ns)==True)[0][0]

    all_tstamps1_s = all_tstamps[:split_idx_all_tstamps]*1e-9
    all_tstamps2_s = all_tstamps[split_idx_all_tstamps:]*1e-9

    images1 = droid.video.images[:split_idx].cpu().numpy()
    images2 = droid.video.images[split_idx:].cpu().numpy()

    disps_up1 = droid.video.disps_up[:split_idx].cpu().numpy()
    disps_up2 = droid.video.disps_up[split_idx:].cpu().numpy()

    poses_c_w1 = droid.video.poses[:split_idx].cpu().numpy()
    poses_c_w2 = droid.video.poses[split_idx:].cpu().numpy()

    intrinsics1 = droid.video.intrinsics[:split_idx].cpu().numpy()
    intrinsics2 = droid.video.intrinsics[split_idx:].cpu().numpy()

    # fmaps = droid.video.fmaps[:t].cpu().numpy()
    # nets = droid.video.nets[:t].cpu().numpy()
    # inps = droid.video.inps[:t].cpu().numpy()

    from scipy.spatial.transform import Slerp
    from scipy.spatial.transform import Rotation as R
    T1 = SE3(torch.tensor(poses_c_w1)).inv().data.cpu().numpy()
    T2 = SE3(torch.tensor(poses_c_w2)).inv().data.cpu().numpy()

    q1_kfs = Slerp(tstamps1_s, R.from_quat(T1[:,3:]),)
    q2_kfs = Slerp(tstamps2_s, R.from_quat(T2[:,3:]))

    max_kf_time1 = np.where((tstamps1_s[-1] < all_tstamps1_s)==True)[0][0]
    max_kf_time2 = np.where((tstamps2_s[-1] < all_tstamps2_s)==True)[0][0]
    interp_q1 = q1_kfs(all_tstamps1_s[:max_kf_time1])
    interp_q2 = q2_kfs(all_tstamps2_s[:max_kf_time2])

    interp_p1 = lerp(T1[:,0:3], tstamps1_s, all_tstamps1_s[:max_kf_time1])
    interp_p2 = lerp(T2[:,0:3], tstamps2_s, all_tstamps2_s[:max_kf_time2])

    interp_poses1 = np.concatenate([interp_p1,interp_q1.as_quat()],1)
    interp_poses2 = np.concatenate([interp_p2,interp_q2.as_quat()],1)

    Path("{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)
    np.save("{}/kf_tstamps_1.npy".format(reconstruction_path), tstamps1*1e9)
    np.save("{}/all_tstamps_1.npy".format(reconstruction_path), all_tstamps1_s*1e9)
    np.save("{}/images_1.npy".format(reconstruction_path), images1)
    np.save("{}/disps_1.npy".format(reconstruction_path), disps_up1)
    np.save("{}/poses_1.npy".format(reconstruction_path), poses1)
    np.save("{}/poses_interp_1.npy".format(reconstruction_path), interp_poses1)
    np.save("{}/intrinsics_1.npy".format(reconstruction_path), intrinsics1)
    # np.save("{}/fmaps_1.npy".format(reconstruction_path), fmaps)
    # np.save("{}/nets_1.npy".format(reconstruction_path), nets)
    # np.save("{}/inps_1.npy".format(reconstruction_path), inps)

    np.save("{}/kf_tstamps_2.npy".format(reconstruction_path), tstamps2*1e9)
    np.save("{}/all_tstamps_2.npy".format(reconstruction_path), all_tstamps2_s*1e9)

    np.save("{}/images_2.npy".format(reconstruction_path), images2)
    np.save("{}/disps_2.npy".format(reconstruction_path), disps_up2)
    np.save("{}/poses_2.npy".format(reconstruction_path), poses2)
    np.save("{}/poses_interp_2.npy".format(reconstruction_path), interp_poses2)
    np.save("{}/intrinsics_2.npy".format(reconstruction_path), intrinsics2)
    # np.save("{}/fmaps_2.npy".format(reconstruction_path), fmaps)
    # np.save("{}/nets_2.npy".format(reconstruction_path), nets)
    # np.save("{}/inps_2.npy".format(reconstruction_path), inps)

if __name__ == '__main__':


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path1", default="/media/Data/Sparsenet/OrbSlam3/TestMappingRelocalization/JenzigTrailsJune/bike1_trail1_results", type=str, help="path to image directory")
    parser.add_argument("--path2", default="/media/Data/Sparsenet/OrbSlam3/TestMappingRelocalization/JenzigTrailsJune/bike2_trail1_results", type=str, help="path to image directory")
    parser.add_argument("--merged", default="/media/Data/Sparsenet/OrbSlam3/TestMappingRelocalization/JenzigTrailsJune/merged", type=str, help="path to image directory")

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


    # first copy stuff to new folder
    tstamps1, all_tstamps1, images1, disps1, poses1, intrinsics1, fmaps1, nets1, inps1 = \
        load_from_saved_reconstruction(args.path1, True)

    tstamps2, all_tstamps2, images2, disps2, poses2, intrinsics2, fmaps2, nets2, inps2 = \
        load_from_saved_reconstruction(args.path2)

    if not os.path.exists(args.merged):
        os.makedirs(args.merged)
    np.save(os.path.join(args.merged, "kf_tstamps.npy"), np.concatenate([tstamps1, tstamps2],0))
    np.save(os.path.join(args.merged, "all_tstamps.npy"), np.concatenate([all_tstamps1, all_tstamps2],0))

    np.save(os.path.join(args.merged, "images.npy"), np.concatenate([images1, images2],0))
    np.save(os.path.join(args.merged, "disps.npy"), np.concatenate([disps1, disps2],0))
    np.save(os.path.join(args.merged, "intrinsics.npy"), np.concatenate([intrinsics1, intrinsics2],0))
    np.save(os.path.join(args.merged, "poses.npy"), np.concatenate([poses1, poses2],0))
    np.save(os.path.join(args.merged, "fmaps.npy"), np.concatenate([fmaps1, fmaps2],0))
    np.save(os.path.join(args.merged, "nets.npy"), np.concatenate([nets1, nets2],0))
    np.save(os.path.join(args.merged, "inps.npy"), np.concatenate([inps1, inps2],0))

    args.stereo = False
    args.do_localization = False
    args.image_size = [328, 584]
    args.disable_vis = True
    droid = DroidMerger(args)

    droid.load_from_saved_reconstruction(args.merged)
    torch.cuda.empty_cache()
    print("#" * 32)
    droid.merge(10)


    save_merged_reconstructions(droid, args.merged, split_timestamp_ns=tstamps2[0])