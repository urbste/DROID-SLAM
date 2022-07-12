import open3d as o3d
import numpy as np
from lietorch import SE3
import torch
import os
from telemetry_converter import TelemetryImporter
from natsort import natsorted
from gps_converter import ECEFtoENU
# import GPS data

global_enu_llh0 = 0

def load_dataset(img_path, path, telemetry_file, llh0):

    img_files = natsorted(os.listdir(img_path)) 
    img_timestamps = [int(n[:n.rfind(".")]) for n in img_files]
    poses_c_w = np.load(os.path.join(path,"poses.npy"))
    frametimes_slam = np.load(os.path.join(path,"tstamps.npy")).astype(np.int)
    tel_importer = TelemetryImporter()
    tel_importer.read_gopro_telemetry(telemetry_file)
    gps_xyz, _ = tel_importer.get_gps_pos_at_frametimes(img_timestamps)

    p_w_c = SE3(torch.tensor(poses_c_w)).inv().translation()[:,0:3]
    
    #get gps at slam keyframe poses
    same_timestamps = list(set.intersection(set(img_timestamps), set(frametimes_slam)))
    valid_mask = np.logical_and(same_timestamps > list(gps_xyz.keys())[0], same_timestamps < list(gps_xyz.keys())[-1])
    same_timestamps_masked = [t for t, m in zip(same_timestamps,valid_mask) if m]

    mask_kfs_poses = [p_w_c[same_timestamps.index(t),:] for t, m in zip(same_timestamps,valid_mask) if m]
        
    gps_enu_at_kfs = [ECEFtoENU(gps_xyz[int(key)], llh0) if int(key) in gps_xyz else print(key) for key in same_timestamps_masked]

    return np.array(mask_kfs_poses), np.array(gps_enu_at_kfs, dtype=np.float32)

tel_importer = TelemetryImporter()
tel_importer.read_gopro_telemetry("/home/zosurban/Downloads/SLAM/bike1_trail1_linear.json")
llh0 = tel_importer.telemetry["gps_llh"][0]

p_w_c2, gps_w_c2 = load_dataset(
    "/home/zosurban/Downloads/SLAM/bike2_trail1_linear_imgs",
    "/home/zosurban/Downloads/SLAM/bike2_result",
    "/home/zosurban/Downloads/SLAM/bike2_trail1_linear.json",
    llh0)



p_w_c1, gps_w_c1 = load_dataset(
    "/home/zosurban/Downloads/SLAM/bike1_trail1_linear_imgs",
    "/home/zosurban/Downloads/SLAM/bike1_result",
    "/home/zosurban/Downloads/SLAM/bike1_trail1_linear.json",
    llh0)

pcl1 = o3d.geometry.PointCloud()
pcl1.points = o3d.utility.Vector3dVector(p_w_c1)
pcl1.paint_uniform_color([1, 0.706, 0])
pcl2 = o3d.geometry.PointCloud()
pcl2.points = o3d.utility.Vector3dVector(p_w_c2)
pcl2.paint_uniform_color([1, 0,  0.706])

pcl1_gps = o3d.geometry.PointCloud()
pcl1_gps.points = o3d.utility.Vector3dVector(gps_w_c1)
pcl2_gps = o3d.geometry.PointCloud()
pcl2_gps.points = o3d.utility.Vector3dVector(gps_w_c2)

corr_list = np.arange(0, p_w_c1.shape[0])
corres1 = o3d.utility.Vector2iVector(np.stack([corr_list,corr_list]).T)
corr_list = np.arange(0, p_w_c2.shape[0])
corres2 = o3d.utility.Vector2iVector(np.stack([corr_list,corr_list]).T)

# Align visual to GPS
trafo_estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)

T1 = o3d.pipelines.registration.registration_ransac_based_on_correspondence(pcl1, pcl1_gps, corres1, 1,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True), ransac_n=20)
T2 = o3d.pipelines.registration.registration_ransac_based_on_correspondence(pcl2, pcl2_gps, corres2, 1,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True), ransac_n=20)

o3d.visualization.draw_geometries([pcl1.transform(T1.transformation),pcl2.transform(T2.transformation), pcl1_gps,pcl2_gps])

