import open3d as o3d
import numpy as np
from lietorch import SE3, Sim3
import torch
import os
from telemetry_converter import TelemetryImporter
from natsort import natsorted
from gps_converter import ECEFtoENU
# import GPS data

global_enu_llh0 = 0

def load_dataset(path):

    poses_c_w = np.load(os.path.join(path,"poses.npy"))
    # frametimes_slam = np.load(os.path.join(path,"tstamps.npy")).astype(np.int)

    p_w_c = SE3(torch.tensor(poses_c_w)).inv().translation()[:,0:3]
        
    return np.array(p_w_c)

def save_transfromed_poses(path, T12):
    poses_c_w = np.load(os.path.join(path,"poses.npy"))
    from scipy.spatial.transform import Rotation as R


    poses_old = SE3(torch.tensor(poses_c_w).float()).inv()
    p_new = (T12[:3,:3] @ np.array(poses_old.data[:,:3]).T).T + T12[:3,3]

    s = np.linalg.det(T12[:3,:3])**(1/3)
    R12 = R.from_matrix(T12[:3,:3] / s)

    q_new = (R.from_quat(poses_old.data[:,3:]) * R12.inv()).as_quat()

    #t = T12[:3,3]
    poses_new = torch.tensor([p_new[:,0],p_new[:,1],p_new[:,2],q_new[:,0],q_new[:,1],q_new[:,2],q_new[:,3]])

    poses_c_w = SE3(poses_new.T).inv()
    #poses_transformed = (Sim3(T12_sim) * ).inv()
    
    np.save(os.path.join(path,"poses_transformed.npy"), poses_c_w.data.numpy())

    return poses_new.T.numpy()

p_w_c2 = load_dataset(
    "/media/Data/projects/DROID-SLAM/data/yt_gopro_mtb/NH_Youtube/MN_results")

p_w_c1 = load_dataset(
    "/media/Data/projects/DROID-SLAM/data/yt_gopro_mtb/NH_Youtube/NH_results")

pcl1 = o3d.geometry.PointCloud()
pcl1.points = o3d.utility.Vector3dVector(p_w_c1)
pcl1.paint_uniform_color([1, 0.706, 0])
pcl2 = o3d.geometry.PointCloud()
pcl2.points = o3d.utility.Vector3dVector(p_w_c2)
pcl2.paint_uniform_color([1, 0,  0.706])


corr_list = np.arange(0, p_w_c1.shape[0])
corres1 = o3d.utility.Vector2iVector(np.stack([corr_list,corr_list]).T)
corr_list = np.arange(0, p_w_c2.shape[0])
corres2 = o3d.utility.Vector2iVector(np.stack([corr_list,corr_list]).T)

# Align visual to GPS
trafo_estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)

T12 = o3d.pipelines.registration.registration_ransac_based_on_correspondence(pcl1, pcl2, corres1, 0.01,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True), ransac_n=6)

transformed_poses = save_transfromed_poses("/media/Data/projects/DROID-SLAM/data/yt_gopro_mtb/NH_Youtube/NH_results", T12.transformation)

pcl1 = o3d.geometry.PointCloud()
pcl1.points = o3d.utility.Vector3dVector(transformed_poses[:,0:3])
pcl1.paint_uniform_color([1, 0.706, 0])
o3d.visualization.draw_geometries([pcl1, pcl2])
