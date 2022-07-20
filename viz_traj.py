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

def load_dataset(path, telemetry_file, llh0):

    poses_c_w = np.load(os.path.join(path,"poses.npy"))
    frametimes_slam_ns = np.load(os.path.join(path,"tstamps.npy")).astype(np.int)

    tel_importer = TelemetryImporter()
    tel_importer.read_gopro_telemetry(telemetry_file)
    gps_xyz, _ = tel_importer.get_gps_pos_at_frametimes(frametimes_slam_ns)
    gravity_vectors = tel_importer.get_gravity_vector_at_times(frametimes_slam_ns)

    p_w_c = SE3(torch.tensor(poses_c_w)).inv().translation()[:,0:3]
    q_w_c = SE3(torch.tensor(poses_c_w)).inv().data[:,3:]
    #get gps at slam keyframe poses
    # same_timestamps = list(set.intersection(set(img_timestamps_ns), set(frametimes_slam)))
    # valid_mask = np.logical_and(same_timestamps > list(gps_xyz.keys())[0], same_timestamps < list(gps_xyz.keys())[-1])
    # same_timestamps_masked = [t for t, m in zip(same_timestamps,valid_mask) if m]

    # mask_kfs_poses = [p_w_c[same_timestamps.index(t),:] for t, m in zip(same_timestamps,valid_mask) if m]
        
    #gps_enu_at_kfs = [ECEFtoENU(gps_xyz[int(key)], llh0) if int(key) in gps_xyz else print(key) for key in same_timestamps_masked]
    #np.array(gps_enu_at_kfs, dtype=np.float32), 
    return p_w_c.numpy(), q_w_c.numpy(), gravity_vectors

tel_importer = TelemetryImporter()
tel_importer.read_gopro_telemetry("/media/Data/Sparsenet/OrbSlam3/TestMappingRelocalization/JenzigTrailsJune/bike1_trail1_linear.json")
llh0 = tel_importer.telemetry["gps_llh"][0]

p_w_c2, q_w_c2, grav2 = load_dataset(
    "/media/Data/Sparsenet/OrbSlam3/TestMappingRelocalization/JenzigTrailsJune/bike2_trail1_results",
    "/media/Data/Sparsenet/OrbSlam3/TestMappingRelocalization/JenzigTrailsJune/bike2_trail1_linear.json",
    llh0)



p_w_c1, q_w_c1, grav1 = load_dataset(
    "/media/Data/Sparsenet/OrbSlam3/TestMappingRelocalization/JenzigTrailsJune/bike1_trail1_results",
    "/media/Data/Sparsenet/OrbSlam3/TestMappingRelocalization/JenzigTrailsJune/bike1_trail1_linear.json",
    llh0)


# pcl1_gps = o3d.geometry.PointCloud()
# pcl1_gps.points = o3d.utility.Vector3dVector(gps_w_c1)
# pcl2_gps = o3d.geometry.PointCloud()
# pcl2_gps.points = o3d.utility.Vector3dVector(gps_w_c2)

# corr_list = np.arange(0, p_w_c1.shape[0])
# corres1 = o3d.utility.Vector2iVector(np.stack([corr_list,corr_list]).T)
# corr_list = np.arange(0, p_w_c2.shape[0])
# corres2 = o3d.utility.Vector2iVector(np.stack([corr_list,corr_list]).T)

# # Align visual to GPS
# trafo_estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)

# T1 = o3d.pipelines.registration.registration_ransac_based_on_correspondence(pcl1, pcl1_gps, corres1, 1,
#     o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True), ransac_n=20)
# T2 = o3d.pipelines.registration.registration_ransac_based_on_correspondence(pcl2, pcl2_gps, corres2, 1,
#     o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True), ransac_n=20)


# find rotation to world frame for first trajectory

def rot_between_vectors(a,b):
    # rotates a -> b
    def skew(vector):
        return np.array([[0, -vector[2], vector[1]], 
                        [vector[2], 0, -vector[0]], 
                        [-vector[1], vector[0], 0]])

    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    v = np.cross(a,b)
    c = np.dot(a,b)
    s = np.linalg.norm(v)

    R = np.eye(3) + skew(v) + np.linalg.matrix_power(skew(v),2)*((1-c)/s**2)

    return R

from scipy.spatial.transform import Rotation as R
R_imu_to_cam = R.from_quat([-0.0002947483347789445,-0.7104874521708413,0.7037096975195423,0.000393761552164708]).as_matrix().T

def get_rot_to_worldframe(gravity_vecors, q_w_c, world_vec=np.array([0,0,-1])):
    mean_vec = []
    for i in range(gravity_vecors.shape[0]):
        R_cam_to_world = rot_between_vectors(gravity_vecors[i], world_vec)
        Rij = R_cam_to_world @ R.from_quat(q_w_c[i,:]).as_matrix().T
        mean_vec.append(R.from_matrix(Rij).as_rotvec())

    return R.from_rotvec(np.median(mean_vec,0)).as_matrix()

R1_to_grav = get_rot_to_worldframe(grav1, q_w_c1)
R2_to_grav = get_rot_to_worldframe(grav2, q_w_c2)




# # init gps and heading direction
# print("Initializing GPS and heading direction...")
# self.mean_gps_ecef = np.zeros(3)
# for gps_time in self.cam_gps:
#     self.mean_gps_ecef += np.asarray(self.cam_gps[gps_time])
# self.mean_gps_ecef /= len(self.cam_gps)
# # get mean ecef coords 
# lla0 = ECEFToLLA(self.mean_gps_ecef)

# for gps_time in self.cam_gps:
#     self.cam_gps[gps_time] = ECEFtoENU(self.cam_gps[gps_time], lla0)
# frame_times = list(self.cam_gps.keys())
# self.heading_vector_forward_motion = np.array(self.cam_gps[frame_times[300]]) - np.array(self.cam_gps[frame_times[0]])
# self.heading_vector_forward_motion[2] = 0.0
# self.heading_vector_forward_motion /= np.linalg.norm(self.heading_vector_forward_motion)
# angle_to_y_axis = np.arcsin(self.heading_vector_forward_motion[0]/np.sqrt(
#     self.heading_vector_forward_motion[1]**2 + self.heading_vector_forward_motion[0]**2))
# self.heading_mat = R.from_rotvec([0,0,angle_to_y_axis]).as_matrix()




pcl1 = o3d.geometry.PointCloud()
pcl1.points = o3d.utility.Vector3dVector((R1_to_grav@p_w_c1.T).T)
pcl1.paint_uniform_color([1, 0.706, 0])
pcl2 = o3d.geometry.PointCloud()
pcl2.points = o3d.utility.Vector3dVector((R2_to_grav@p_w_c2.T).T)
pcl2.paint_uniform_color([1, 0,  0.706])


world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])

visualizer = o3d.visualization.Visualizer()
visualizer.create_window(width=600, height=500, left=450, top=250)
visualizer.add_geometry(pcl1)
visualizer.add_geometry(pcl2)
visualizer.add_geometry(world_frame)

view_ctl = visualizer.get_view_control()
view_ctl.set_front((1, 0, 0))
view_ctl.set_up((0, 0, 1))  # can't set z-axis as the up-axis
view_ctl.set_lookat((0, 0, 0))

visualizer.run()
visualizer.destroy_window()
