import open3d as o3d
import numpy as np
from lietorch import SE3
import torch
import os
from telemetry_converter import TelemetryImporter
from natsort import natsorted
from gps_converter import ECEFtoENU
# import GPS data
import droid_backends

global_enu_llh0 = 0

CAM_POINTS = np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5]])

CAM_LINES = np.array([
    [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])

def create_camera_actor(g, scale=0.05):
    """ build open3d camera polydata """
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))

    color = (g * 1.0, 0.5 * (1-g), 0.9 * (1-g))
    camera_actor.paint_uniform_color(color)
    return camera_actor

def create_point_actor(points, colors):
    """ open3d point cloud from numpy array """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud

def load_dataset(path):
    poses_c_w = np.load(os.path.join(path,"poses.npy"))
    images_np = np.load(os.path.join(path,"images.npy"))
    intrinsics_np = np.load(os.path.join(path,"intrinsics.npy"))
    disps = torch.tensor(np.load(os.path.join(path,"disps.npy"))).float().cuda()
    intrinsics0 = torch.tensor(intrinsics_np)[0].float().cuda()
    # convert poses to 4x4 matrix
    poses = torch.tensor(poses_c_w).float().cuda()
    Ps = SE3(poses).inv().matrix().cpu().numpy()

    images = torch.tensor(images_np).float().cuda()
    images = images.cpu()[:,[2,1,0],3::8,3::8].permute(0,2,3,1) / 255.0
    points = droid_backends.iproj(SE3(poses).inv().data, disps,intrinsics0).cpu()

    thresh = 0.005 * torch.ones_like(disps.mean(dim=[1,2]))


    count = droid_backends.depth_filter(
        poses, disps, intrinsics0, torch.arange(0,poses_c_w.shape[0],1).long().cuda(), thresh)

    count = count.cpu()
    disps = disps.cpu()
    masks = ((count >= 2) & (disps > .5*disps.mean(dim=[1,2], keepdim=True)))
    
    camera_actors = []
    point_cloud = o3d.geometry.PointCloud()
    for i in range(poses_c_w.shape[0]):
        pose = Ps[i]

        mask = masks[i].reshape(-1)
        pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
        clr = images[i].reshape(-1, 3)[mask].cpu().numpy()
        
        cam_actor = create_camera_actor(True)
        cam_actor.transform(pose)
        camera_actors.append(cam_actor)
        ## add point actor ###
        point_cloud += create_point_actor(pts, clr)
        
    return camera_actors, point_cloud

camera_actors, point_cloud = load_dataset(
    "/home/zosurban/Projects/DROID-SLAM-urbste/data/steffen/merged")

# pcl1 = o3d.geometry.PointCloud()
# pcl1.points = o3d.utility.Vector3dVector(p_w_c1)
# pcl1.paint_uniform_color([1, 0.706, 0])

o3d.visualization.draw_geometries([camera_actors])

