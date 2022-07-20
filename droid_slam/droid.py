import torch
import lietorch
import numpy as np

from droid_net import DroidNet
from depth_video import DepthVideo
#from droid_slam.frame_localizer import ReLocalizer
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from torch.multiprocessing import Process
from factor_graph import FactorGraph


class Droid:
    def __init__(self, args):
        super(Droid, self).__init__()
        self.load_weights(args.weights)
        self.args = args
        self.disable_vis = args.disable_vis

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, args.do_localization, thresh=args.filter_thresh)

        # frontend process
        self.frontend = DroidFrontend(self.net, self.video, self.args)
        
        # backend process
        self.backend = DroidBackend(self.net, self.video, self.args)

        # relocalizer
        # self.localizer = ReLocalizer(self.net, self.video, self.args)

        # visualizer
        if not self.disable_vis:
            from visualization import droid_visualization
            self.visualizer = Process(target=droid_visualization, args=(self.video,))
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)

        # localize new video
        self.localize_new_video = args.do_localization
        self.is_localized = False

        self.all_video_timestamps_ns = []

    def load_weights(self, weights):
        """ load trained model weights """

        print(weights)
        self.net = DroidNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()

    def track(self, tstamp, image, depth=None, intrinsics=None):
        """ main thread - update map """
        with torch.no_grad():
            # check there is enough motion             
            self.filterx.track(tstamp, image, depth, intrinsics)

            self.frontend()

            # global bundle adjustment
            # self.backend()
            self.all_video_timestamps_ns.append(tstamp)
            
        return self.frontend.t1-1

    def terminate(self, stream=None):
        """ terminate the visualization process, return poses [t, q] """

        del self.frontend

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(5)

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(5)

        #camera_trajectory = self.traj_filler(stream)
        #return camera_trajectory.inv().data.cpu().numpy()

    def load_from_saved_reconstruction(self, recon_path):
        # load numpy arrays
        tstamps = np.load("{}/tstamps.npy".format(recon_path))
        images = np.load("{}/images.npy".format(recon_path))
        disps = np.load("{}/disps.npy".format(recon_path))
        poses = np.load("{}/poses.npy".format(recon_path))
        intrinsics = np.load("{}/intrinsics.npy".format(recon_path))
        fmaps = np.load("{}/fmaps.npy".format(recon_path))
        nets = np.load("{}/nets.npy".format(recon_path))
        inps = np.load("{}/inps.npy".format(recon_path))

        # init tensors
        self.video.images = torch.tensor(images, device="cuda", dtype=torch.uint8).share_memory_()
        self.video.disps_up = torch.tensor(disps, device="cuda").share_memory_()
        #depth = self.video.disps_up[:,3::8,3::8]
        #self.disps_sens = torch.where(depth>0, 1.0/depth, depth)

        self.video.poses = torch.tensor(poses, device="cuda", dtype=torch.float).share_memory_()
        self.video.tstamp = torch.tensor(tstamps, device="cuda", dtype=torch.float).share_memory_()
        self.video.intrinsics = torch.tensor(intrinsics, device="cuda", dtype=torch.float).share_memory_()

        self.video.fmaps = torch.tensor(fmaps, dtype=torch.half, device="cuda").share_memory_()
        self.video.nets = torch.tensor(nets, dtype=torch.half, device="cuda").share_memory_()
        self.video.inps = torch.tensor(inps, dtype=torch.half, device="cuda").share_memory_()

        self.video.counter = torch.multiprocessing.Value('i', self.video.fmaps.shape[0])

        # initialize the factor graph from the old video
        #t = self.video.counter.value
        #self.frontend.graph = FactorGraph(self.video, self.net.update, corr_impl="alt", max_factors=10*t, upsample=True)

        # add only the first 5 frames from the old trajectory
        #self.frontend.graph.add_proximity_factors(0, 0, 2, 2, 0.3, 16)
        # for i in range(5):
        #     for j in range(i+1,5):
        #         self.frontend.graph.add_factors([i],[j])

        # # extend tensors
        # self.video.images = torch.cat([self.video.images, torch.zeros_like(self.video.images)],0)
        # self.video.disps_up = torch.cat([self.video.disps_up, torch.zeros_like(self.video.disps_up)],0)
        # self.video.disps_sens = torch.cat([self.video.disps_sens, torch.zeros_like(self.video.disps_sens)],0)

        # self.video.poses = torch.cat([self.video.poses, torch.zeros_like(self.video.poses)],0)
        # self.video.tstamp = torch.cat([self.video.tstamp, torch.zeros_like(self.video.tstamp)],0)
        # self.video.intrinsics = torch.cat([self.video.intrinsics, torch.zeros_like(self.video.intrinsics)],0)

        # self.video.fmaps = torch.cat([self.video.fmaps, torch.zeros_like(self.video.fmaps)],0)
        # self.video.nets = torch.cat([self.video.nets, torch.zeros_like(self.video.nets)],0)
        # self.video.inps = torch.cat([self.video.inps, torch.zeros_like(self.video.inps)],0)

        # print("Successfully loaded video and created factor graph.")