import torch
import lietorch
import numpy as np

from droid_net import DroidNet
from depth_video import DepthVideo
from droid_slam.frame_localizer import ReLocalizer
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from torch.multiprocessing import Process


class Droid:
    def __init__(self, args):
        super(Droid, self).__init__()
        self.load_weights(args.weights)
        self.args = args
        self.disable_vis = args.disable_vis

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh)

        # frontend process
        self.frontend = DroidFrontend(self.net, self.video, self.args)
        
        # backend process
        self.backend = DroidBackend(self.net, self.video, self.args)

        # relocalizer
        self.localizer = ReLocalizer(self.net, self.video, self.args)

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
            if self.localize_new_video and not self.is_localized:
                self.is_localized = self.localize(tstamp, image, None, intrinsics)
            else:
                # check there is enough motion
                self.filterx.track(tstamp, image, depth, intrinsics)

            # local bundle adjustment
            if self.localize_new_video and self.is_localized:
                self.frontend.is_initialized = True

            # global bundle adjustment
            # self.backend()

    def localize(self, tstamp, image, depth=None, intrinsics=None):
        with torch.no_grad():
            
            return self.localizer.localize_frame(tstamp, image, 8.5, None, intrinsics)


    def terminate(self, stream=None):
        """ terminate the visualization process, return poses [t, q] """

        del self.frontend

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(7)

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(12)

        camera_trajectory = self.traj_filler(stream)
        return camera_trajectory.inv().data.cpu().numpy()

    def load_from_saved_reconstruction(self, recon_path):
        tstamps = np.load("{}/tstamps.npy".format(recon_path))
        images = np.load("{}/images.npy".format(recon_path))
        disps = np.load("{}/disps.npy".format(recon_path))
        poses = np.load("{}/poses.npy".format(recon_path))
        intrinsics = np.load("{}/intrinsics.npy".format(recon_path))
        fmaps = np.load("{}/fmaps.npy".format(recon_path))
        nets = np.load("{}/nets.npy".format(recon_path))
        inps = np.load("{}/inps.npy".format(recon_path))

        self.video.images = torch.tensor(images, device="cuda", dtype=torch.uint8).share_memory_()
        self.video.disps_up = torch.tensor(disps, device="cuda").share_memory_()
        depth = self.video.disps_up[:,3::8,3::8]
        self.disps_sens = torch.where(depth>0, 1.0/depth, depth)

        self.video.poses = torch.tensor(poses, device="cuda", dtype=torch.float).share_memory_()
        self.video.tstamp = torch.tensor(tstamps, device="cuda", dtype=torch.float).share_memory_()
        self.video.intrinsics = torch.tensor(intrinsics, device="cuda", dtype=torch.float).share_memory_()

        self.video.fmaps = torch.tensor(fmaps, dtype=torch.half, device="cuda").share_memory_()
        self.video.nets = torch.tensor(nets, dtype=torch.half, device="cuda").share_memory_()
        self.video.inps = torch.tensor(inps, dtype=torch.half, device="cuda").share_memory_()

        self.video.counter = torch.multiprocessing.Value('i', self.video.fmaps.shape[0]-1)

        self.localizer.init_factor_graph_from_video()
