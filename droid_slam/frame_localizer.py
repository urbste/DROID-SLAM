import cv2
import torch
import lietorch

from collections import OrderedDict
from droid_net import DroidNet
from droid_slam.factor_graph import FactorGraph

import geom.projective_ops as pops
from modules.corr import CorrBlock

import numpy as np

class ReLocalizer:
    """ This class is used to filter incoming frames and extract features """

    def __init__(self, net, video, args, thresh=8, device="cuda:0"):
        
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update
        
        self.video = video
        self.thresh = thresh
        self.device = device

        self.upsample = args.upsample
        self.beta = args.beta
        self.backend_thresh = args.backend_thresh
        self.backend_radius = args.backend_radius
        self.backend_nms = args.backend_nms

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
    
    def init_factor_graph_from_video(self):
        # initialize the factor graph from the old video
        t = self.video.counter.value
        self.graph = FactorGraph(self.video, self.update, corr_impl="alt", max_factors=16*t, upsample=True)
        self.graph.add_proximity_factors(rad=self.backend_radius, 
                                    nms=self.backend_nms, 
                                    thresh=self.backend_thresh, 
                                    beta=self.beta)
        print("Updated factor graph")

    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image):
        """ context features """
        net, inp = self.cnet(image).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def localize_frame(self, tstamp, image, localization_thresh, depth=None, intrinsics=None):
        
        Id = lietorch.SE3.Identity(1,).data.squeeze()
        ht = image.shape[-2] // 8
        wd = image.shape[-1] // 8

        # normalize images
        inputs = image[None, :, [2,1,0]].to(self.device) / 255.0
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features from video to be localized
        fmap = self.__feature_encoder(inputs)
        net, inp = self.__context_encoder(inputs[:,[0]])

        coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
        # now correlation block on all feature maps in the map
        small_motion_mags = []
        all_motion_mags = []
        indices = []
        for i in range(self.video.fmaps.shape[0]):

            corr = CorrBlock(self.video.fmaps[[i]], fmap[None,[0]])(coords0)
            _, delta, weight1 = self.update(self.video.nets[[i]][None], self.video.inps[[i]][None], corr)

            # check motion magnitue / add new frame to video
            motion_mag = (delta).norm(dim=-1).mean().item()
            # print("Motion magnitude {} at timestamp {}".format(motion_mag, self.video.tstamp[i]*1e-9))
            all_motion_mags.append(motion_mag)
            if motion_mag < localization_thresh:
                small_motion_mags.append(motion_mag)
                indices.append(i)
        
        print("Smallest magnitude {}".format(np.min(all_motion_mags)))
        if len(indices) > 0:
            print("Found {} close frames: ".format(len(indices)))
            sorted_motion_mags = np.argsort(np.array(small_motion_mags))
            sorted_indices = [indices[idx] for idx in sorted_motion_mags]
            print("Closest id: ",sorted_indices[0])
            closest_img = self.video.images[sorted_indices[0]].permute(1, 2, 0).cpu().numpy()
            image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()


            # add the localized frame to the video
            closest_pose = self.video.poses[sorted_indices[0]]
            self.video.append(tstamp, image[0], closest_pose, None, depth, intrinsics / 8.0, fmap, net[0], inp[0])
            print("Added image to map")


            # update pose
            # add detected frame and some neighbors
            current_frame_idx = self.video.counter.value
            self.graph.add_factors([sorted_indices[0]], [current_frame_idx])
            self.graph.update(motion_only=True)

            cv2.imshow("closest image", np.concatenate([image_np, closest_img],0))
            cv2.waitKey(1)
            print("")

            # add spatial proximity factors to factor graph

            # add time add_neighborhood_factors


            return True
        return False
