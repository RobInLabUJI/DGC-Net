import numpy as np
import cv2
import os
from os import path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from model.net import DGCNet

class DeNormalize:
    '''
    Removes normalization using the mean, std specified
    '''
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class DGCVision:
    def __init__(self):
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        self.restore_image = DeNormalize(mean, std)
        self.dataset_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.use_cuda = torch.cuda.is_available()
        device = torch.device('cuda:0' if self.use_cuda else 'cpu')

        self.net = DGCNet(mask=True)
        self.net.load_state_dict(torch.load(osp.join('pretrained_models',
                                                'dgcm',
                                                'checkpoint.pth'),
                                       map_location=torch.device(device))['state_dict'])
        self.net.eval()
        self.net.to(device);
        self.IMG_SIZE = (240, 240)
        
    def predict(self, img1, img2):
        self.inputShape = img1.shape
        self.input1 = self.dataset_transforms(cv2.resize(img1, self.IMG_SIZE)).unsqueeze(0)
        input2 = self.dataset_transforms(cv2.resize(img2, self.IMG_SIZE)).unsqueeze(0)
        if self.use_cuda:
            self.input1 = self.input1.cuda()
            input2 = input2.cuda()
        with torch.no_grad():
            self.estimates_grid_pyr, self.match_mask = self.net(self.input1, input2)
            
    def warp(self):
        (h, w, _) = self.inputShape
        warp_img = F.grid_sample(self.input1, self.estimates_grid_pyr[-1].permute(0, 2, 3, 1))
        warp_img = self.restore_image(warp_img.squeeze()).permute(1, 2, 0).cpu().numpy()
        return cv2.resize(warp_img, (w,h))
    
    def getMatchabilityMask(self):
        return self.match_mask.permute(0, 2, 3, 1).cpu().numpy()[0]
        
    def getFlow(self):
        self.correspondence_map = self.estimates_grid_pyr[-1].permute(0, 2, 3, 1).cpu().numpy()[0]
        size = self.IMG_SIZE[0]
        flow = np.zeros((size, size, 2,))
        for i in range(size):
            for j in range(size):
                ws, hs = j * 2.0 / size - 1, i * 2.0 / size - 1
                flow[i,j] = np.array([ws,hs]) - self.correspondence_map[i,j]
        return flow
        
    def target(self, x, y):
        size = self.IMG_SIZE[0]
        xs = x * size / self.inputShape[1]
        ys = y * size / self.inputShape[0]
        ws, hs = xs * 2.0 / size - 1, ys * 2.0 / size - 1
        d = np.zeros((size,size,2))
        d[:,:,0] = self.correspondence_map[:,:,0] - ws
        d[:,:,1] = self.correspondence_map[:,:,1] - hs
        m = np.linalg.norm(d,axis=2)
        yt, xt = np.unravel_index(np.argmin(m, axis=None), m.shape)
        xt = xt * self.inputShape[1] / size
        yt = yt * self.inputShape[0] / size
        return (xt, yt)

             
