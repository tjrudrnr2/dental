import numpy as np
import cv2
import torch
import os
import glob
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim 
from skimage import exposure
import lpips


def get_histogram(src):
    # return histogram (256,1)
    return cv2.calcHist([src],[0],None,[256],[0,256]).astype('int32').squeeze()

def cos_sim(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def get_ssim(src,tar):
    # TODO
    return ssim(src,tar)

def get_lpips():
    # TODO
    return 


def histogram_matching(src,ref):
    matched=exposure.match_histograms(src,ref)
    return matched

def Histogram_equalization(im):
    # must input grayscale
    return cv2.equalizeHist(im)


def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)