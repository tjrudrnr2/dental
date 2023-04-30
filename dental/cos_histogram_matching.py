import os
import glob
import numpy as np
import torch.nn as nn
import torch
import cv2
from PIL import Image
from skimage import exposure
import matplotlib.pyplot as plt
from utils import *
import lpips
R_path=glob.glob("data/processed/R/*.jpg")
V_path=glob.glob("data/processed/V/*.jpg")



for im_path in R_path:
    im=cv2.imread(im_path,cv2.IMREAD_GRAYSCALE)
    hist=cv2.calcHist([im],[0],None,[256],[0,256]).astype('int32')
    ds=0,0
    for target_path in V_path:
        target_im=cv2.imread(target_path,cv2.IMREAD_GRAYSCALE)
        target_hist=cv2.calcHist([target_im],[0],None,[256],[0,256]).astype('int32')
        d=cos_sim(hist.squeeze(),target_hist.squeeze())
        if ds[0]<=d:
            ds=d,target_im
    
    matched=histogram_matching(im,ds[1])
    cv2.imwrite(os.path.join("./generated_images/COS_Histogram_matching",im_path[-10:]),matched)
