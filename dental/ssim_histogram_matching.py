from pathlib import Path
from itertools import chain
from torch.utils import data
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
import cv2
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


for im_path in V_path:
    a=0
    im=cv2.imread(im_path,cv2.IMREAD_GRAYSCALE)
    ds=1,0
    for target_path in R_path:
        target_im=cv2.resize(cv2.imread(target_path,cv2.IMREAD_GRAYSCALE),(256,256))
        d=structural_similarity(cv2.resize(im,(256,256)),target_im)
        
        if ds[0]>=d:
            ds=d,target_path
        
    matched=histogram_matching(im,cv2.imread(ds[1],cv2.IMREAD_GRAYSCALE))
    cv2.imwrite(os.path.join("./generated_images/SSIM_Histogram_matching",im_path[-10:]),matched)