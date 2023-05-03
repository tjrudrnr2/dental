import numpy as np
import cv2
import torch
import os
import glob
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim 
from skimage import exposure
import lpips
from random import randrange
from PIL import Image
from models.base_model import BaseGenerator
import torch
from datasets.dataloader import *
import torchvision.transforms as transforms
from metric.fid import *
import shutil

def generate_test_sample(model,dataloader,save_path,device=torch.device('cpu')):
    torch_to_image = transforms.Compose([
        transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),  # [-1, 1] to [0, 1]
        transforms.ToPILImage()
    ])
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    k=0
    for i,batch in enumerate(dataloader):
        if k==5000:
            break
        for im in batch:
            sample=model(im.unsqueeze(0).to(device))
            sample=torch_to_image(sample[0])
            sample.save(os.path.join(save_path,(str(k)+'.jpg')))
            k+=1

def calculate_fid(model,data_loader,save_path,base_path ,img_size=256,batch_size=50,device=torch.device('cpu')):
    # after generate sample and calculate fid
    
    generate_test_sample(model,data_loader,save_path,device=device)
    fid=calculate_fid_given_paths(base_path,save_path,img_size,batch_size)
    shutil.rmtree(save_path)
    return fid

def autocrop(pil_img, pct_focus=0.3, matrix_HW_pct=0.3, sample=1):
    """
    random crop from an input image
    Args:
        - pil_img
        - pct_focus(float): PCT of margins to remove based on image H/W
        - matrix_HW_pct(float): crop size in PCT based on image Height
        - sample(int): number of random crops to return
    returns:
        - crop_list(list): list of PIL cropped images
    """
    x, y = pil_img.size
    img_focus = pil_img.crop((x*pct_focus, y*pct_focus, x*(1-pct_focus), y*(1-pct_focus)))
    x_focus, y_focus = img_focus.size
    matrix = round(matrix_HW_pct*y_focus)
    crop_list = []
    for i in range(sample):
        x1 = randrange(0, x_focus - matrix)
        y1 = randrange(0, y_focus - matrix)
        cropped_img = img_focus.crop((x1, y1, x1 + matrix, y1 + matrix))
        #display(cropped_img)
        crop_list.append(cropped_img)
    return crop_list

def get_histogram(src):
    # return histogram (256,1)
    return cv2.calcHist([src],[0],None,[256],[0,256]).astype('int32').squeeze()



def get_ssim(src,tar):
    # TODO
    return ssim(src,tar)

def get_lpips():
    # TODO
    return 

def cosin_metric(x1, x2):
  return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

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
            
