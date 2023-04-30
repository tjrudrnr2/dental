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


from pathlib import Path
from itertools import chain
from torch.utils import data
from torchvision import transforms


def listdir(dname):
    # 해당 경로 하위의 모든 image파일 경로 
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames

class DefaultDataset(data.Dataset):
    def __init__(self, root, resize=256):
        self.samples=listdir(root)
        self.targets = None
        self.resize=resize
        if self.resize:
            self.transform=transforms.Compose([
                transforms.Resize([self.resize,self.resize]),
                transforms.ToTensor(),
                ])

    def __getitem__(self, index):
        
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img,index
    
    def __len__(self):

        return len(self.samples)

def get_train_loader(root,target_root,batchsize=32,num_workers=16,shuffle=False):
    
    dataset=DefaultDataset(root)
    target_dataset=DefaultDataset(target_root)

    loader=data.DataLoader(dataset=dataset,
                           batch_size=1,
                           num_workers=num_workers,
                           shuffle=shuffle
                          )
    target_loader=data.DataLoader(dataset=target_dataset,
                                  batch_size=batchsize,
                                  num_workers=num_workers,
                                  shuffle=shuffle
                                  )
    
    return loader,target_loader

loader,target_loader=get_train_loader('./data/processed/V','./data/processed/R')
loss_fn_alex=lpips.LPIPS(net='vgg')
GPU_NUM = 5
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU

loss_fn_alex.cuda()
a=0
print(len(loader))
loader_path=list(map(lambda x:os.path.basename(x),np.array(loader.dataset.samples)))
target_path=list(map(lambda x:os.path.basename(x),np.array(target_loader.dataset.samples)))
for img,index in loader:
    ds=100,0
    img_path=loader_path[index]
    
    for target in target_loader:
        d=loss_fn_alex(img.cuda(),target[0].cuda()).squeeze()
        for i in range(len(d)):
            if d[i]<=ds[0]:
                ds=d[i],target_path[target[1][i]]
    matched=histogram_matching(cv2.imread(os.path.join('./data/processed/V',img_path),cv2.IMREAD_GRAYSCALE),
                               cv2.imread(os.path.join('./data/processed/R',ds[1]),cv2.IMREAD_GRAYSCALE))

    
    
    cv2.imwrite(os.path.join("./generated_images/LPIPS_Histogram_matching",img_path),matched)
