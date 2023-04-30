from pathlib import Path
from itertools import chain
import os
import random

from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import copy
import pandas as pd
import torchvision.utils as vutils

def listdir(dname):
    # 해당 경로 하위의 모든 image파일 경로 
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames

class DefaultDataset(data.Dataset):
    def __init__(self, root,transform=None):
        self.samples=listdir(root)
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img
    def __len__(self):
        return len(self.samples)
    
class CustomCelebADataset(Dataset):
    def __init__(self, root_dir, split, transforms=None):
        self.image_folder = 'img_align_celeba'
        self.root_dir = root_dir

        self.annotation_file = 'list_eval_partition.csv'
        self.transform = transforms
        self.split = split
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[self.split]

        df = pd.read_csv(self.root_dir + self.annotation_file)

        self.filename = df.loc[df['partition']
                               == split_, :].reset_index(drop=True)
        self.length = len(self.filename)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = int(idx)
        img = Image.open(os.path.join(self.root_dir, self.image_folder,
                         self.filename.iloc[idx, ].values[0])).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        target = False
        return img, target

def get_train_loader(root,target_root,batchsize=8,num_workers=4,shuffle=True,size=[64,64],resize=True,gray=False):
    
    if resize:
        transform=transforms.Compose([
            transforms.Resize(size),
            # transforms.RandomCrop(size),
            # transforms.Resize([resize,resize]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],
                                 std=[0.5,0.5,0.5])
            ])
        celeba_transform=transforms.Compose([
            transforms.CenterCrop(160),
            transforms.Resize(size),
            # transforms.RandomCrop(size),
            # transforms.Resize([resize,resize]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],
                                 std=[0.5,0.5,0.5])
            ])
    else:
        transform=transforms.Compose([
            transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],
                                 std=[0.5,0.5,0.5])
            ])
    
    dataset=DefaultDataset(root,transform)
    target_dataset=CustomCelebADataset(target_root,split='train',transforms=celeba_transform)

    loader=data.DataLoader(dataset=dataset,
                           batch_size=batchsize,
                           num_workers=num_workers,
                           shuffle=shuffle
                          )
    target_loader=data.DataLoader(dataset=target_dataset,
                                  batch_size=batchsize,
                                  num_workers=num_workers,
                                  shuffle=shuffle
                                  )
    
    img = next(iter(dataset))
    vutils.save_image(img[:], f'domain_A_img.png', normalize=True)
    target_img, _ = next(iter(target_dataset))
    vutils.save_image(target_img[:], f'domain_B_img.png', normarlize=True)
    
    return loader,target_loader

def get_test_loader(root,batch_size=8,size=[310,650],shuffle=False,resize=False):
    if resize:
        transform=transforms.Compose([
            transforms.RandomCrop(size),
            transforms.Resize([resize,resize]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],
                                 std=[0.5,0.5,0.5])
            ])
    
    else:
        transform=transforms.Compose([
            transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],
                                 std=[0.5,0.5,0.5])
            ])
    dataset=DefaultDataset(root,transform)
    loader=data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle
                           )
    return loader
    

def get_eval_loader(root ,shuffle=False,batch_size=4,num_workers=4,drop_last=False):
    # evaluate dataloader 생성
    print('Preparing DataLoader for the evaluation phase...')
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],
                             std=[0.5,0.5,0.5])
        ])
    dataset=DefaultDataset(root,transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)