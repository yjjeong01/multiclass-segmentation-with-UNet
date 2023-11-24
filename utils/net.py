import glob
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from . import mytransforms


def dataset(path):
    f_list = glob.glob(path)
    img_list = []
    mask_list = []
    mask_num = 2
    for i in f_list:
        dinfo = torchvision.datasets.ImageFolder(root=i)
        data_num = int(len(dinfo.targets) / mask_num)
        img_list += dinfo.samples[:data_num]
        mask_list += dinfo.samples[data_num:]

    path_mtx = img_list + mask_list

    return path_mtx


class dataload(Dataset):
    def __init__(self, H, W, data_path, aug=True):
        self.H = H
        self.W = W
        self.aug = aug
        self.mask_num = 2
        self.data_num = len(data_path) // self.mask_num
        self.path_mtx = np.array(data_path)[:, :1].reshape(self.mask_num, self.data_num)
        self.pixel_colors = [113, 132, 154, 175, 184, 188, 192, 167, 143, 102, 182, 174, 136, 58, 131, 191, 145, 119, 110]
        self.num_classes = len(self.pixel_colors)

        self.mask_trans = transforms.Compose([transforms.Resize((self.H, self.W)),
                                              transforms.Grayscale(),
                                              mytransforms.Affine(0,
                                                                  translate=[0, 0],
                                                                  scale=1,
                                                                  fillcolor=0),
                                              transforms.ToTensor()])
        self.col_trans = transforms.Compose([transforms.ColorJitter(brightness=random.random())])

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        imgs = torch.zeros(self.num_classes + 1, self.H, self.W, dtype=torch.float)

        if self.aug:
            self.mask_trans.transforms[2].degrees = random.randrange(-25, 25)
            self.mask_trans.transforms[2].translate = [random.uniform(0, 0.05), random.uniform(0, 0.05)]
            self.mask_trans.transforms[2].scale = random.uniform(0.9, 1.1)

        for k in range(self.mask_num):
            if k == 0:
                X = Image.open(self.path_mtx[k, idx])
                if self.aug:
                    X = self.col_trans(X)
                imgs[k] = self.mask_trans(X)

            elif k == 1:
                X = cv2.imread(self.path_mtx[k, idx])
                X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)

                for i in range(self.num_classes):
                    tmp = X.copy()
                    tmp[tmp != self.pixel_colors[i]] = 0
                    tmp = Image.fromarray(tmp)
                    imgs[i + 1] = self.mask_trans(tmp)

        img, mask = imgs[0:1], imgs[1:]

        return [img, mask]


# UNET parts
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


## Original
class UNet(nn.Module):
    def __init__(self, n_channels, num_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.num_classes = num_classes
        factor = 2

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor)
        self.up2 = Up(512, 256 // factor)
        self.up3 = Up(256, 128 // factor)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
