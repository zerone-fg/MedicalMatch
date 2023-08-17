from dataset.transform import random_rot_flip, random_rotate, blur, obtain_cutmix_box

from copy import deepcopy
import h5py
import math
import numpy as np
import os
from PIL import Image
import random
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
import albumentations as A

def norm01(x):
    return np.clip(x, 0, 255) / 255

class ISICDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, portion=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.portion = portion
        self.sample_list = []

        if mode == "train_l" or mode == 'train_u':
            with open(id_path + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

            num_sample = len(self.sample_list)
            assert portion <= num_sample
            if mode == "train_l":
                self.sample_list = self.sample_list[:portion]
            else:
                self.sample_list = self.sample_list[:]

        elif mode == "val":
            with open(id_path + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        p = 0.5
        self.strong_op = A.Compose([
            A.ColorJitter(0.5, 0.5, 0.5, 0.25)
        ])

        self.weak_op = A.Compose([
            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p),
            A.ShiftScaleRotate(p=p),
            A.GaussNoise(p=p)
        ])

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.root, "Image", self.sample_list[item])).resize((224, 224))
        label = Image.open(os.path.join(self.root, "Label", self.sample_list[item])).convert('L').resize(
                (224, 224))

        image = np.array(image)  ### (224, 224, 3)
        label = np.array(label)
        label = np.where(label == 255, 1, 0)

        if self.mode == 'val':
            return torch.from_numpy(norm01(image)).float().permute(2, 0, 1), torch.from_numpy(label).long()

        ###### 进行0.5的弱增强 #####
        tsf = self.weak_op(image=image.astype('uint8'), mask=label)
        img, label = tsf['image'], tsf["mask"]

        x, y, _ = img.shape

        if self.mode == 'train_l':
            return torch.from_numpy(norm01(img)).float().permute(2, 0, 1), torch.from_numpy(label).long()

        img_s1, img_s2 = img.copy(), img.copy()

        tsf_1 = self.strong_op(image=img_s1.astype('uint8'), mask=label)
        img_s1 = torch.from_numpy(norm01(tsf_1["image"])).float().permute(2, 0, 1)

        cutmix_box1 = obtain_cutmix_box(self.size, p=0.5)

        tsf_2 = self.strong_op(image=img_s2.astype('uint8'), mask=label)
        img_s2 = torch.from_numpy(norm01(tsf_2["image"])).float().permute(2, 0, 1)

        cutmix_box2 = obtain_cutmix_box(self.size, p=0.5)

        img = torch.from_numpy(norm01(img)).float().permute(2, 0, 1)

        return img, img_s1, img_s2, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.sample_list)
