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
# from keras.utils import to_categorical


class ACDCDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('splits/%s/valtest.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):

        id = self.ids[item]
        sample = h5py.File(os.path.join(self.root, id), 'r')
        img = sample['image'][:]
        mask = sample['label'][:]

        if self.mode == 'val':
            return torch.from_numpy(img).float(), torch.from_numpy(mask).long()

        # if random.random() > 0.5:
        #     img, mask = random_rot_flip(img, mask)
        # elif random.random() > 0.5:
        #     img, mask = random_rotate(img, mask)
        x, y = img.shape
        img = zoom(img, (self.size / x, self.size / y), order=0)
        mask = zoom(mask, (self.size / x, self.size / y), order=0)  ### (224, 224)

        # mask_down = zoom(mask, (56 / 224, 56 / 224), order=0)
        #
        # label_one_hot = F.one_hot(torch.tensor(mask_down).to(torch.int64), num_classes=4)  ##(h, w, cls)
        # # label_one_hot = to_categorical(mask_down, 4)
        # label_one_hot = label_one_hot.reshape(-1, 4)
        #
        # gt_cos = label_one_hot @ label_one_hot.transpose(-1, -2)  ### (h*w, h*w)

        # if self.mode == 'train_l':
        return torch.from_numpy(img).unsqueeze(0).float(), torch.from_numpy(np.array(mask)).long()

        # img = Image.fromarray((img * 255).astype(np.uint8))
        # img_s1, img_s2 = deepcopy(img), deepcopy(img)
        # img = torch.from_numpy(np.array(img)).unsqueeze(0).float() / 255.0
        #
        # if random.random() < 0.8:
        #     img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        # img_s1 = blur(img_s1, p=0.5)
        # cutmix_box1 = obtain_cutmix_box(self.size, p=0.5)
        # img_s1 = torch.from_numpy(np.array(img_s1)).unsqueeze(0).float() / 255.0
        #
        # if random.random() < 0.8:
        #     img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        # img_s2 = blur(img_s2, p=0.5)
        # cutmix_box2 = obtain_cutmix_box(self.size, p=0.5)
        # img_s2 = torch.from_numpy(np.array(img_s2)).unsqueeze(0).float() / 255.0
        #
        # # ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))
        #
        # return img, img_s1, img_s2, cutmix_box1, cutmix_box2, mask

    def __len__(self):
        return len(self.ids)
