import os
import cv2
import torch
import numpy as np
from rich.progress import track
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

def ds2_collade_func(data):
    max_image_width = max([img.shape[2] for img in data])
    max_image_height = max([img.shape[1] for img in data])

    X_train = torch.ones(size=[len(data), 3, max_image_height, max_image_width], dtype=torch.float32)

    for i, img in enumerate(data):
        _, h, w = img.size()
        X_train[i, :, :h, :w] = img
    
    return X_train

def closest_divisible_by_patch_size(n, patch_size=16):
    lower = (n // patch_size) * patch_size
    higher = lower + patch_size

    if abs(n - lower) < abs(n - higher):
        return lower
    else:
        return higher


class DeepScoresDataset(Dataset):
    def __init__(self, data_path="Data/ds2_complete/images/", reduce_ratio=0.5) -> None:
        self.x = []
        for imgname in track(os.listdir(data_path)):
            self.x.append(os.path.join(data_path, imgname))
        
        self.reduce_ratio = reduce_ratio
        self.tensorTransform = ToTensor()
        super().__init__()
    
    def get_sample(self, index):
        img = cv2.imread(self.x[index])
        width = closest_divisible_by_patch_size(int(np.ceil(img.shape[1] * self.reduce_ratio)))
        height = closest_divisible_by_patch_size(int(np.ceil(img.shape[0] * self.reduce_ratio)))
        img = cv2.resize(img, (width, height))
        return self.tensorTransform(img)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        img = cv2.imread(self.x[index])
        width = closest_divisible_by_patch_size(int(np.ceil(img.shape[1] * self.reduce_ratio)))
        height = closest_divisible_by_patch_size(int(np.ceil(img.shape[0] * self.reduce_ratio)))
        img = cv2.resize(img, (width, height))
        return self.tensorTransform(img)

    def get_max_hw(self):
        m_width = 3508
        m_height = 2480

        return m_height, m_width