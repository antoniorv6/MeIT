import os
import cv2
import torch
import numpy as np
from rich.progress import track
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from BeIT.MaskGenerator import MaskingGenerator
from ModelManager import DVAE

def ds2_collade_func(data):
    max_image_width = max([img.shape[2] for img in data])
    max_image_height = max([img.shape[1] for img in data])

    X_train = torch.ones(size=[len(data), 3, max_image_height, max_image_width], dtype=torch.float32)

    for i, img in enumerate(data):
        _, h, w = img.size()
        X_train[i, :, :h, :w] = img
    
    return X_train

def ds2_masking_collade_func(data):
    images = [sample[0] for sample in data]
    gt = [sample[1] for sample in data]
    masks = [sample[2] for sample in data]
    
    max_image_width = max([img.shape[2] for img in images])
    max_image_height = max([img.shape[1] for img in images])

    X_train = torch.ones(size=[len(images), 3, max_image_height, max_image_width], dtype=torch.float32)

    for i, img in enumerate(images):
        _, h, w = img.size()
        X_train[i, :, :h, :w] = img

    return X_train, masks[0].unsqueeze(0), gt[0]

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


class DeepScoresMasking(Dataset):
    def __init__(self, data_path="Data/ds2_complete/images/", checkpoint_path="weights/DVAE.ckpt", reduce_ratio=0.5) -> None:
        self.x = []
        for imgname in track(os.listdir(data_path)):
            self.x.append(os.path.join(data_path, imgname))
        
        self.reduce_ratio = reduce_ratio
        self.tensorTransform = ToTensor()
        self.mask_generator = MaskingGenerator()
        self.vae = DVAE.load_from_checkpoint(checkpoint_path, map_location="cpu")
        self.vae.eval()
        super().__init__()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        with torch.no_grad():
            img = cv2.imread(self.x[index])
            width = closest_divisible_by_patch_size(int(np.ceil(img.shape[1] * self.reduce_ratio)))
            height = closest_divisible_by_patch_size(int(np.ceil(img.shape[0] * self.reduce_ratio)))
            img = cv2.resize(img, (width, height))
            img = self.tensorTransform(img)

            gt = self.vae.model.get_codebook_indices(img.unsqueeze(0))
            bool_masked_pos = self.mask_generator(height=img.size(1)//16, width=img.size(2)//16) 
            bool_masked_pos = torch.tensor(bool_masked_pos).bool().flatten()

        return img, gt, bool_masked_pos 

    def get_max_hw(self):
        m_width = 3508
        m_height = 2480

        return m_height, m_width