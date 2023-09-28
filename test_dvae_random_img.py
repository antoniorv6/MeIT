import torch
import cv2
import numpy as np
from ModelManager import DVAE
from data import DeepScoresDataset
from torchvision.transforms import ToPILImage, Grayscale


dataset = DeepScoresDataset()
model = DVAE.load_from_checkpoint("weights/DVAE.ckpt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img = dataset.get_sample(150).unsqueeze(0).to(device)
with torch.no_grad():
    codes = model.model.get_codebook_indices(img)
    hard_recons = model.model.decode(codes, img.size(2)//16, img.size(3)//16)
    orgimg = Grayscale()(ToPILImage()(img.squeeze(0)))
    pilimg = Grayscale()(ToPILImage()(hard_recons.squeeze(0)))
    orgimg = np.array(orgimg)
    hrec = np.array(pilimg)
    cv2.imwrite("original.png",orgimg)
    cv2.imwrite("hrec.png",hrec)




