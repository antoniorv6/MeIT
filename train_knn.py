import os
import cv2
import numpy as np
import torch

from rich.progress import track
from ModelManager import MiT_MAE
from sklearn.neighbors import KNeighborsClassifier
from transformers import BeitImageProcessor
from data import closest_divisible_by_patch_size
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

from joblib import dump, load

y_train = []
x_train = []

y_test = []
x_test = []

dataset = "grandstaff_dataset"
path_ds = "grandstaff_dataset/partitions_fpgrandstaff"

transform = BeitImageProcessor(do_resize=False, do_center_crop=False)
model = MiT_MAE.load_from_checkpoint("weights/MiT_MAE_B-epoch=6-step=1700000-val_loss=0.06149.ckpt")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with torch.no_grad():
    with open(f"Data/{path_ds}/excerpts/fold_0/train.txt") as trainfile:
        lines = trainfile.readlines()
        for line in track(lines):
            imgpath = os.path.join(f"Data/{dataset}/", ".".join(line.split('.')[:-1])+".png")
            img = cv2.imread(imgpath)
            y_train.append(imgpath.split("/")[3])
            width = closest_divisible_by_patch_size(int(np.ceil(2100 * 0.4)), patch_size=32)
            height = closest_divisible_by_patch_size(int(np.ceil(2970 * 0.4)), patch_size=32)
            img = cv2.resize(img, (width, height))
            img = transform(img, return_tensors="pt").pixel_values
            emb = model(img.to(device))
            emb = torch.mean(emb, dim=1).squeeze()
            x_train.append(emb.cpu().detach().numpy())
    
    labelencoder = LabelEncoder().fit(y_train)
    y = labelencoder.transform(y_train)
    kneighbors = KNeighborsClassifier(n_neighbors=1).fit(x_train, y)

with open(f"Data/{path_ds}/excerpts/fold_0/val.txt") as trainfile:
    lines = trainfile.readlines()
    for line in track(lines):
        imgpath = os.path.join(f"Data/{dataset}/", ".".join(line.split('.')[:-1])+".png")
        img = cv2.imread(imgpath)
        y_test.append(imgpath.split("/")[3])
        width = closest_divisible_by_patch_size(int(np.ceil(2100 * 0.4)), patch_size=32)
        height = closest_divisible_by_patch_size(int(np.ceil(2970 * 0.4)), patch_size=32)
        img = cv2.resize(img, (width, height))
        img = transform(img, return_tensors="pt").pixel_values
        emb = model(img.to(device))
        emb = torch.mean(emb, dim=1).squeeze()
        x_test.append(emb.cpu().detach().numpy())
    
predictions = kneighbors.predict(x_test)
print(accuracy_score(predictions, labelencoder.transform(y_test)) * 100)
print(f1_score(predictions, labelencoder.transform(y_test), average='macro') * 100)



        
    