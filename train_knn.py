import os
import cv2
import numpy as np
import torch

from rich.progress import track
from ModelManager import MeITHuggingFace
from sklearn.neighbors import KNeighborsClassifier
from transformers import BeitImageProcessor
from data import closest_divisible_by_patch_size
from sklearn.preprocessing import LabelEncoder

from joblib import dump, load

y_train = []
x_train = []

y_test = []
x_test = []

transform = BeitImageProcessor(do_resize=False, do_center_crop=False)
model = MeITHuggingFace.load_from_checkpoint("MeIT-B_16.ckpt").model.beit
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with torch.no_grad():
    with open("Data/grandstaff_dataset/partitions_fpgrandstaff/excerpts/fold_0/train.txt") as trainfile:
        lines = trainfile.readlines()
        for line in track(lines):
            imgpath = os.path.join("Data/grandstaff_dataset/", ".".join(line.split('.')[:-1])+".png")
            img = cv2.imread(imgpath)
            y_train.append(imgpath.split("/")[3])
            width = closest_divisible_by_patch_size(int(np.ceil(2100 * 0.4)), patch_size=16)
            height = closest_divisible_by_patch_size(int(np.ceil(2970 * 0.4)), patch_size=16)
            img = cv2.resize(img, (width, height))
            img = transform(img, return_tensors="pt").pixel_values
            emb = model(img.to(device)).last_hidden_state
            emb = torch.mean(emb[:, 1:, :], dim=1).squeeze()
            x_train.append(emb.cpu().detach().numpy())
    
    labelencoder = LabelEncoder().fit(y_train)
    y = labelencoder.transform(y_train)
    kneighbors = KNeighborsClassifier(n_neighbors=1).fit(x_train, y)
    dump(labelencoder, 'labels_meit.joblib')
    dump(kneighbors, 'knn_meit.joblib')



        
    