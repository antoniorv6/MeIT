import os
import cv2
import numpy as np
import torch

from rich.progress import track
from sklearn.neighbors import KNeighborsClassifier
from data import closest_divisible_by_patch_size
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

from joblib import dump, load

y_train = []
x_train = []

y_test = []
x_test = []

with open("Data/grandstaff_dataset/partitions_fpgrandstaff/excerpts/fold_0/train.txt") as trainfile:
    lines = trainfile.readlines()
    for line in track(lines):
        imgpath = os.path.join("Data/grandstaff_dataset/", ".".join(line.split('.')[:-1])+".png")
        img = cv2.imread(imgpath)
        y_train.append(imgpath.split("/")[3])
        width = closest_divisible_by_patch_size(int(np.ceil(2100 * 0.4)), patch_size=16)
        height = closest_divisible_by_patch_size(int(np.ceil(2970 * 0.4)), patch_size=16)
        img = cv2.resize(img, (width, height))
        img = img.flatten()
        x_train.append(img)

labelencoder = LabelEncoder().fit(y_train)
y = labelencoder.transform(y_train)
kneighbors = KNeighborsClassifier(n_neighbors=1).fit(x_train, y)

with open("Data/grandstaff_dataset/partitions_fpgrandstaff/excerpts/fold_0/val.txt") as trainfile:
    lines = trainfile.readlines()
    for line in track(lines):
        imgpath = os.path.join("Data/grandstaff_dataset/", ".".join(line.split('.')[:-1])+".png")
        img = cv2.imread(imgpath)
        y_test.append(imgpath.split("/")[3])
        width = closest_divisible_by_patch_size(int(np.ceil(2100 * 0.4)), patch_size=16)
        height = closest_divisible_by_patch_size(int(np.ceil(2970 * 0.4)), patch_size=16)
        img = cv2.resize(img, (width, height))
        img = img.flatten()
        x_test.append(img)

predictions = kneighbors.predict(x_test)
print(accuracy_score(predictions, labelencoder.transform(y_test)) * 100)
print(f1_score(predictions, labelencoder.transform(y_test), average='macro') * 100)





        
    