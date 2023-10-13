import os
from sklearn.model_selection import train_test_split

files = os.listdir("Data/ds2_complete/images")

train_files, test_files = train_test_split(files, test_size=0.01)

with open("train.txt", "w+") as trfile:
    trfile.write("\n".join(train_files))
with open("test.txt", "w+") as trfile:
    trfile.write("\n".join(test_files))
