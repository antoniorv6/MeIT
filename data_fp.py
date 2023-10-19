import re
import cv2
import gin
import torch
import numpy as np
import cv2

from data_augmentation.data_augmentation import augment, convert_img_to_tensor
from utils import check_and_retrieveVocabulary
from rich import progress
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BeitImageProcessor

def closest_divisible_by_patch_size(n, patch_size=16):
    lower = (n // patch_size) * patch_size
    higher = lower + patch_size

    if abs(n - lower) < abs(n - higher):
        return lower
    else:
        return higher

def load_set(path, base_folder="string_quartets_dataset", fileformat="png", krn_type="bekrn", reduce_ratio=0.4):
    x = []
    y = []
    with open(path) as datafile:
        lines = datafile.readlines()
        for line in progress.track(lines):
            excerpt = line.replace("\n", "")
            try:
                with open(f"Data/{base_folder}/{'.'.join(excerpt.split('.')[:-1])}.{krn_type}") as krnfile:
                    krn_content = krnfile.read()
                    fname = ".".join(excerpt.split('.')[:-1])
                    img = cv2.imread(f"Data/{base_folder}/{fname}.{fileformat}")
                    width = closest_divisible_by_patch_size(int(np.ceil(2100 * reduce_ratio)), patch_size=32)
                    height = closest_divisible_by_patch_size(int(np.ceil(2970 * reduce_ratio)), patch_size=32)
                    img = cv2.resize(img, (width, height))
                    y.append([content + '\n' for content in krn_content.split("\n")])
                    x.append(img)
                    
            except Exception:
                print(f'Error reading Data/GrandStaff/{excerpt}')

    return x, y

def batch_preparation_img2seq(data):
    images = [sample[0] for sample in data]
    dec_in = [sample[1] for sample in data]
    gt = [sample[2] for sample in data]

    X_train = torch.cat(images, dim=0)
    
    max_length_seq = max([len(w) for w in gt])

    decoder_input = torch.zeros(size=[len(dec_in),max_length_seq])
    y = torch.zeros(size=[len(gt),max_length_seq])

    for i, seq in enumerate(dec_in):
        decoder_input[i, 0:len(seq)-1] = torch.from_numpy(np.asarray([char for char in seq[:-1]]))
    
    for i, seq in enumerate(gt):
        y[i, 0:len(seq)-1] = torch.from_numpy(np.asarray([char for char in seq[1:]]))
    
    return X_train, decoder_input.long(), y.long()

class OMRIMG2SEQDataset(Dataset):
    def __init__(self, augment=False) -> None:
        self.teacher_forcing_error_rate = 0.2
        self.x = None
        self.y = None
        self.augment = augment

        super().__init__()
    
    def apply_teacher_forcing(self, sequence):
        errored_sequence = sequence.clone()
        for token in range(1, len(sequence)):
            if np.random.rand() < self.teacher_forcing_error_rate and sequence[token] != self.padding_token:
                errored_sequence[token] = np.random.randint(0, len(self.w2i))
        
        return errored_sequence

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.augment:
            x = augment(self.x[index])
        else:
            x = convert_img_to_tensor(self.x[index])
        
        y = torch.from_numpy(np.asarray([self.w2i[token] for token in self.y[index]]))
        decoder_input = self.apply_teacher_forcing(y)
        return x, decoder_input, y

    def get_max_hw(self):
        m_width = np.max([img.shape[1] for img in self.x])
        m_height = np.max([img.shape[0] for img in self.x])

        return m_height, m_width
    
    def get_max_seqlen(self):
        return np.max([len(seq) for seq in self.y])

    def vocab_size(self):
        return len(self.w2i)

    def get_gt(self):
        return self.y
    
    def set_dictionaries(self, w2i, i2w):
        self.w2i = w2i
        self.i2w = i2w
        self.padding_token = w2i['<pad>']
    
    def get_dictionaries(self):
        return self.w2i, self.i2w
    
    def get_i2w(self):
        return self.i2w


class GrandStaffDataset(OMRIMG2SEQDataset):
    def __init__(self, data_path, augment=False, num_systems_gen=1, size=None) -> None:
        self.augment = augment
        self.teacher_forcing_error_rate = 0.2
        self.x, self.y = load_set(data_path)
        self.y = self.preprocess_gt(self.y)
        self.tensorTransform = transforms.ToTensor()
        self.num_sys_gen = num_systems_gen
        self.processor = BeitImageProcessor(do_resize=False, do_center_crop=False)
        if size == None:
            self.size = len(self.x)
        else:
            self.size = size
    
    def erase_numbers_in_tokens_with_equal(self, tokens):
        return [re.sub(r'(?<=\=)\d+', '', token) for token in tokens]

    def generate_system(self, num_sys_gen=1, padding=0):
        random_indices = np.random.randint(0, len(self.x)-1, size=num_sys_gen)
        images = [self.x[index] for index in random_indices]
        gts = [self.y[index] for index in random_indices]
        image_height = sum([img.shape[0] for img in images]) + (padding * num_sys_gen) #n-pixel padding between systems
        image_width = max([img.shape[1] for img in images])
        complete_image = np.full(shape=(image_height, image_width, 3), fill_value=255, dtype=np.uint8)
        current_index = 0
        for sample in images[:num_sys_gen]:
            complete_image[current_index:current_index+sample.shape[0], :sample.shape[1], :] = sample
            current_index += sample.shape[0] + padding

        sequence = gts[0][:-5]
        for seq in gts[1:-1]:
            sequence += seq[5:-5]
        sequence += gts[-1][5:]
        
        cv2.imwrite("test.png", complete_image)

        return complete_image, sequence

    def __getitem__(self, index):
        x = self.processor(self.x[index], return_tensors="pt").pixel_values
        y = torch.from_numpy(np.asarray([self.w2i[token] for token in self.y[index]]))
        decoder_input = self.apply_teacher_forcing(y)
        return x, decoder_input, y
    
    def __len__(self):
        return self.size
    
    def preprocess_gt(self, Y):
        for idx, krn in enumerate(Y):
            krnlines = []
            krn = "".join(krn)
            krn = krn.replace(" ", " <s> ")
            krn = krn.replace("·", "")
            lines = krn.split("\n")
            for line in lines:
                line = line.replace("\t", " <t> ")
                line = line.split(" ")
                if len(line) > 1:
                    line.append("<b>")
                    krnlines.append(line)
                    
            Y[idx] = self.erase_numbers_in_tokens_with_equal(['<bos>'] + sum(krnlines, []) + ['<eos>'])
        return Y

@gin.configurable
def load_data(data_path, vocab_name, val_path=None):
    if val_path == None:
        val_path = data_path
    train_dataset = GrandStaffDataset(data_path=f"{data_path}/train.txt", augment=True)
    val_dataset = GrandStaffDataset(data_path=f"{val_path}/val.txt", size=1000)
    test_dataset = GrandStaffDataset(data_path=f"{data_path}/test.txt")

    w2i, i2w = check_and_retrieveVocabulary([train_dataset.get_gt(), val_dataset.get_gt(), test_dataset.get_gt()], "vocab/", f"{vocab_name}")

    train_dataset.set_dictionaries(w2i, i2w)
    val_dataset.set_dictionaries(w2i, i2w)
    test_dataset.set_dictionaries(w2i, i2w)

    return train_dataset, val_dataset, test_dataset 



     


