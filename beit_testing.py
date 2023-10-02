import cv2
import numpy as np
import torch
from data import closest_divisible_by_patch_size
from BeIT.ViT import ViTModel
from BeIT.MaskGenerator import MaskingGenerator
from ModelManager import DVAE
from torchvision.transforms import ToTensor


img = cv2.imread("lg-398211963448579514-aug-gutenberg1939--page-52.png")
width = closest_divisible_by_patch_size(int(np.ceil(img.shape[1] * 0.5)))
height = closest_divisible_by_patch_size(int(np.ceil(img.shape[0] * 0.5)))
img = cv2.resize(img, (width, height))
img = ToTensor()(img).unsqueeze(0)
print(img.size())


model = ViTModel(max_img_size=(3480, 2512), 
                 num_channels=3, 
                 d_model=768, 
                 patch_size=(16,16), 
                 use_masking=True, attention_heads=12, dim_ff=3072, num_enc_layers=12)

vae = DVAE.load_from_checkpoint("DVAE.ckpt", map_location='cpu')

masker = MaskingGenerator(max_patch_prob=0.4)

with torch.no_grad():
    codes = vae.model.get_codebook_indices(img)
    bool_masked_pos = masker(height=img.size(2)//16, width=img.size(3)//16)#torch.randint(low=0, high=2, size=(1, codes.size(1))).bool()
    bool_masked_pos = torch.tensor(bool_masked_pos).bool().flatten().unsqueeze(0)
    output = model(img, bool_masked_pos)[:, 1:, :]
    print(output[bool_masked_pos].size())
    print(codes[bool_masked_pos].size())


