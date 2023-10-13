import torch
import cv2
import numpy as np
from torchinfo import summary
from data import DeepScoresDataset, ds2_collade_func
from torch.utils.data import DataLoader
from transformers import BeitForMaskedImageModeling, BeitConfig, BeitImageProcessor
from einops.layers.torch import Rearrange
from einops import rearrange

from vit_pytorch.vit import ViT
from vit_pytorch.mae import MAE

def patches_to_image(tensor, original_height=None, original_width=None, p1=32, p2=32):
    b, num_patches, _ = tensor.size()
    c = 3  # Deduced from tensor shape

    if original_height:
        h = original_height
        w = (num_patches * p2) // (h // p1)
    elif original_width:
        w = original_width
        h = (num_patches * p1) // (w // p2)
    else:
        raise ValueError("Either original_height or original_width must be provided.")

    return tensor.reshape(b, h//p1, w//p2, p1, p2, c).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

dataset = DeepScoresDataset(patch_size=32)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=ds2_collade_func)
max_height, max_width = dataset.get_max_hw()
mean = dataset.processor.image_mean
std = dataset.processor.image_std

print(max_height, max_width)

model = ViT(
    image_size = (max_height, max_width),
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048
)

mae = MAE(
    encoder=model,
    decoder_dim=512,
    masking_ratio=0.4,
    decoder_depth=6
)

rearr = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 32, p2 = 32)
sample = next(iter(dataloader))
#original_img = sample.clone()
loss, preds, mask = mae(sample, return_predictions=True)

patched_image = rearr(sample).squeeze()
mask = mask.squeeze()
preds = preds.squeeze()
unmasked_image = patched_image.clone()

for idx, prediction in enumerate(preds):
    unmasked_image[mask[idx], :] = prediction

img = patches_to_image(unmasked_image.unsqueeze(0), original_height=max_height)

img = img.permute(0,2,3,1).squeeze().cpu().detach().numpy()

img_denormalized = (img * std / 2 + 0.5) * 255
img_denormalized = img_denormalized[:, :, ::-1]
img_uint8 = img_denormalized.astype(np.uint8)
cv2.imwrite("test.png", img_uint8)

patches = unmasked_image.reshape(1, 962, 3, 32, 32).cpu().detach().numpy()
#img = np.zeros((max_height, max_width, 3))
#rows = max_height // 32
#cols = max_width // 32
#
#for i in range(rows):
#    for j in range(cols):
#        img[i*32:(i+1)*32, j*32:(j+1)*32, :] = patches[0, i*cols + j].transpose(1, 2, 0)

#for i in range(len(mask)):
#        unmasked_image[mask[i], :] = preds[i, :]
#
#patches = unmasked_image.reshape(1, 962, 3, 32, 32).cpu().detach().numpy()
#
#img = np.zeros((max_height, max_width, 3))
#
#rows = max_height // 32
#cols = max_width // 32
#
#for i in range(rows):
#    for j in range(cols):
#        img[i*32:(i+1)*32, j*32:(j+1)*32, :] = patches[0, i*cols + j].transpose(1, 2, 0)

#img_denormalized = (img * std / 2 + 0.5) * 255
#img_denormalized = img_denormalized[:, :, ::-1]
#img_rescaled = (img_denormalized * 255).clip(0, 255)
#img_uint8 = img_denormalized.astype(np.uint8)
#cv2.imwrite("test.png", img_uint8)
#print(unmasked_image.size())
#for i in range()

#print(preds.size())
#print(mask.size())


#print(loss)
#print(preds[:, 1:, :].reshape(1,3,1184,832).size())

#summary(mae, input_size=[(1,3, max_height, max_width)])
#dataset = DeepScoresMasking()
#dataloader = DataLoader(dataset, batch_size=1, num_workers=20, collate_fn=ds2_masking_collade_func)
#
#image_processor = BeitImageProcessor(do_resize=False, do_center_crop=False)
#custom_config = BeitConfig(use_mask_token=True, image_size=(1184, 848))
#model = BeitForMaskedImageModeling(custom_config)#.from_pretrained("microsoft/beit-base-patch16-224-pt22k", config=custom_config)
#print(summary(model, input_size=[(1,3,1184, 848)], dtypes=[torch.float]))
#
#for (sample, mask, gt) in dataloader:
#    print(sample.size())
#    print(mask.size())
#    print(gt.size())
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    print(model(sample.to(device), bool_masked_pos=mask.to(device), labels=gt[mask].to(device)))
#    import sys
#    sys.exit()