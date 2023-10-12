import torch
from torchinfo import summary
from data import DeepScoresDataset, ds2_collade_func
from torch.utils.data import DataLoader
from transformers import BeitForMaskedImageModeling, BeitConfig, BeitImageProcessor

from vit_pytorch.vit import ViT
from vit_pytorch.mae import MAE

dataset = DeepScoresDataset(patch_size=32)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=ds2_collade_func)
max_height, max_width = dataset.get_max_hw()
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
    masking_ratio=0.75,
    decoder_depth=6
)

sample = next(iter(dataloader))
loss, preds = mae(sample, return_predictions=True)
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