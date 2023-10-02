import torch
from data import DeepScoresMasking, ds2_masking_collade_func
from torch.utils.data import DataLoader
from ModelManager import get_MeIT_model
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

torch.set_float32_matmul_precision('high')

def main():
    dataset = DeepScoresMasking()
    dataloader = DataLoader(dataset, batch_size=1, num_workers=20, collate_fn=ds2_masking_collade_func)
    mheight, mwidth = dataset.get_max_hw()

    model = get_MeIT_model(mheight, mwidth)

    x, mask, gt = next(iter(dataloader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = model(x.to(device), patch_mask=mask)
    print(out.size())



    

if __name__ == "__main__":
    main()
