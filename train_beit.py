import torch
from data import DeepScoresMasking, ds2_masking_collade_func
from torch.utils.data import DataLoader
from ModelManager import get_MeIT_model
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import torch.nn as nn

torch.set_float32_matmul_precision('high')

def main():
    dataset = DeepScoresMasking()
    dataloader = DataLoader(dataset, batch_size=1, num_workers=20, collate_fn=ds2_masking_collade_func)
    mheight, mwidth = dataset.get_max_hw()

    model = get_MeIT_model(mheight, mwidth)
    
    wandb_logger = WandbLogger(project='LITIS_STAY', group=f"SSL", name=f"MeIT-S", log_model=False)
    
    checkpointer = ModelCheckpoint(dirpath=f"weights/", filename=f"MeIT-S", every_n_train_steps=100, verbose=True)

    trainer = Trainer(max_epochs=5, logger=wandb_logger, callbacks=[checkpointer])

    trainer.fit(model, train_dataloaders=dataloader)
    

if __name__ == "__main__":
    main()
