import fire

import torch
from data import DeepScoresDataset, ds2_collade_func
from torch.utils.data import DataLoader
from ModelManager import get_MAE_VIT_model
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

torch.set_float32_matmul_precision('high')

def main(model_name, patch_size):
    train_dataset = DeepScoresDataset(patch_size=patch_size , samples_file="train.txt")
    val_dataset = DeepScoresDataset(patch_size=patch_size , samples_file="test.txt")

    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=20, collate_fn=ds2_collade_func)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=20, collate_fn=ds2_collade_func)
    
    mheight, mwidth = train_dataset.get_max_hw()

    model = get_MAE_VIT_model(mheight, mwidth, patch_size)

    wandb_logger = WandbLogger(project='LITIS_STAY', group=f"SSL", name=f"{model_name}", log_model=False)
    
    bas_name = '{epoch}-{step}-{val_loss:.5f}'
    checkpointer = ModelCheckpoint(dirpath=f"weights/", filename=f"{model_name}-{bas_name}", every_n_train_steps=10000, save_top_k=-1, verbose=True)

    trainer = Trainer(max_epochs=10, val_check_interval=10000, check_val_every_n_epoch=None, logger=wandb_logger, callbacks=[checkpointer])

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


def launch(model_name, patch_size, checkpoint_path=None):
    main(model_name, patch_size=patch_size)

if __name__ == "__main__":
    fire.Fire(launch)