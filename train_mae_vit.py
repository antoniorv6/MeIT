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
    dataset = DeepScoresDataset(patch_size=patch_size)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=20, collate_fn=ds2_collade_func)
    
    mheight, mwidth = dataset.get_max_hw()

    print(mheight, mwidth)
    
    model = get_MAE_VIT_model(mheight, mwidth)

    wandb_logger = WandbLogger(project='LITIS_STAY', group=f"SSL", name=f"{model_name}", log_model=False)
    
    checkpointer = ModelCheckpoint(dirpath=f"weights/", filename=f"{model_name}", every_n_train_steps=500, verbose=True)

    trainer = Trainer(max_epochs=1, logger=wandb_logger, callbacks=[checkpointer])

    trainer.fit(model, train_dataloaders=dataloader)


def launch(model_name, patch_size, checkpoint_path=None):
    main(model_name, patch_size=patch_size)

if __name__ == "__main__":
    fire.Fire(launch)