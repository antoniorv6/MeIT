import fire
import torch
from data import DeepScoresMasking, ds2_masking_collade_func
from torch.utils.data import DataLoader
from ModelManager import get_MeIT_model
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import torch.nn as nn

torch.set_float32_matmul_precision('high')

def main(model_name, patch_size, vocab_size):
    dataset = DeepScoresMasking(checkpoint_path=f"weights/DVAE_{patch_size}.ckpt", patch_size=patch_size)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=20, collate_fn=ds2_masking_collade_func)
    mheight, mwidth = dataset.get_max_hw()

    model = get_MeIT_model(mheight, mwidth, vocab_size=vocab_size, patch_size=(patch_size,patch_size))
    
    wandb_logger = WandbLogger(project='LITIS_STAY', group=f"SSL", name=f"{model_name}", log_model=False)
    
    checkpointer = ModelCheckpoint(dirpath=f"weights/", filename=f"{model_name}", every_n_train_steps=100, verbose=True)

    trainer = Trainer(max_epochs=10, logger=wandb_logger, accumulate_grad_batches=8, callbacks=[checkpointer])

    trainer.fit(model, train_dataloaders=dataloader)
    
def launch(model_name, patch_size, vocab_size):
    main(model_name, patch_size, vocab_size)

if __name__ == "__main__":
    fire.Fire(launch)
