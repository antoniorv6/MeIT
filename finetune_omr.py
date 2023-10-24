from torch.utils.data import DataLoader
from data_fp import load_data, batch_preparation_img2seq
from ModelManager import get_DAN_MeIT

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

def main():
    train_set, val_set, test_set = load_data(data_path="Data/grandstaff_dataset/partitions_fpgrandstaff/excerpts/fold_0", vocab_name="grandstaff")
    w2i, i2w = train_set.get_dictionaries()

    train_dataloader = DataLoader(train_set, batch_size=1, num_workers=20, collate_fn=batch_preparation_img2seq)
    val_dataloader = DataLoader(val_set, batch_size=1, num_workers=20, collate_fn=batch_preparation_img2seq)
    test_dataloader = DataLoader(val_set, batch_size=1, num_workers=20, collate_fn=batch_preparation_img2seq)

    maxheight = max([train_set.get_max_hw()[0], val_set.get_max_hw()[0], test_set.get_max_hw()[0]])
    maxwidth = max([train_set.get_max_hw()[1], val_set.get_max_hw()[1], test_set.get_max_hw()[1]])
    maxlen = max([train_set.get_max_seqlen(), val_set.get_max_seqlen(), test_set.get_max_seqlen()])

    model = get_DAN_MeIT(maxheight, maxwidth, maxlen, "weights/MiT_MAE/MiT_MAE_B-epoch=6-step=1700000-val_loss=0.06149.ckpt", 
                         out_categories=len(w2i), padding_token=w2i['<pad>'], w2i=w2i, i2w=i2w)
    
    wandb_logger = WandbLogger(project='LITIS_STAY', group=f"SSL", name=f"MeIT_MAE_GS", log_model=False)

    early_stopping = EarlyStopping(monitor='val_SER', min_delta=0.01, patience=5, mode="min", verbose=True)
    
    checkpointer = ModelCheckpoint(dirpath=f"weights/GrandStaff/", filename=f"DAN_MEiT_MAE", 
                                   monitor='val_SER', mode='min',
                                   save_top_k=1, verbose=True)

    trainer = Trainer(max_epochs=10000, check_val_every_n_epoch=10, logger=wandb_logger, callbacks=[checkpointer, early_stopping])
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)

if __name__ == "__main__":
    main()