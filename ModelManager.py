from typing import Any
import numpy as np
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import wandb
import lightning.pytorch as L
import torch.nn as nn

from math import sqrt, exp
from torchinfo import summary
from torchvision.utils import make_grid
from dalle_pytorch import DiscreteVAE

from own_impl.ViT import ViTModel
from transformers import get_linear_schedule_with_warmup, BeitConfig, BeitForMaskedImageModeling

from vit_pytorch.vit import ViT
from vit_pytorch.mae import MAE

class DVAE(L.LightningModule):
    def __init__(self, num_layers=4, patch_size=16, vocab_size=8192) -> None:
        super().__init__()
        self.model = DiscreteVAE(
                    image_size = 1024,
                    num_layers = num_layers,           # number of downsamples - ex. 256 / (2 ** 3) = (32 x 32 feature map)
                    num_tokens = vocab_size,        # number of visual tokens. in the paper, they used 8192, but could be smaller for downsized projects
                    codebook_dim = 512,       # codebook dimension
                    hidden_dim = 64,          # hidden dimension
                    num_resnet_blocks = 1,    # number of resnet blocks
                    temperature = 0.9,        # gumbel softmax temperature, the lower this is, the harder the discretization
                    straight_through = False, # straight-through for gumbel softmax. unclear if it is better one way or the other
                    )
        self.temp = 0.9
        self.patch_size = patch_size
        self.save_hyperparameters()
    
    def forward(self, x, return_loss=False, return_recons=False, temp=0.9):
        return self.model(x, return_loss=return_loss, return_recons=return_recons, temp=temp)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        return {"optimizer":optimizer, "lr_scheduler":scheduler}
    
    def training_step(self, train_batch, batch_idx):
        k=1
        x = train_batch
        if batch_idx % 500 == 0:
            loss, recons = self.forward(x, return_loss=True, return_recons=True)
            images = x
            codes = self.model.get_codebook_indices(x[:k])
            hard_recons = self.model.decode(codes, x.size(2)//self.patch_size, x.size(3)//self.patch_size)

            images, recons = map(lambda t: t[:k], (images, recons))
            images, recons, hard_recons, codes = map(lambda t: t.detach().cpu(), (images, recons, hard_recons, codes))
            images, recons, hard_recons = map(lambda t: make_grid(t.float(), nrow = int(sqrt(k)), normalize = True, range = (-1, 1)), (images, recons, hard_recons))
            self.logger.log_image('Original images', [wandb.Image(image) for image in images])
            self.logger.log_image('Reconstructions', [wandb.Image(image) for image in recons])
            self.logger.log_image('Hard reconstructions', [wandb.Image(image) for image in hard_recons])
            self.logger.experiment.log({'Codebook indices': wandb.Histogram(codes)})
            self.temp = max(self.temp * exp(-1e-6 * self.global_step), 0.5)

        else:
            loss = self.forward(x, return_loss=True, temp=self.temp)

        self.log('loss', loss, on_epoch=True, batch_size=k, prog_bar=True)
        
        return loss

### MY OWN IMPLEMENTATION OF THE BEIT

class MeIT(L.LightningModule):
    def __init__(self, max_image_size, vocab_size=8192, num_channels=3, d_model=256, attention_heads=8, dim_ff=1024, num_layers=6, patch_size=(16,16)) -> None:
        super().__init__()
        self.model = ViTModel(max_img_size=max_image_size, 
                 num_channels=num_channels, 
                 d_model=d_model, 
                 patch_size=patch_size, 
                 use_masking=True, attention_heads=attention_heads, 
                 dim_ff=dim_ff, num_enc_layers=num_layers)
        
        self.outLayer = nn.Linear(in_features=d_model, out_features=vocab_size)
        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()
    
    def forward(self, x, patch_mask=None):
        return self.outLayer(self.model(x, patch_mask))
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            weight_decay=0.05)
        
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x = train_batch[0]
        mask = train_batch[1]
        gt = train_batch[2]
        logits = self.forward(x, patch_mask=mask)
        logits = logits[:, 1:, :]
        logits = logits[mask]
        gt = gt[mask]
        loss = self.loss(logits.unsqueeze(0).permute(0,2,1).contiguous(), gt.unsqueeze(0))
        #loss = self.loss(logits.permute(0,2,1).contiguous(), gt)
        self.log('loss', loss, on_epoch=True, batch_size=1, prog_bar=True)
        return loss
    
####

#### HUGGINGFACE TRANSFORMERS IMPLEMENTATION FOR SANITY CHECKING

class MeITHuggingFace(L.LightningModule):
    def __init__(self, max_image_size, vocab_size=8192, num_channels=3, d_model=768, attention_heads=12, dim_ff=3072, num_layers=12, patch_size=(16,16)) -> None:
        super().__init__()
        custom_config = BeitConfig(use_mask_token=True, image_size=max_image_size, patch_size=patch_size, vocab_size=vocab_size)
        self.model = BeitForMaskedImageModeling(custom_config)
        
        self.outLayer = nn.Linear(in_features=d_model, out_features=vocab_size)
        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()
    
    def forward(self, x, patch_mask=None):
        return self.model(x, bool_masked_pos=patch_mask)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            weight_decay=0.05)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=40000,
            num_training_steps=400000
        )

        #return optimizer
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # Step-wise LR scheduling
            }
        }
    
    def training_step(self, train_batch, batch_idx):
        x = train_batch[0]
        mask = train_batch[1]
        gt = train_batch[2]
        output = self.model(x, bool_masked_pos=mask, labels=gt[mask])
        loss = output.loss
        self.log('loss', loss, on_epoch=True, batch_size=8, prog_bar=True)
        return loss
####

class MeIT_MAE(L.LightningModule):
    def __init__(self, image_size):
        super().__init__()
        self.model = ViT(
            image_size = image_size,
            patch_size = 32,
            num_classes = 1000,
            dim = 1024,
            depth = 6,
            heads = 8,
            mlp_dim = 2048
        )

        self.mae = MAE(
            encoder=self.model,
            decoder_dim=512,
            masking_ratio=0.75,
            decoder_depth=6
        )

        self.save_hyperparameters()
    
    def forward(self, x):
        patches = self.mae.to_patch(x)
        num_patches, _ = self.model.pos_embedding.shape[-2:]
        tokens = self.mae.patch_to_emb(patches)
        if self.model.pool == "cls":
            tokens += self.model.pos_embedding[:, 1:(num_patches + 1)]
        elif self.model.pool == "mean":
            tokens += self.model.pos_embedding.to(self.device, dtype=tokens.dtype) 
        
        return self.model.transformer(tokens)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
            lr=1.5e-3,
            betas=(0.9, 0.999),
            weight_decay=0.05)
        
        return optimizer
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=40000,
            num_training_steps=400000
        )

        #return optimizer
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # Step-wise LR scheduling
            }
        }

    def training_step(self, train_batch, batch_idx):
        x = train_batch
        loss = self.mae(x)
        self.log('loss', loss, on_epoch=True, batch_size=1, prog_bar=True)
        return loss

## UTILITY FUNCS

def get_DVAE_model(maxheight, maxwidth, num_layers=4, patch_size=16, vocab_size=8192, in_channels=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DVAE(num_layers=num_layers, patch_size=patch_size, vocab_size=vocab_size).to(device)
    summary(model, input_size=[(1,in_channels,maxheight,maxwidth)], dtypes=[torch.float])
    return model

def get_MeIT_model(maxheight, maxwidth, in_channels=3, vocab_size=8192, patch_size=(16,16)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MeITHuggingFace(max_image_size=(maxheight, maxwidth), vocab_size=vocab_size, patch_size=patch_size).to(device)
    summary(model, input_size=[(1, in_channels,maxheight,maxwidth)], dtypes=[torch.float])
    return model

def get_MAE_VIT_model(maxheight, maxwidth, in_channels=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MeIT_MAE(image_size=(maxheight, maxwidth)).to(device)
    summary(model, input_size=[(1, in_channels, maxheight, maxwidth)], dtypes=[torch.float])
    return model
    

    


