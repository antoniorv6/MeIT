from typing import Any, Optional
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
from vit_pytorch.simple_vit import SimpleViT
from vit_pytorch.mae import MAE
from vit_pytorch.mpp import MPP
from einops import rearrange

from DAN.DanDecoder import Decoder

from einops.layers.torch import Rearrange

from metrics_computing import compute_poliphony_metrics

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
class MiT_MAE(L.LightningModule):
    def __init__(self, image_size, patch_size):
        super().__init__()
        self.img_size = image_size
        self.model = SimpleViT(
            image_size = image_size,
            patch_size = patch_size,
            num_classes = 1000,
            dim = 768,
            depth = 12,
            heads = 12,
            mlp_dim = 3072
        )

        self.mae = MAE(
            encoder=self.model,
            decoder_dim=512,
            masking_ratio=0.40,
            decoder_depth=6
        )

        self.rand_index = np.random.randint(0, 200)
        self.patch_size = patch_size
        self.valrr = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
        self.val_losses = []
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
        #optimizer = torch.optim.AdamW(self.parameters(),
        #    lr=1.5e-3,
        #    betas=(0.9, 0.999),
        #    weight_decay=0.05)
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=3e-4,
                                     betas=(0.9,0.999))
        
        return optimizer
        
    def training_step(self, train_batch, batch_idx):
        x = train_batch
        loss = self.mae(x)
        self.log('loss', loss, on_epoch=True, batch_size=1, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx) -> STEP_OUTPUT | None:
        x = val_batch
        if batch_idx % self.rand_index  == 0:
            image = x
            loss, preds, mask = self.mae(x, return_predictions=True)
            patched_image = self.valrr(image).squeeze()
            mask = mask.squeeze()
            preds = preds.squeeze()
            unmasked_image = patched_image.clone()

            for idx, prediction in enumerate(preds):
                unmasked_image[mask[idx], :] = prediction
            
            prediction = patches_to_image(unmasked_image.unsqueeze(0), original_height=image.size(2), p1=self.patch_size, p2=self.patch_size)

            self.logger.log_image('Original image', [wandb.Image(image)])
            self.logger.log_image('Reconstruction', [wandb.Image(prediction)])

            self.val_losses.append(loss.item())

            return loss

        loss = self.mae(x)
        self.val_losses.append(loss.item())
        return loss
    
    def on_validation_epoch_end(self) -> None:
        self.log('val_loss', np.mean(self.val_losses))
        self.val_losses = []



class MiT_MPP(L.LightningModule):
    def __init__(self, image_size, patch_size):
        super().__init__()
        self.img_size = image_size
        self.model = ViT(
            image_size = image_size,
            patch_size = patch_size,
            num_classes = 1000,
            dim = 768,
            depth = 12,
            heads = 12,
            mlp_dim = 3072
        )

        self.mpp = MPP(
            transformer=self.model,
            patch_size=32,
            dim=768,
            mask_prob=0.15,
            random_patch_prob=0.3,
            replace_prob=0.5
        )

        self.rand_index = np.random.randint(0, 200)
        self.patch_size = patch_size
        self.val_losses = []
        self.save_hyperparameters()
    
    def forward(self, x):
        patches = rearrange(x,
                          'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                          p1=self.patch_size,
                          p2=self.patch_size)
        #num_patches, _ = self.model.pos_embedding.shape[-2:]
        tokens = self.mpp.patch_to_emb(patches)
        
        return self.model.transformer(tokens)[:, 1:, :]
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
            lr=1.5e-3,
            betas=(0.9, 0.999),
            weight_decay=0.05)
        
        return optimizer
        
    def training_step(self, train_batch, batch_idx):
        x = train_batch
        loss = self.mpp(x)
        self.log('loss', loss, on_epoch=True, batch_size=1, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx) -> STEP_OUTPUT | None:
        x = val_batch
        loss = self.mpp(x)
        self.val_losses.append(loss.item())
        return loss
    
    def on_validation_epoch_end(self) -> None:
        self.log('val_loss', np.mean(self.val_losses))
        self.val_losses = []


####### DAN
class DAN(L.LightningModule):
    def __init__(self, encoder_path, maxlen, out_categories, padding_token, w2i, i2w, out_dir, d_model=None, dim_ff=None) -> None:
        super().__init__()
        self.encoder = MiT_MAE.load_from_checkpoint(encoder_path)
        self.decoder = Decoder(d_model, dim_ff, maxlen, out_categories)

        self.padding_token = padding_token

        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token)

        self.valpredictions = []
        self.valgts = []

        self.w2i = w2i
        self.i2w = i2w
        self.maxlen = maxlen
        self.out_dir=out_dir

        self.save_hyperparameters()

    def forward(self, x, y_pred):
        encoder_output = self.encoder(x)
        return self.forward_decoder(encoder_output, y_pred, cache=None)

    def forward_encoder(self, x):
        return self.encoder(x)
    
    def forward_decoder(self, encoder_output, last_preds, cache=None):
        b, _, _ = encoder_output.size()
        reduced_size = [s.shape[1] for s in encoder_output]
        ylens = [len(sample) for sample in last_preds]
        cache = cache

        features = encoder_output.permute(1, 0, 2).contiguous()
        enhanced_features = encoder_output.permute(1, 0, 2).contiguous()
        output, predictions, _, _, weights = self.decoder(features, enhanced_features, last_preds[:, :], reduced_size, 
                                                          [max(ylens) for _ in range(b)], encoder_output.size(), 
                                                          start=0, cache=cache, keep_all_weights=True)
    
        return output, predictions, cache, weights
    
    def configure_optimizers(self):
        return torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=0.0001, amsgrad=False)

    def training_step(self, train_batch):
        x, di, y = train_batch
        output, predictions, cache, weights = self.forward(x, di)
        loss = self.loss(predictions, y)
        self.log("loss", loss, on_epoch=True, batch_size=1, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x, _, y = val_batch
        encoder_output = self.forward_encoder(x)
        predicted_sequence = torch.from_numpy(np.asarray([self.w2i['<bos>']])).to(device).unsqueeze(0)
        cache = None
        for i in range(self.maxlen):
            output, predictions, cache, weights = self.forward_decoder(encoder_output, predicted_sequence.long(), cache=cache)
            predicted_token = torch.argmax(predictions[:, :, -1]).item()
            predicted_sequence = torch.cat([predicted_sequence, torch.argmax(predictions[:, :, -1], dim=1, keepdim=True)], dim=1)
            if predicted_token == self.w2i['<eos>']:
                break
        
        dec = "".join([self.i2w[token.item()] for token in predicted_sequence.squeeze(0)[1:]])
        dec = dec.replace("<t>", "\t")
        dec = dec.replace("<b>", "\n")
        dec = dec.replace("<s>", " ")

        gt = "".join([self.i2w[token.item()] for token in y.squeeze(0)[:-1]])
        gt = gt.replace("<t>", "\t")
        gt = gt.replace("<b>", "\n")
        gt = gt.replace("<s>", " ")

        self.valpredictions.append(dec)
        self.valgts.append(gt)
    
    def test_step(self, test_batch, batch_idx):
        self.validation_step(test_batch, batch_idx)
    
    def get_dictionaries(self):
        return self.w2i, self.i2w

class Poliphony_DAN(DAN):
    def __init__(self, encoder_path, maxlen, out_categories, padding_token, w2i, i2w, out_dir) -> None:
        super().__init__(encoder_path, maxlen, out_categories, padding_token, w2i, i2w, out_dir, d_model=768, dim_ff=768)
    
    def on_validation_epoch_end(self):
        cer, ser, ler = compute_poliphony_metrics(self.valpredictions, self.valgts)
        
        random_index = np.random.randint(0, len(self.valpredictions))
        predtoshow = self.valpredictions[random_index]
        gttoshow = self.valgts[random_index]
        print(f"[Prediction] - {predtoshow}")
        print(f"[GT] - {gttoshow}")

        self.log('val_CER', cer, prog_bar=True)
        self.log('val_SER', ser, prog_bar=True)
        self.log('val_LER', ler, prog_bar=True)

        self.valpredictions = []
        self.valgts = []

        return ser

    def on_test_epoch_end(self):
        cer, ser, ler = compute_poliphony_metrics(self.valpredictions, self.valgts)

        for index, sample in enumerate(self.valpredictions):
            with open(f"{self.out_dir}/hyp/{index}.krn", "w+") as krnfile:
                krnfile.write(sample)
        
        for index, sample in enumerate(self.valgts):
            with open(f"{self.out_dir}/gt/{index}.krn", "w+") as krnfile:
                krnfile.write(sample)

        self.log('test_CER', cer)
        self.log('test_SER', ser)
        self.log('test_LER', ler)

        self.valpredictions = []
        self.valgts = []

        return ser


#######

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

def get_MAE_VIT_model(maxheight, maxwidth, patch_size, in_channels=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiT_MAE(image_size=(maxheight, maxwidth), patch_size=patch_size).to(device)
    summary(model, input_size=[(1, in_channels, maxheight, maxwidth)], dtypes=[torch.float])
    return model

def get_MPP_VIT_model(maxheight, maxwidth, patch_size, in_channels=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiT_MPP(image_size=(maxheight, maxwidth), patch_size=patch_size).to(device)
    summary(model, input_size=[(1, in_channels, maxheight, maxwidth)], dtypes=[torch.float])
    return model

def get_DAN_MeIT(maxheight, maxwidth, maxlength, encoder_checkpoint, out_categories, padding_token, w2i, i2w):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Poliphony_DAN(encoder_path=encoder_checkpoint, maxlen=maxlength, out_categories=out_categories, 
                          padding_token=padding_token, w2i=w2i, i2w=i2w, out_dir="out").to(device)
    summary(model, input_size=[(1,3,maxheight,maxwidth), (1,maxlength)], dtypes=[torch.float, torch.long])
    return model

#if __name__ == "__main__":
    #get_DAN_MeIT(1280, 1280, 1000, "weights/MiT_MAE_B-epoch=1-step=480000-val_loss=0.10487.ckpt", 7000, 0, {}, {})

    