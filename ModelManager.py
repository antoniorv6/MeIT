import torch
import wandb
import lightning.pytorch as L

from math import sqrt, exp
from torchinfo import summary
from torchvision.utils import make_grid
from dalle_pytorch import DiscreteVAE

class DVAE(L.LightningModule):
    def __init__(self, vocab_size=8192) -> None:
        super().__init__()
        self.model = DiscreteVAE(
                    image_size = 1024,
                    num_layers = 4,           # number of downsamples - ex. 256 / (2 ** 3) = (32 x 32 feature map)
                    num_tokens = vocab_size,        # number of visual tokens. in the paper, they used 8192, but could be smaller for downsized projects
                    codebook_dim = 512,       # codebook dimension
                    hidden_dim = 64,          # hidden dimension
                    num_resnet_blocks = 1,    # number of resnet blocks
                    temperature = 0.9,        # gumbel softmax temperature, the lower this is, the harder the discretization
                    straight_through = False, # straight-through for gumbel softmax. unclear if it is better one way or the other
                    )
        self.temp = 0.9
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
            hard_recons = self.model.decode(codes, x.size(2)//16, x.size(3)//16)

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

def get_DVAE_model(maxheight, maxwidth, in_channels=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DVAE().to(device)
    summary(model, input_size=[(1,in_channels,maxheight,maxwidth)], dtypes=[torch.float])
    return model
    

    


