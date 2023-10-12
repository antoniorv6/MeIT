import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchinfo import summary
from collections import OrderedDict
from torch.nn.init import xavier_uniform_

from transformers import BeitConfig, BeitModel

class PositionalEncoding1D(nn.Module):

    def __init__(self, dim, len_max):
        super(PositionalEncoding1D, self).__init__()
        self.len_max = len_max
        self.dim = dim
        self.pe = torch.zeros((1, dim, len_max), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), requires_grad=False)

        div = torch.exp(-torch.arange(0., dim, 2) / dim * torch.log(torch.tensor(10000.0))).unsqueeze(1)
        l_pos = torch.arange(0., len_max)

        self.pe[:, ::2, :] = torch.sin(l_pos * div).unsqueeze(0)
        self.pe[:, 1::2, :] = torch.cos(l_pos * div).unsqueeze(0)

        self.pe = self.pe.permute(0,2,1).contiguous()

    def forward(self, x, start):
        """
        Add 1D positional encoding to x
        x: (B, L, C)
        start: index for x[:,:, 0]
        """
        if isinstance(start, int):
            return x + self.pe[:, start:start+x.size(1), :].to(x.device)
        else:
            for i in range(x.size(0)):
                x[i] = x[i] + self.pe[0, start[i]:start[i]+x.size(1), :]
            return x

class PatchEmbeddings(nn.Module):
    def __init__(self, num_channels, d_model, patch_size=16) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_embedding_layer = nn.Conv2d(num_channels, d_model, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        return self.patch_embedding_layer(x).flatten(2).transpose(1, 2)

class Embeddings(nn.Module):
    def __init__(self, max_img_size, num_channels, d_model, patch_size, use_masking=False) -> None:
        super().__init__()
        # Token de clasificación, es un token de embedding (por lo que tiene el tamaño de un token de un batch y una profundidad específica)
        self.classification_token = nn.Parameter(torch.zeros(1,1, d_model))
        # Si usamos máscaras, tendremos que hacer lo mismo para el preentrenamiento. Es decir, generar un embedding para el token de máscaras
        if use_masking:
            self.mask_token = nn.Parameter(torch.zeros(1,1, d_model))
        else:
            self.mask_token = None
        
        self.max_num_patches = ((max_img_size[0] // patch_size[0]) * (max_img_size[1] // patch_size[1])) + 1

        self.positional_encoding = PositionalEncoding1D(dim=d_model, len_max=self.max_num_patches)
        self.embedding_layer = PatchEmbeddings(num_channels, d_model, patch_size=patch_size)

    def forward(self, x, mim_mask=None):
        # Dividimos la imagen en patches
        x = self.embedding_layer(x)
        batch_size, seq_len, _ = x.size()
        
        if mim_mask is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            w = mim_mask.unsqueeze(-1).type_as(mask_tokens)
            x = x * (1 - w) + mask_tokens * w

        class_tokens = self.classification_token.expand(batch_size, -1, -1)
        x = torch.cat((class_tokens, x), dim=1)
        x = self.positional_encoding(x, start=0)
        
        return x

class MHA(nn.Module):

    def __init__(self, embedding_dim, num_heads=None, dropout=0, proj_value=True) -> None:
        super().__init__()

        self.proj_value = proj_value
        self.lq = nn.Linear(embedding_dim, embedding_dim)
        self.lk = nn.Linear(embedding_dim, embedding_dim)
        if proj_value:
            self.lv = nn.Linear(embedding_dim, embedding_dim)
        
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale_factor = float(self.head_dim) ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, query, key, value, key_pad_mask=None, attn_mask=None):
        
        target_len, b, c = query.size()
        source_len = key.size(0)

        q = self.lq(query)
        k = self.lk(key)
        v = self.lv(value) if self.proj_value else value

        q = torch.reshape(q, (target_len, b*self.num_heads, self.head_dim)).transpose(0, 1)
        k = torch.reshape(k, (source_len, b*self.num_heads, self.head_dim)).transpose(0, 1)
        v = torch.reshape(v, (source_len, b*self.num_heads, self.head_dim)).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1,2))

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if attn_mask.dtype == torch.bool:
                attn_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_weights += attn_mask

        if key_pad_mask is not None:
            attn_output_weigths = attn_weights.view(b, self.num_heads, target_len, source_len)
            attn_output_weigths = attn_weights.masked_fill(key_pad_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
            attn_output_weigths = attn_weights.view(b*self.num_heads, target_len, source_len)
        else:
            attn_output_weigths = attn_weights
        
        attn_output_weigths_raw = self.softmax(attn_output_weigths)
        attn_output_weigths = self.dropout(attn_output_weigths_raw)

        attn_output = torch.bmm(attn_output_weigths, v)
        attn_output = attn_output.transpose(0,1).contiguous().view(target_len, b, c)
        attn_output = self.out_proj(attn_output)
        
        return attn_output

    def init_weights(self):
        xavier_uniform_(self.in_proj_q.weight)
        xavier_uniform_(self.in_proj_k.weight)
        if self.proj_value:
            xavier_uniform_(self.in_proj_v.weight)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, attention_heads, dim_ff, dropout=0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.dim_ff = dim_ff

        self.input_mha = MHA(embedding_dim=self.d_model,
                             num_heads=attention_heads,
                             proj_value=True,
                             dropout=dropout)
        
        self.norm1 = nn.LayerNorm(self.d_model)

        self.ffNet = nn.Sequential(
            nn.Linear(self.d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim_ff, self.d_model)
        )

        self.dropout = nn.Dropout(0.1)

        self.norm2 = nn.LayerNorm(self.d_model)
    
    def forward(self, x):
        x = x.permute(1,0,2).contiguous()
        x2 = self.input_mha(x, x, x)
        x = x + self.dropout(x2)
        x = self.norm1(x)
        x2 = self.ffNet(x)
        x = x + self.dropout(x2)
        x = self.norm2(x)
        return x.permute(1,0,2).contiguous()

class ViTModel(nn.Module):
    def __init__(self, max_img_size, num_channels, d_model, patch_size, use_masking, attention_heads, dim_ff, num_enc_layers) -> None:
        super().__init__()
        self.embedding_layer = Embeddings(max_img_size=max_img_size, num_channels=num_channels, 
                          d_model=d_model, patch_size=patch_size, 
                          use_masking=use_masking)
        self.encoderStack = nn.Sequential(OrderedDict([(f"enc_layer_{idx}", EncoderLayer(d_model, attention_heads, dim_ff)) for idx in range(num_enc_layers)]))
    
    def forward(self, x, mask_tokens=None):
        x = self.embedding_layer(x, mask_tokens)
        x = self.encoderStack(x)
        
        return x


    
