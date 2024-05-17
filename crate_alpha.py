import torch
from torch import nn
import os
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch.nn.init as init
import pdb

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class ISTABlock(nn.Module):
    """Transformer MLP / feed-forward block."""
    
    def __init__(self, d, eta=0.1, lmbda=0.1):
        super(ISTABlock, self).__init__()
        self.eta = eta
        self.lmbda = lmbda

        self.D = nn.Parameter(torch.randn(d, d) * 0.02)
        
    def forward(self, x):
        n, l, d = x.shape

        Dx = torch.einsum('dp,nlp->nld', self.D, x)
        lasso_grad = torch.einsum('pd,nlp->nld', self.D, Dx - x)
        x = F.relu(x - self.eta * lasso_grad - self.eta * self.lmbda)
      
        return x


class OvercompleteISTABlock(nn.Module):
    """Transformer MLP / feed-forward block."""
    def __init__(self, d, overcomplete_ratio=4, eta=0.1, lmbda=0.1, decouple=True):
        super(OvercompleteISTABlock, self).__init__()
        self.eta = eta
        self.lmbda = lmbda
        self.overcomplete_ratio = overcomplete_ratio
        self.decouple = decouple
        self.d = d

        # Define the matrix D
        self.D = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d, overcomplete_ratio * d)))

        if self.decouple:
            self.D1 = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d, overcomplete_ratio * d)))

    def forward(self, x):
        """Applies CRATE OvercompleteISTABlock module."""
        
        # First step of PGD: initialize at z0 = 0, compute lasso prox, get z1
        negative_lasso_grad = torch.einsum("pd,nlp->nld", self.D, x)
        z1 = F.relu(self.eta * negative_lasso_grad - self.eta * self.lmbda)

        # Second step of PGD: initialize at z1, compute lasso prox, get z2
        Dz1 = torch.einsum("dp,nlp->nld", self.D, z1)
        lasso_grad = torch.einsum("pd,nlp->nld", self.D, Dz1 - x)
        z2 = F.relu(z1 - self.eta * lasso_grad - self.eta * self.lmbda)

        if self.decouple:
            xhat = torch.einsum("dp,nlp->nld", self.D1, z2)
        else:
            xhat = torch.einsum("dp,nlp->nld", self.D, z2)
      
        return xhat
    

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.qkv = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x,  return_attention=False, return_key = False):
        if return_key:
            return self.qkv(x)
        
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h = self.heads)

        dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)
        if return_attention:
            return attn

        out = torch.matmul(attn, w)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, IsOvercompleteISTABlock , residual_mlp, decouple, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.heads = heads
        self.depth = depth
        self.dim = dim
        self.residual_mlp = residual_mlp

        for _ in range(depth):
            
            if IsOvercompleteISTABlock:
                block = PreNorm(dim, OvercompleteISTABlock(d=dim,decouple=decouple))
            else:
                block = PreNorm(dim, ISTABlock(d=dim))

            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                block
            ]))


    def forward(self, x):
      
        for attn, ff in self.layers:
            # pdb.set_trace()
            x = attn(x) + x
            x = ff(x) + x if self.residual_mlp else ff(x)
        return x 


class CRATE(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, IsOvercompleteISTABlock, residual_mlp, decouple ,pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.patch_size = patch_size
        self.dim = dim
        self.residual_mlp = residual_mlp

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=patch_size, stride=patch_size, bias=True, dtype=torch.float32, padding='valid')

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, IsOvercompleteISTABlock, residual_mlp, decouple, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):

        x = self.conv1(img)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        
        x = self.transformer(x)
        feature_pre = x

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        feature_last = x
     
        return self.mlp_head(x)

    def get_last_selfattention(self, img, layer = 5):
        x = self.conv1(img)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)


        for i, (attn, ff) in enumerate(self.transformer.layers):
            if i < layer:
                grad_x = attn(x) + x
                x = ff(grad_x) + grad_x if self.residual_mlp else ff(grad_x)
            else:
                attn_map = attn(x, return_attention=True)
                # print(attn_map.shape)
                return attn_map
            

def CRATE_tiny():
    return CRATE(image_size=224,
                    patch_size=16,
                    num_classes=1000,
                    dim=384,
                    depth=12,
                    heads=6,
                    dropout=0.0,
                    emb_dropout=0.0,
                    dim_head=384//6)

def CRATE_small():
    return CRATE(image_size=224,
                    patch_size=16,
                    num_classes=1000,
                    dim=576,
                    depth=12,
                    heads=12,
                    dropout=0.0,
                    emb_dropout=0.0,
                    dim_head=576//12)

def CRATE_base():
    return CRATE(image_size=224,
                patch_size=8,
                num_classes=19167,
                dim=768,
                depth=12,
                heads=12,
                dropout=0.0,
                IsOvercompleteISTABlock=True,
                residual_mlp=True,
                decouple=True,
                emb_dropout=0.0,
                dim_head=768//12)

def CRATE_large():
    return CRATE(image_size=224,
                patch_size=8,
                num_classes=1000,
                dim=1024,
                depth=24,
                heads=16,
                dropout=0.0,
                IsOvercompleteISTABlock=True,
                residual_mlp=True,
                decouple=True,
                emb_dropout=0.0,
                dim_head=1024//16)


import math
from typing import Union, List, Tuple
import types


class CRATEFeat(nn.Module):
    def __init__(self,  feat_dim, pretrained_path = None, depth = 11, crate_arch = 'base', patch_size = 8, device = 'cpu'):
        super().__init__()
        if crate_arch == 'small':
            self.model = CRATE_small_21k()
        elif crate_arch == 'base':
            self.model = CRATE_base()
        elif  crate_arch == 'large':
            self.model = CRATE_large()
        elif crate_arch == 'demo':
            self.model = CRATE_base_demo()
            self.model.mlp_head = nn.Sequential(
                nn.LayerNorm(768),
                nn.Linear(768,768)
            )
            self.model.head = nn.Linear(768, 21842)
            
        self.feat_dim = feat_dim
        self.patch_size = patch_size
        self.depth = depth
        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
            # self.model.load_state_dict(state_dict['model'], strict=False)
            self.model.load_state_dict(state_dict)
            print('Loading weight from {}'.format(pretrained_path))
        self.model = self.patch_vit_resolution(self.model, stride = patch_size) # mark
        self.model.to(device)
        

    def patch_vit_resolution(self, model: nn.Module, stride: int) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        patch_size = model.patch_size
        stride = (stride, stride)
        model.interpolate_pos_encoding = types.MethodType(CRATEFeat._fix_pos_enc(patch_size, stride), model)
        return model
        
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        """
        Creates a method for position encoding interpolation.
        :param patch_size: patch size of the model.
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :return: the interpolation method
        """
        def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
            npatch = x.shape[1] - 1
            N = self.pos_embedding.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embedding
            class_pos_embed = self.pos_embedding[:, 0]
            patch_pos_embed = self.pos_embedding[:, 1:]
            dim = self.dim
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
                align_corners=False, recompute_scale_factor=False
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding
    
    def forward_attn(self, img, layer = 11):
        with torch.no_grad():
            
            h, w = img.shape[2], img.shape[3]
            feat_h, feat_w = h // self.patch_size, w // self.patch_size
            img = img[:, :, :feat_h * self.patch_size, :feat_w * self.patch_size]
            pos_em = self.model.interpolate_pos_encoding(img, w, h)
            self.model.pos_embedding = nn.Parameter(self.model.interpolate_pos_encoding(img, w, h))
            attentions = self.model.get_last_selfattention(img, layer = layer)
            return attentions
        
    def forward(self, img):
        with torch.no_grad():
            
            h, w = img.shape[2], img.shape[3]
            feat_h, feat_w = h // self.patch_size, w // self.patch_size
            img = img[:, :, :feat_h * self.patch_size, :feat_w * self.patch_size]
            pos_em = self.model.interpolate_pos_encoding(img, w, h)
            self.model.pos_embedding = nn.Parameter(self.model.interpolate_pos_encoding(img, w, h))
            attentions = self.model.get_last_selfattention(img, layer = self.depth)
            bs, nb_head, nb_token = attentions.shape[0], attentions.shape[1], attentions.shape[2]
            qkv = self.model.get_last_key(img, depth = self.depth)
            qkv = qkv[None, :, :, :]
            return qkv[:, :, 1:, :]
            




# model = CRATE_base()
# dir = '/HDD_data_storage_2u_1/jinruiyang/shared_space/code/SCLIP/clip/converted_checkpoints/'
# pth_path = os.path.join(dir,'B32_ablation_in21k_mlp_nodecouple_x1_mixup_open_warm10_4096_lr5e5_91e_norangaug_no_label_sm_v3_128_checkpoint.pth')
# # # 加载预训练权重
# model_weights = torch.load(pth_path)
# # # 将权重加载到模型
# model.load_state_dict(model_weights)

# for k,v in model.state_dict().items():
#     print(k,v.shape)


# model = model.eval()

# bs = 32
# img_size = 224
# fake_image = torch.ones((bs, 3, img_size, img_size))  # (batch_size, channels, height, width)

# res = model(fake_image)
# # torch.abs(d['conv1.weight']).sum()
# # torch.abs(d['conv1.bias']).sum()
# print(res,res.shape)
# # total_params = sum(p.numel() for p in model.parameters())
# # print(f"Total Parameters: {total_params}")
# #np.mean(d['conv1.weight'])