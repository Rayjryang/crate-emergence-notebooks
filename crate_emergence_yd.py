import sys
# modify the path if it's not your directory
import torch
import crate_alpha
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO
from tqdm import tqdm
import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image


device = 'cuda'
# device = 'cpu'



url = 'xx the path of your checkpoint'  #  crate alpha  L/14 on 21k

patch_size = 14

model = crate_alpha.CRATEFeat(feat_dim = 1024, crate_arch = 'large', patch_size = patch_size ,pretrained_path = url, depth = 24, device = device)

token_index = 100*16


def collect_attention_maps(img_dir, image_size, patch_size, layer):
   

    img_list = sorted(os.listdir(img_dir))
    attn_list = []
    resized_images = []
    for img_name in tqdm(img_list):
        img_path = os.path.join(img_dir, img_name)
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        # transform = pth_transforms.Compose([
        #     pth_transforms.Resize((image_size, image_size)),
        #     pth_transforms.ToTensor(),
        #     pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])
        
        transform = pth_transforms.Compose([
            pth_transforms.Resize(image_size),
            pth_transforms.CenterCrop(size=(image_size, image_size)),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


        img = transform(img)
        w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
        img = img[:, :w, :h].unsqueeze(0)
        w_featmap = img.shape[-2] // patch_size
        h_featmap = img.shape[-1] // patch_size

        attentions, attentions_norm = model.forward_attn(img.to(device), layer=layer) #.  attentions: [b,h,seq_len,seq_len] # seq_len = n + 1
        attentions = attentions_norm
        nh = attentions.shape[1] # h
        # attentions = attentions[0, :, 0, 1:].reshape(nh, -1) #[h,seq_len-1]
        attentions = attentions[0, :, token_index, 1:].reshape(nh, -1) #[h,seq_len-1]
        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
        attn_list.append(attentions)
        resized_images.append(torchvision.utils.make_grid(img, normalize=True, scale_each=True).permute(1, 2, 0))

    return attn_list, resized_images


model_name='crate_alpha_L14_in21k'

# img_dir = './demo'
img_dir = './coco_stuff'
image_size = 224*4
# patch_size = 8
layer = 23


# save_dir = f'./vis_save/clip/{model_name}'
save_dir = f'./vis_save/{model_name}_coco_stuff'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


attentions_list, imgs = collect_attention_maps(img_dir, image_size, patch_size, layer)


for i, img in enumerate(imgs):
    attentions = attentions_list[i]
    nh = attentions.shape[0]
    plt.figure(figsize=(15, 3))
    plt.subplot(1, nh + 1,1)
    plt.imshow(img)
    plt.axis('off')
    for j in range(nh):
        plt.subplot(1, nh + 1, j + 2)
        plt.imshow(attentions[j])
        plt.axis('off')
    # 保存图像
    # save_path = os.path.join(save_dir, f"{model_name}_attention_map_token_{token_index}_{image_size}_{layer}_{i}.png")
    save_path = os.path.join(save_dir, f"{model_name}_attention_map_renorm_token_{token_index}_{image_size}_{layer}_{i}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # 关闭当前图像，释放内存

