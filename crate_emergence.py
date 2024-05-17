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

# url = '/HDD_data_storage_2u_1/jinruiyang/shared_space/code/SCLIP/clip/converted_checkpoints/fixed_ablation_ftin1k_base_in21k_res_mlp_decouple_x4_mixup_open_warm10_4096_lr1e5_91e_v3_128.pth'
# url = '/HDD_data_storage_2u_1/jinruiyang/shared_space/code/SCLIP/clip/converted_checkpoints/finetune_in1k_res_mlp_fixed_decouple_x4_yesra_mixup_open_warm10_4096_lr1e5_wd01_dp01_91e_L8_load_L8_v3_256_checkpoint.pth' #  crate alpha  L/8 on 1k
url = '/HDD_data_storage_2u_1/jinruiyang/shared_space/code/SCLIP/clip/converted_checkpoints/B8_in21k_res_mlp_fixed_decouple_x4_no_mixup_open_warm10_4096_lr5e5_wd01_91e_no_randaug_no_label_sm_v3_256_spot_checkpoint.pth' #  crate alpha  B/8 on 21k

# url = '/HDD_data_storage_2u_1/jinruiyang/shared_space/code/SCLIP/clip/converted_checkpoints/B32_ablation_in21k_mlp_nodecouple_x1_mixup_open_warm10_4096_lr5e5_91e_norangaug_no_label_sm_v3_128_checkpoint.pth' # vanilla crate b32 on 21k
# url = '/HDD_data_storage_2u_1/jinruiyang/shared_space/code/SCLIP/clip/converted_checkpoints/B32_ablation_in21k_res_mlp_decouple_x4_mixup_open_warm10_4096_lr5e5_91e_norangaug_no_label_sm_v3_128_checkpoint.pth' # crate alpha b32 on 21k
# url = '/HDD_data_storage_2u_1/jinruiyang/shared_space/code/SCLIP/clip/converted_checkpoints/B32_ablation_in21k_mlp_nodecouple_x4_mixup_open_warm10_4096_lr5e5_91e_norangaug_no_label_sm_v3_128_checkpoint.pth' # crate,4x,no decouple, no residual  b32 on 21k
# url = '/HDD_data_storage_2u_1/jinruiyang/shared_space/code/SCLIP/clip/converted_checkpoints/B32_ablation_in21k_mlp_decouple_x4_mixup_open_warm10_4096_lr5e5_91e_norangaug_no_label_sm_v3_128_checkpoint.pth' # crate,4x,decouple, no residual  b32 on 21k






model = crate_alpha.CRATEFeat(feat_dim = 768, crate_arch = 'base', patch_size = 8 ,pretrained_path = url, device = device)




def collect_attention_maps(img_dir, image_size, patch_size, layer):
   

    img_list = sorted(os.listdir(img_dir))
    attn_list = []
    resized_images = []
    for img_name in tqdm(img_list):
        img_path = os.path.join(img_dir, img_name)
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        transform = pth_transforms.Compose([
            pth_transforms.Resize((image_size, image_size)),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # transform = pth_transforms.Compose([
        #     pth_transforms.Resize(256),
        #     pth_transforms.CenterCrop(size=(image_size, image_size)),
        #     pth_transforms.ToTensor(),
        #     pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])


        img = transform(img)
        w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
        img = img[:, :w, :h].unsqueeze(0)
        w_featmap = img.shape[-2] // patch_size
        h_featmap = img.shape[-1] // patch_size

        attentions = model.forward_attn(img.to(device), layer=layer)
        nh = attentions.shape[1]
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
        attn_list.append(attentions)
        resized_images.append(torchvision.utils.make_grid(img, normalize=True, scale_each=True).permute(1, 2, 0))

    return attn_list, resized_images

model_name='B8'
dataset = 'in21k'

img_dir = './demo'
image_size = 224
patch_size = 8
layer = 11


save_dir = f'./vis_save/crate_alpha_{model_name}_{dataset}'
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
    save_path = os.path.join(save_dir, f"{model_name}_attention_map_{image_size}_{layer}_{i}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # 关闭当前图像，释放内存



img_dir = './semantic'
image_size = 224
patch_size = 8
layer = 10

# from matplotlib.backends.backend_pdf import PdfPages

attentions_list, imgs = collect_attention_maps(img_dir, image_size, patch_size, layer)
# selected heads with semantic meaning


columns = [0, 1, 3, 4]
for i, img in enumerate(imgs):
    attentions = attentions_list[i]
    nh = len(columns)
    plt.figure(figsize=(15, 3))
    plt.subplot(1, nh + 1,1)
    plt.imshow(img)
    plt.axis('off')
    for j, head_idx in enumerate(columns):
        plt.subplot(1, nh + 1, j + 2)
        plt.imshow(attentions[head_idx])
        plt.axis('off')

    # 将当前图像添加到 PDF 文件
    # pdf.savefig()
    # plt.close()  # 关闭当前图像，防止下一个图像叠加在同一个图上

   # 保存图像
    save_path = os.path.join(save_dir, f"{model_name}_semantic_{image_size}_{layer}_{i}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # 关闭当前图像，释放内存
