import torch
# import clip
import requests
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms as tfms
from diffusers import StableDiffusionPipeline, DDIMScheduler 
from dataset import DiffDataset, TinyDataset, LocalDataset
import argparse
import os
from scipy import io
from diffusers import DiTPipeline
from diffusers.utils.torch_utils import randn_tensor
from transformers import ViTModel, ViTImageProcessor
import ipdb

def collate_fn(batch):
    # 过滤掉 batch 中为 None 的样本
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    # 将图像和标签分别堆叠成批次
    # images = torch.stack([item['images'] for item in batch])
    images = torch.cat([item['images']['pixel_values'] for item in batch], dim=0)
    labels = torch.tensor([item['labels'] for item in batch])
    idx = torch.tensor([item['idx'] for item in batch])
    paths = [item['paths'] for item in batch]

    return {
        'images': images,
        'labels': labels,
        'idx':idx,
        'paths': paths
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for InstructPix2Pix."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        # default="zh-plus/tiny-imagenet", # Load from website, but will face the index matching error in sample1() function
        default='/scratch/zhao.lin1/dataset/tiny-imagenet-200',
        help="dataset_path",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default='/scratch/zhao.lin1/dataset/tiny-imagenet-200/train_vit_image_feature_CLS',
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--end",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.gpu)
   
    
    diffdataset = LocalDataset(args.dataset_path,args.start, args.end)
    trainloader = torch.utils.data.DataLoader(diffdataset, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers, drop_last=False, collate_fn=collate_fn)
    
    
    # 加载模型
    # CLIP
    # model, preprocess = clip.load("ViT-B/32", device=device)

    # ViT
    model = ViTModel.from_pretrained("/scratch/zhao.lin1/dit_imagenet").to(device)
    # feature_extractor = ViTFeatureExtractor.from_pretrained("/scratch/zhao.lin1/dit_imagenet")
    # processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

    # pipe = DiTPipeline.from_pretrained("/home/user1/workspace/leilu/linzhao/dit_imagenet", torch_dtype=torch.float16)
    # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config) 
    # pipe = pipe.to(device)

    for sample in trainloader:
        with torch.no_grad():
            # CLIP
            # image_features = model.encode_image(sample['images'].to(device))
            
            # ViT
            inputs = sample['images'].to(device)
            # inputs = feature_extractor(images=sample['images'], return_tensors="pt").to(device)
            outputs = model(inputs)
            image_embedding = outputs.last_hidden_state[:,0]  # CLS token embedding
            
            
            for i in range(image_embedding.shape[0]):
                os.makedirs(os.path.join(args.save_dir, sample['paths'][i].split('train')[1].split('/')[1]), exist_ok=True)
                save_path = os.path.join(args.save_dir, sample['paths'][i].split('train')[1].split('/')[1], sample['paths'][i].split('train')[1].split('/')[3].split('.')[0]+'.pt') 
                torch.save(image_embedding[i].cpu(), save_path) 

            # for i in range(latent.latent_dist.mean.shape[0]):
                # latent = ls[i] 
                # os.makedirs(os.path.join(args.save_dir, sample['paths'][i].split('train')[1].split('/')[1]), exist_ok=True)
                # save_path = os.path.join(args.save_dir, sample['paths'][i].split('train')[1].split('/')[1], sample['paths'][i].split('train')[1].split('/')[3].split('.')[0]+'.pt') 
                # torch.save(latent.latent_dist.mean, mean_save_path) 


                # os.makedirs(os.path.join(args.save_dir, 'mean', sample['paths'][i].split('train')[1].split('/')[1]), exist_ok=True)
                # mean_save_path = os.path.join(args.save_dir, 'mean', sample['paths'][i].split('train')[1].split('/')[1], sample['paths'][i].split('train')[1].split('/')[3].split('.')[0]+'.pt') 
                # torch.save(latent.latent_dist.mean, mean_save_path) 

            # ViT
            # inputs = feature_extractor(images=sample['images'], return_tensors="pt").to(device)
            # # 对图像编码，得到图像嵌入
            # outputs = model(**inputs)
            # image_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token embedding

            # VAE
            # images = sample['images'].to(device)
            # latent = pipe.vae.encode(images.to(device, dtype=torch.float16) * 2 - 1)
            # ls = 0.18215 * latent.latent_dist.sample()

            # for i in range(latent.latent_dist.mean.shape[0]):
            #     # latent = ls[i] 
            #     # os.makedirs(os.path.join(args.save_dir, sample['paths'][i].split('train')[1].split('/')[1]), exist_ok=True)
            #     # save_path = os.path.join(args.save_dir, sample['paths'][i].split('train')[1].split('/')[1], sample['paths'][i].split('train')[1].split('/')[3].split('.')[0]+'.pt') 
            #     torch.save(latent.latent_dist.mean, mean_save_path) 
                # os.makedirs(os.path.join(args.save_dir, 'mean', sample['paths'][i].split('train')[1].split('/')[1]), exist_ok=True)
                # mean_save_path = os.path.join(args.save_dir, 'mean', sample['paths'][i].split('train')[1].split('/')[1], sample['paths'][i].split('train')[1].split('/')[3].split('.')[0]+'.pt') 
                # torch.save(latent.latent_dist.mean, mean_save_path) 



if __name__ == "__main__":
    main()