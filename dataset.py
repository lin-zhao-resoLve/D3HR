import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import Dataset
import os
from PIL import Image
from datasets import load_dataset
from torchvision.transforms.v2 import Compose, InterpolationMode
from transformers import ViTModel, ViTImageProcessor

class DiffDataset(Dataset):
    def __init__(self, img_dir_path, start, end, transform=None):
        self.img_dir_path = img_dir_path
        self.img_paths = []
        for dir_name in sorted(os.listdir(self.img_dir_path))[start:end]:
            for image_name in sorted(os.listdir(os.path.join(self.img_dir_path,dir_name))):
                self.img_paths.append(os.path.join(self.img_dir_path,dir_name, image_name))
                
        self.corr_label = {}
        for i, dir in enumerate(sorted(os.listdir(self.img_dir_path))):
            self.corr_label[dir] = i 
              
        transform = transforms.Compose([
            transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        self.transform = transform
        

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_class = self.corr_label[img_path.split('/')[-2]]
        image = Image.open(img_path).convert('RGB')  # 打开图像并转换为RGB

        if self.transform:
            image = self.transform(image)

        return image, img_path, img_class



class TinyDataset(Dataset):
    def __init__(self, img_dir_path, start, end, transform=None):
        self.ds = load_dataset(img_dir_path)['train']
        # self.ds_cur = self.ds[start:end]
              
        transform = transforms.Compose([
            transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        self.transform = transform
        

    def __len__(self):
        return len(self.ds['label'])

    def __getitem__(self, idx):
        try:
            img_class = self.ds[idx]['label']
            image = self.ds[idx]['image']
            
            if image.mode != 'RGB':  # 如果图像不是 RGB（比如 'L' 表示灰度图）
                image = image.convert('RGB')

            if self.transform:
                image = self.transform(image)
                
            sample = {'images':image, 
                      'labels':img_class, 
                      'idx':idx}

            return sample
        
        except (IOError, SyntaxError) as e:
            print(f"Skipping corrupted image: {idx}, error: {e}")
            return None
        

class LocalDataset(Dataset):
    def __init__(self, img_dir_path, begin=0, end=0, transform=None):
        self.img_dir_path = img_dir_path
        
        # Load the WNID to index mapping
        # self.wnid_to_index = self.load_mapping(os.path.join(img_dir_path, 'tiny-imagenet-mapping.txt'))
        self.wnid_to_index = self.load_mapping('/home/zhao.lin1/DD-DDIM-Inversion/ds_inf/tiny-imagenet-200/tiny-imagenet-mapping.txt')
        
        self.img_paths = []
        self.img_labels = []
        self.processor = ViTImageProcessor.from_pretrained('/scratch/zhao.lin1/dit_imagenet')

        # Load images from train, val, and test directories
        for split in ['train']:
            split_dir = os.path.join(self.img_dir_path, split)
            if os.path.exists(split_dir):
                for dir_name in sorted(os.listdir(split_dir))[begin:end]:
                    dir_path = os.path.join(split_dir, dir_name,'images')
                    for image_name in sorted(os.listdir(dir_path)):
                        image_path = os.path.join(dir_path, image_name)
                        self.img_paths.append(image_path)
                        self.img_labels.append(self.wnid_to_index[dir_name])  # Map WNID to index
        
        # # Define transformations
        # if transform is None:
        #     self.transform = transforms.Compose([
        #         transforms.Resize(256),
        #         transforms.ToTensor()
        #     ])
        # else:
        #     self.transform = transform
        
        #clip feature extractor
        if transform is None:
            self.transform = Compose(
                    [
                    transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                    transforms.Resize(size=224, antialias=True, interpolation=InterpolationMode.BICUBIC),
                    transforms.CenterCrop(size=(224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                    ]
                )

    def load_mapping(self, mapping_file):
        wnid_to_index = {}
        with open(mapping_file, 'r') as f:
            for line in f:
                wnid, index = line.strip().split('\t')
                wnid_to_index[wnid] = int(index)
        return wnid_to_index

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.img_paths[idx]
            img_label = self.img_labels[idx]
            image = Image.open(img_path)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = self.processor(images=image, return_tensors="pt")
            # if self.transform:
            #     image = self.transform(image)
            
                
            sample = {
                'images': image,
                'paths': img_path,
                'labels': img_label,
                'idx': idx
            }

            return sample
        
        except (IOError, SyntaxError) as e:
            print(f"Skipping corrupted image: {idx}, error: {e}")
            return None


# def load_dataset(args):
#     # Obtain dataloader
#     transform_train = transforms.Compose([
#         transforms.Resize((args.size, args.size)),
#         transforms.ToTensor(),
#     ])
#     if args.dataset == 'cifar10':
#         transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.491, 0.482, 0.447), (0.202, 0.199, 0.201))
#         ])
#         trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=False,
#                                     transform=transform_train)
#         testset = datasets.CIFAR10(root=args.data_dir, train=False, download=False,
#                                    transform=transform_test)
#     elif args.dataset == 'cifar100':
#         transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.507, 0.486, 0.441), (0.267, 0.256, 0.276))
#         ])
#         trainset = datasets.CIFAR100(root=args.data_dir, train=True, download=False,
#                                     transform=transform_train)
#         testset = datasets.CIFAR100(root=args.data_dir, train=False, download=False,
#                                    transform=transform_test)  
#     elif args.dataset == 'imagenet_1k':
#         transform_test = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor()
#         ])
#         trainset = datasets.ImageFolder(root=args.data_dir + "/train", 
#                                         transform=transform_train)
#         testset = datasets.ImageFolder(root=args.data_dir + "/val", 
#                                        transform=transform_train)
#     elif args.dataset == 'tiny_imagenet':
#         transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
#         trainset = datasets.ImageFolder(root=args.data_dir + "/train", 
#                                         transform=transform_train)
#         testset = datasets.ImageFolder(root=args.data_dir + "/val", 
#                                        transform=transform_train)

#     trainloader = torch.utils.data.DataLoader(
#         trainset, batch_size=args.batch_size, shuffle=True,
#         num_workers=args.num_workers, drop_last=False
#     )
#     testloader = torch.utils.data.DataLoader(
#         testset, batch_size=args.batch_size, shuffle=False,
#         num_workers=args.num_workers
#     )

#     return trainloader, testloader



# if __name__ == "__main__":
    