import os
import random
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import json
from torch.utils.data import Dataset

    
def find_subclasses(spec, nclass, phase=0):
    classes = []
    cls_from = nclass * phase
    cls_to = nclass * (phase + 1)
    if spec == 'woof':
        file_list = './misc/class_woof.txt'
    elif spec == 'im100':
        file_list = './misc/class_100.txt'
    else:
        file_list = './misc/class_indices.txt'
    with open(file_list, 'r') as f:
        class_name = f.readlines()
    for c in class_name:
        c = c.split('\n')[0]
        classes.append(c)
    classes = classes[cls_from:cls_to]
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    return classes, class_to_idx


def find_original_classes(spec, classes):
    file_list = './misc/class_indices.txt'
    with open(file_list, 'r') as f:
        all_classes = f.readlines()
    all_classes = [class_name.split('\n')[0] for class_name in all_classes]
    original_classes = []
    for class_name in classes:
        original_classes.append(all_classes.index(class_name))
    return original_classes


def load_mapping_imgwoof(mapping_file, names):
    new_mapping = {}
    with open(mapping_file, 'r') as file:
        data = json.load(file)
        if "tiny" in mapping_file:
            for index, line in enumerate(file):
                # 提取每一行的编号（n开头部分）并将行号-1
                key = line.split()[0]
                new_mapping[key] = index 
        else:
            new_mapping = {item["wnid"]: names.index(item["name"]) for item in data.values() if item['name'] in names}
    return new_mapping


def load_mapping(mapping_file):
    new_mapping = {}
    with open(mapping_file, 'r') as file:
        data = json.load(file)
        if "tiny" in mapping_file:
            for index, line in enumerate(file):
                # 提取每一行的编号（n开头部分）并将行号-1
                key = line.split()[0]
                new_mapping[key] = index 
        else:
            new_mapping = {item["wnid"]: item["index"] for item in data.values()}
    return new_mapping



def load_mapping_txt(mapping_file):
        wnid_to_index = {}
        with open(mapping_file, 'r') as f:
            for line in f:
                wnid, index = line.strip().split('\t')
                wnid_to_index[wnid] = int(index)
        return wnid_to_index

def find_classes(class_file):
    with open(class_file) as r:
        classes = list(map(lambda s: s.strip(), r.readlines()))

    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return class_to_idx

class ImageFolder(Dataset):
    def __init__(self, split=None, txt_file=None, subset=None, mapping_file=None, transform=None):
        super(ImageFolder, self).__init__()
        self.split = split
        self.image_paths = []
        self.targets = []
        self.samples = []
        self.subset = subset
        if self.subset == 'imagenet_1k':
            self.wnid_to_index = load_mapping(mapping_file)
        elif self.subset == 'tinyimagenet':
            self.wnid_to_index = find_classes(mapping_file)
        if split == 'train':
            self._load_from_txt(txt_file)
        else:
            self._load_from_txt(txt_file)
        self.transform = transform


    def _load_from_txt(self, txt_file):
        with open(txt_file, "r") as file:
            image_paths = file.readlines()

        # 去掉每行的换行符
        self.image_paths = [path.strip() for path in image_paths]
        for path in self.image_paths:
            self.samples.append(path)
            if self.subset == 'cifar10' or self.subset == 'cifar100':
                class_index = int(path.split('/')[-2][-3:])
            else:
                # if self.split == 'test':
                #     class_index = self.wnid_to_index[path.split('/')[-2]]
                # elif self.split == 'train':
                class_index = self.wnid_to_index[path.split('/')[-2]]
            self.targets.append(class_index)


    #combine ten txt
    def _load_from_txt_1(self, txt_file):

        image_paths_10 = []
        for kk in range(10):
            txt_file=f'/scratch/zhao.lin1/tinyimagenet_finetune_start_step_18_ddim_inversion_10_min_images_{kk}/train.txt'
            with open(txt_file, "r") as file:
                image_paths = file.readlines()

            image_paths_10.append([path.strip() for path in image_paths])


        for kk in range(len(image_paths)):
            number = random.randint(0, 9)
            self.image_paths.append(image_paths_10[number][kk])
            if self.subset == 'cifar10' or self.subset == 'cifar100':
                class_index = int(path.split('/')[-2][-3:])
            else:
                # if self.split == 'test':
                #     class_index = self.wnid_to_index[path.split('/')[-2]]
                # elif self.split == 'train':
                class_index = self.wnid_to_index[image_paths_10[number][kk].split('/')[-2]]
            self.targets.append(class_index)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        try:
            sample = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image in case of error
            sample = Image.new('RGB', (256, 256))
        sample = self.transform(sample)
        # class_dir = img_path.split('/')[-2]
        return sample, self.targets[index]

    def __len__(self):
        return len(self.targets)


class Imagewoof(Dataset):
    def __init__(self, split=None, txt_file=None, subset=None, mapping_file=None, transform=None):
        super(Imagewoof, self).__init__()
        self.split = split
        self.image_paths = []
        self.targets = []
        self.samples = []
        self.subset = subset
        self.names = ["Australian_terrier", "Border_terrier", "Samoyed", "beagle", "Shih-Tzu", "English_foxhound", "Rhodesian_ridgeback", "dingo", "golden_retriever", "Old_English_sheepdog"]
        self.wnid_to_index = load_mapping_imgwoof(mapping_file, self.names)
        self._load_from_txt(txt_file)
        self.transform = transform


    def _load_from_txt(self, txt_file):
        with open(txt_file, "r") as file:
            image_paths = file.readlines()

        # 去掉每行的换行符
        image_paths = [path.strip() for path in image_paths]
        for path in image_paths:
            self.samples.append(path)
            if self.subset == 'cifar10' or self.subset == 'cifar100':
                class_index = int(path.split('/')[-2][-3:])
            else:
                # if self.split == 'test':
                #     class_index = self.wnid_to_index[path.split('/')[-2]]
                # elif self.split == 'train':
                if path.split('/')[-2] in list(self.wnid_to_index.keys()):
                    class_index = self.wnid_to_index[path.split('/')[-2]]
                    self.image_paths.append(path)
                    self.targets.append(class_index)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        try:
            sample = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image in case of error
            sample = Image.new('RGB', (256, 256))
        sample = self.transform(sample)
        # class_dir = img_path.split('/')[-2]
        return sample, self.targets[index]

    def __len__(self):
        return len(self.targets)



# class ImageFolder(torchvision.datasets.ImageFolder):
#     def __init__(self, nclass, ipc, mem=False, spec='none', phase=0, **kwargs):
#         super(ImageFolder, self).__init__(**kwargs)
#         self.mem = mem
#         self.spec = spec
#         self.classes, self.class_to_idx = find_subclasses(
#             spec=spec, nclass=nclass, phase=phase
#         )
#         self.original_classes = find_original_classes(spec=self.spec, classes=self.classes)
#         self.samples, self.targets = self.load_subset(ipc=ipc)
#         if self.mem:
#             self.samples = [self.loader(path) for path in self.samples]

#     def load_subset(self, ipc=-1):
#         all_samples = torchvision.datasets.folder.make_dataset(
#             self.root, self.class_to_idx, self.extensions
#         )
#         samples = np.array([item[0] for item in all_samples])
#         targets = np.array([item[1] for item in all_samples])

#         if ipc == -1:
#             return samples, targets
#         else:
#             sub_samples = []
#             sub_targets = []
#             for c in range(len(self.classes)):
#                 c_indices = np.where(targets == c)[0]
#                 #random.shuffle(c_indices)
#                 sub_samples.extend(samples[c_indices[:ipc]])
#                 sub_targets.extend(targets[c_indices[:ipc]])
#             return sub_samples, sub_targets

#     def __getitem__(self, index):
#         if self.mem:
#             sample = self.samples[index]
#         else:
#             sample = self.loader(self.samples[index])
#         sample = self.transform(sample)
#         return sample, self.targets[index]

#     def __len__(self):
#         return len(self.targets)
def random_stitch_crop_4(image):
    """随机从 stitch 的四个子区域中裁剪一个"""
    w, h = image.size  # 获取图像的宽和高
    w_half, h_half = w // 2, h // 2
    
    # 定义四个区域的坐标
    regions = [
        (0, 0, w_half, h_half),  # 左上
        (w_half, 0, w, h_half),  # 右上
        (0, h_half, w_half, h),  # 左下
        (w_half, h_half, w, h),  # 右下
    ]
    
    # 随机选择一个区域
    x1, y1, x2, y2 = random.choice(regions)
    return image.crop((x1, y1, x2, y2))  # 裁剪并返回

def transform_imagenet(args):
    resize_test = [transforms.Resize(args.input_size // 7 * 8), transforms.CenterCrop(args.input_size)]
    # resize_test = [transforms.Resize(args.input_size), transforms.CenterCrop(args.input_size)]

    cast = [transforms.ToTensor()]

    aug = [
        # transforms.Resize(224), 
        # transforms.Lambda(random_stitch_crop_4),
        # ShufflePatches(args.factor),
        transforms.RandomResizedCrop(
            size=args.input_size,
            # scale=(0.5, 1.0),
            # scale=(1 / args.factor, args.max_scale_crops),
            scale=(0.08, args.max_scale_crops),
            antialias=True,
        ),
        transforms.RandomHorizontalFlip()
    ]

    normalize = [transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )]

    train_transform = transforms.Compose(aug +cast+ normalize)
    test_transform = transforms.Compose(resize_test + cast + normalize)

    return train_transform, test_transform


sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def load_data(args, coreset=False, resize_only=False, mem_flag=True, trainset_only=False):
    train_transform, test_transform = transform_imagenet(args)
    # if len(args.data_dir) == 1:
    #     train_dir = os.path.join(args.data_dir[0], 'train')
    #     val_dir = os.path.join(args.data_dir[0], 'val')
    # else:
    #     train_dir = args.data_dir[0]
    #     val_dir = os.path.join(args.data_dir[1], 'val')

    if resize_only:
        train_transform = transforms.Compose([
            transforms.Resize((512, 512)),
        ])
    elif coreset:
        train_transform = test_transform

    # train_dataset = ImageFolder(
    #     nclass=args.nclass,
    #     ipc=args.ipc,
    #     mem=mem_flag,
    #     spec=args.spec,
    #     phase=args.phase,
    #     root=train_dir,
    #     transform=train_transform,
    # )

    if args.subset == 'imagewoof':
        # Imagewoor
        train_dataset = Imagewoof(
            split = 'train',
            txt_file=args.txt_file,
            mapping_file=args.mapping_file,
            subset = args.subset,
            transform=train_transform,
        )  
    else:
        train_dataset = ImageFolder(
            split = 'train',
            txt_file=args.txt_file,
            mapping_file=args.mapping_file,
            subset = args.subset,
            transform=train_transform,
        )



    if trainset_only:
        return train_dataset

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=24,
        pin_memory=True,
        worker_init_fn=set_worker_sharing_strategy,
    )
    if args.subset == 'cifar10':
        val_dataset = torchvision.datasets.CIFAR10(root='/scratch/zhao.lin1/', train=False, download=True, transform=test_transform)
    elif args.subset == 'cifar100':
        val_dataset = torchvision.datasets.CIFAR100(root='/scratch/zhao.lin1/', train=False, download=True, transform=test_transform)
    elif args.subset == 'imagewoof':
        val_dataset = Imagewoof(
                split = 'test',
                txt_file=args.val_txt_file,
                mapping_file=args.mapping_file,
                subset = args.subset,
                transform=test_transform,
            )
    else:
        val_dataset = ImageFolder(
                split = 'test',
                txt_file=args.val_txt_file,
                mapping_file=args.mapping_file,
                subset = args.subset,
                transform=test_transform,
            )



    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=24,
        pin_memory=True,
        worker_init_fn=set_worker_sharing_strategy,
    )
    print("load data successfully")

    return train_dataset, train_loader, val_loader


class ShufflePatches(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def shuffle_weight(self, img, factor):
        h, w = img.shape[1:]
        tw = w // factor
        patches = []
        for i in range(factor):
            i = i * tw
            if i != factor - 1:
                patches.append(img[..., i : i + tw])
            else:
                patches.append(img[..., i:])
        random.shuffle(patches)
        img = torch.cat(patches, -1)
        return img

    def forward(self, img):
        img = self.shuffle_weight(img, self.factor)
        img = img.permute(0, 2, 1)
        img = self.shuffle_weight(img, self.factor)
        img = img.permute(0, 2, 1)
        return img
