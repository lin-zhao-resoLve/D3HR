import datetime
import os
import json
import time
import warnings
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import transforms as mix_transforms
import utils
import wandb
from torch.utils.data import Dataset, DataLoader
from sampler import RASampler
from tiny_imagenet_dataset import TinyImageNet
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
import random
from PIL import Image
import ipdb


def load_mapping(mapping_file):
    new_mapping = {}
    with open(mapping_file, 'r') as file:
        for index, line in enumerate(file):
            # 提取每一行的编号（n开头部分）并将行号-1
            key = line.split()[0]
            new_mapping[key] = index 
    return new_mapping

class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, mapping_file, json_file='', split='train', transform=None, random_selection=False):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load class mapping and json file
        self.wnid_to_index = load_mapping(mapping_file)
        if json_file != '':
            self._load_from_json(json_file)
        
        elif split == 'train':
            self._load_train_data(random_selection)
        elif split == 'val':
            self._load_val_data()
        else:
            self._load_test_data()

    def _load_from_json(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        for class_name, image_files in data.items():
            if class_name not in self.wnid_to_index:
                continue
            
            class_index = self.wnid_to_index[class_name]
            class_path = os.path.join(self.root_dir, class_name, 'images')
            
            # Iterate over the image files by key50_far_latents_train_vit_image_feature_CLS.json
            if "key50" in json_file:
                for img_file in image_files:
                    img_path = os.path.join(class_path, img_file)
                    if os.path.exists(img_path):
                        self.images.append(img_path)
                        self.labels.append(class_index)
            elif "cluster" in json_file:
            
                # Iterate over the image files (which are dictionaries containing 'index', 'file_name', 'distance')
                for img_info in image_files:
                    img_file = img_info["file_name"]  # Extract the file name from the dictionary
                    img_path = os.path.join(class_path, img_file)
                    # Check if the image file exists
                    if os.path.exists(img_path):
                        self.images.append(img_path)  # Add the image path to the list
                        self.labels.append(class_index)  # Add the corresponding class index

    def _load_train_data(self, random_selection):
        class_dirs = sorted(os.listdir(self.root_dir))
        
        for class_name in class_dirs:
            if class_name not in self.wnid_to_index:
                continue
                
            class_path = os.path.join(self.root_dir,  class_name, 'images')
            if not os.path.exists(class_path):
                continue
                
            image_files = sorted(os.listdir(class_path))
            
            if random_selection:
                # Randomly select 10 images for this class
                image_files = random.sample(image_files, min(50, len(image_files)))
            
            for img_file in image_files:
                self.images.append(os.path.join(class_path, img_file))
                self.labels.append(self.wnid_to_index[class_name])

    def _load_val_data(self):
        # Load validation annotations
        val_annotations_path = os.path.join(os.path.dirname(self.root_dir),'val', 'val_annotations.txt')
        img_to_class = {}
        
        with open(val_annotations_path, 'r') as f:
            for line in f:
                img_file, class_name, *_ = line.strip().split('\t')
                img_to_class[img_file] = class_name
        
        images_dir = os.path.join(self.root_dir, 'images')
        image_files = sorted(os.listdir(images_dir))
        
        for img_file in image_files:
            if img_file in img_to_class and img_to_class[img_file] in self.wnid_to_index:
                self.images.append(os.path.join(images_dir, img_file))
                self.labels.append(self.wnid_to_index[img_to_class[img_file]])

    def _load_test_data(self):
        # Load test annotations
        test_annotations_path = os.path.join(os.path.dirname(self.root_dir), 'test', 'test_annotations.txt')
        img_to_class = {}
        
        with open(test_annotations_path, 'r') as f:
            for line in f:
                img_file, class_name, *_ = line.strip().split('\t')
                img_to_class[img_file] = class_name
        
        images_dir = os.path.join(self.root_dir, 'images')
        image_files = sorted(os.listdir(images_dir))
        
        for img_file in image_files:
            if img_file in img_to_class and img_to_class[img_file] in self.wnid_to_index:
                self.images.append(os.path.join(images_dir, img_file))
                self.labels.append(self.wnid_to_index[img_to_class[img_file]])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image in case of error
            image = Image.new('RGB', (64, 64))
            
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms(is_training):
    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2302, 0.2265, 0.2262])
    if is_training:
        # return transforms.Compose([
        #     transforms.RandomResizedCrop(224),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        #     transforms.RandomRotation(10),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
        return transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # return transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
        return transforms.Compose([
        transforms.ToTensor(),
        normalize,
        ])

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")

def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, args):
    mapping_file = "/home/zhao.lin1/DD-DDIM-Inversion/ds_inf/tiny-imagenet-200/tiny-imagenet-mapping.txt" 

    train_transform = get_transforms(is_training=True)
    eval_transform = get_transforms(is_training=False)


    # Distilled dataset
    print("Loading distilled dataset")
    # json_dir = '/home/zhao.lin1/DD-DDIM-Inversion/ds_inf/key50_far_latents_train_vit_image_feature_CLS.json'
    # json_dir = ''
    # json_dir = '/home/zhao.lin1/DD-DDIM-Inversion/ds_inf/vit_Kmeans++_farthest_1_cluster_per_class.json'
    # json_dir = '/home/zhao.lin1/DD-DDIM-Inversion/ds_inf/vit_Kmeans++_farthest_50_cluster_per_class.json'
    # json_dir = '/home/zhao.lin1/DD-DDIM-Inversion/ds_inf/vit_Kmeans++_closest_1_cluster_per_class.json'
    # json_dir = '/home/zhao.lin1/DD-DDIM-Inversion/ds_inf/vit_Kmeans++_closest_50_cluster_per_class.json'
    json_dir = '/home/zhao.lin1/DD-DDIM-Inversion/ds_inf/key50_far_latent_vit_not_force_true_label.json'
    dataset = TinyImageNetDataset(traindir, mapping_file, json_file=json_dir, split='train', transform=train_transform, random_selection=False)
    

    # Random dataset
    # print("Loading random dataset")
    # dataset = TinyImageNetDataset(traindir, mapping_file, json_file, split='train', transform=train_transform, random_selection=True)
    
    # Test dataset
    val_dir = "/scratch/zhao.lin1/dataset/tiny-imagenet-200/val"
    print("Loading validation dataset")
    dataset_test = TinyImageNetDataset(val_dir, mapping_file, json_file='', split='val', transform=eval_transform)

    print("Creating data loaders")
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args):
    global best_acc1
    best_acc1 = 0
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    collate_fn = None
    num_classes = 200
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(mix_transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(mix_transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    print("Creating model")
    model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    model.maxpool = nn.Identity()
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        else:
            evaluate(model, criterion, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    from tqdm import tqdm
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler)
        lr_scheduler.step()
        acc1 = evaluate(model, criterion, data_loader_test, device=device)
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            if args.save_all:
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))
            if acc1 > best_acc1:
                best_acc1 = max(acc1, best_acc1)
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint_best.pth"))
    wandb.log({'last_acc1': acc1})
    wandb.log({"best_acc1": best_acc1})
    print(f"Best Accuracy {best_acc1:.3f}")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training on Tiny ImageNet", add_help=add_help)
    # distilled_dir = "/home/user1/workspace/leilu/linzhao/dataset/tiny-imagenet-200/key10-real_mindis_images_train_ddim_inversion_s20_kmeasns++"
    # random_10_dir = "/home/user1/workspace/leilu/linzhao/dataset/tiny-imagenet-200/random_10_per_class"
    
    parser.add_argument("--data-path", default="/scratch/zhao.lin1/dataset/tiny-imagenet-200", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=100, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=200, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=1000, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--save-all", action="store_true", help="save checkpoint for all epochs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load, must be None to load 200 classes model")
    parser.add_argument("--wandb-key", default=None, type=str, help="the key for wandb login")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    wandb.login(key=args.wandb_key)
    wandb.init(project='train_tiny', name=args.output_dir)
    main(args)
    wandb.finish()
