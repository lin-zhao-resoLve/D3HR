import math
import time
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.nn.functional as F
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR

from argument import args
from models import load_model
from utils.data_utils import load_data
from utils.validate_utils import (
    AverageMeter,
    accuracy,
    get_parameters,
)


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    # data
    print('phase:', args.phase)
    print("process data from {}".format(args.data_dir[0]))

    train_dataset, train_loader, val_loader = load_data(args, mem_flag=False)

    best_acc_l = []
    for i in range(args.repeat):
        print(f'Repeat: {i + 1} / {args.repeat}')
        best_acc = train(args, train_dataset, train_loader, val_loader)
        best_acc_l.append(best_acc)

    print(f'[Repeat {args.repeat}] Mean, Std acc: {np.mean(best_acc_l):.1f} {np.std(best_acc_l):.1f}')


def train(args, train_dataset, train_loader, val_loader):
    # model
    print("=> using pytorch pre-trained teacher model '{}'".format(args.arch_name))

    student_model = load_model(
        model_name=args.stud_name,
        pretrained=False,
        dataset=args.subset,
        classes=args.classes,
        input_size=args.input_size,
    )
    student_model = torch.nn.DataParallel(student_model).cuda()

    student_model.train()

    # optimizer
    if args.sgd:
        optimizer = torch.optim.SGD(
            get_parameters(student_model),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            get_parameters(student_model),
            lr=args.adamw_lr,
            betas=[0.9, 0.999],
            weight_decay=args.adamw_weight_decay,
        )

    # lr scheduler
    if args.cos == True:
        scheduler = LambdaLR(
            optimizer,
            lambda step: 0.5 * (1.0 + math.cos(math.pi * step / args.epochs / 2))
            if step <= args.epochs
            else 0,
            last_epoch=-1,
        )
    else:
        scheduler = LambdaLR(
            optimizer,
            lambda step: (1.0 - step / args.epochs) if step <= args.epochs else 0,
            last_epoch=-1,
        )

    best_acc1 = 0
    best_epoch = 0
    args.optimizer = optimizer
    args.scheduler = scheduler
    args.train_loader = train_loader
    args.val_loader = val_loader

    for epoch in tqdm(range(args.epochs)):
        train_epoch(epoch, train_loader, student_model, args)

        # if epoch % 10 == 9 or epoch == args.epochs - 1:
        #     if epoch > args.epochs * 0.8:
        #         top1 = validate(student_model, args, epoch)
        #     else:
        #         top1 = 0
        # else:
        #     top1 = 0
        if args.ipc == 10:
            if epoch > 1900:
                top1 = validate(student_model, args, epoch)
            else:
                top1 = 0
        elif args.ipc == 50:
            if epoch > 1400:
                top1 = validate(student_model, args, epoch)  
            else:
                top1 = 0   
        elif args.ipc == 100:
            if epoch > 900:
                top1 = validate(student_model, args, epoch)     
            else:
                top1 = 0
        scheduler.step()
        if top1 > best_acc1:
            best_acc1 = max(top1, best_acc1)
            best_epoch = epoch

    print(f"Train Finish! Best accuracy is {best_acc1}@{best_epoch}")

    return best_acc1


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mixup_data(data, target, device, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = data.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_data = lam * data + (1 - lam) * data[index, :]
    target_a, target_b = target, target[index]
    return mixed_data, target_a, target_b, lam



def mixup_criterion(criterion, pred, target_a, target_b, lam):
    return lam * criterion(pred, target_a) + (1 - lam) * criterion(pred, target_b)



def train_epoch(epoch, train_loader, student_model, args):
    """Generate soft labels and train"""
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    optimizer = args.optimizer
    criterion = nn.CrossEntropyLoss()
    student_model.train()
    t1 = time.time() 
    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.cuda()
        labels = labels.cuda()


        mixed_data, target_a, target_b, lam = mixup_data(images, labels, labels.device, alpha=1.0)


        if batch_idx % args.accum_steps == 0:
            optimizer.zero_grad()

        outputs = student_model(mixed_data)
        # outputs = student_model(images)


        loss = mixup_criterion(criterion, outputs, target_a, target_b, lam)
        # loss = criterion(outputs, labels)

        # loss = -torch.sum(soft_mix_label * pred_mix_label) / pred_mix_label.shape[0] * (args.temperature**2)
        # loss += torch.sum(soft_mix_label * soft_pred_mix_label) / pred_mix_label.shape[0] * (args.temperature**2)

        loss = loss / args.accum_steps

        loss.backward()
        if batch_idx % args.accum_steps == (args.accum_steps - 1):
            optimizer.step()

    t1 = time.time()


def validate(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_function = nn.CrossEntropyLoss()

    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for data, target in args.val_loader:
            target = target.type(torch.LongTensor)
            data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = loss_function(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    logInfo = (
        "TEST:\nIter {}: loss = {:.6f},\t".format(epoch, objs.avg)
        + "Top-1 err = {:.6f},\t".format(100 - top1.avg)
        + "Top-5 err = {:.6f},\t".format(100 - top5.avg)
        + "val_time = {:.6f}".format(time.time() - t1)
    )
    if args.verbose:
        print(logInfo)
    return top1.avg


if __name__ == "__main__":
    main(args)
