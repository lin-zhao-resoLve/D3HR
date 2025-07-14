import math
import time
import random
import numpy as np
from tqdm import tqdm
from torchvision.models import resnet18
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
import torchvision
from utils.validate_utils import (
    mix_aug,
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
    checkpoint0 = "/scratch/zhao.lin1/in_trained_models/trained_model_seed_0_checkpoint_final.pt"
    checkpoint1 = "/scratch/zhao.lin1/in_trained_models/trained_model_seed_1_checkpoint_final.pt"
    checkpoint2 = "/scratch/zhao.lin1/in_trained_models/trained_model_seed_2_checkpoint_final.pt"
    checkpoint3 = "/scratch/zhao.lin1/in_trained_models/trained_model_seed_3_checkpoint_final.pt"
    checkpoint4 = "/scratch/zhao.lin1/in_trained_models/trained_model_seed_4_checkpoint_final.pt"
    teacher_model0 = torchvision.models.get_model('resnet18', num_classes=1000, weights = None).to('cuda')
    teacher_model1 = torchvision.models.get_model('resnet18', num_classes=1000, weights = None).to('cuda')
    teacher_model2 = torchvision.models.get_model('resnet18', num_classes=1000, weights = None).to('cuda')
    teacher_model3 = torchvision.models.get_model('resnet18', num_classes=1000, weights = None).to('cuda')
    teacher_model4 = torchvision.models.get_model('resnet18', num_classes=1000, weights = None).to('cuda')

    teacher_model0.load_state_dict(torch.load(checkpoint0))
    teacher_model1.load_state_dict(torch.load(checkpoint1))
    teacher_model2.load_state_dict(torch.load(checkpoint2))
    teacher_model3.load_state_dict(torch.load(checkpoint3))
    teacher_model4.load_state_dict(torch.load(checkpoint4))

    student_model = load_model(
        model_name=args.stud_name,
        pretrained=False,
        # classes=train_dataset.original_classes,
        input_size=args.input_size
    ).cuda()
    # teacher_model = torch.nn.DataParallel(teacher_model).cuda()
    # student_model = torch.nn.DataParallel(student_model).cuda()

    teacher_model0.eval()
    teacher_model1.eval()
    teacher_model2.eval()
    teacher_model3.eval()
    teacher_model4.eval()
    student_model.train()

    # freeze all layers
    for param in teacher_model0.parameters():
        param.requires_grad = False
    for param in teacher_model1.parameters():
        param.requires_grad = False
    for param in teacher_model2.parameters():
        param.requires_grad = False
    for param in teacher_model3.parameters():
        param.requires_grad = False
    for param in teacher_model4.parameters():
        param.requires_grad = False

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
    teacher_models = [torch.jit.script(teacher_model0),torch.jit.script(teacher_model1),torch.jit.script(teacher_model2),torch.jit.script(teacher_model3),torch.jit.script(teacher_model4)]

    for epoch in tqdm(range(args.epochs)):
        train_epoch(epoch, train_loader, teacher_models, student_model, args)

        # if epoch % 10 == 9 or epoch == args.epochs - 1:
        #     if epoch > args.epochs * 0.8:
        #         top1 = validate(student_model, args, epoch)
        #     else:
        #         top1 = 0
        # else:
        #     top1 = 0

        if epoch > 280:
                top1 = validate(student_model, args, epoch)
        else:
            top1 = 0
        # print(top1)
        scheduler.step()
        if top1 > best_acc1:
            best_acc1 = max(top1, best_acc1)
            best_epoch = epoch

    print(f"Train Finish! Best accuracy is {best_acc1}@{best_epoch}")

    return best_acc1


def train_epoch(epoch, train_loader, teacher_models, student_model, args):
    """Generate soft labels and train"""
    objs = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()

    optimizer = args.optimizer
    loss_function_kl = nn.KLDivLoss(reduction="batchmean")
    student_model.train()
    t1 = time.time()
    for batch_idx, (images, labels) in enumerate(train_loader):
        with torch.no_grad():
            images = images.cuda()
            labels = labels.cuda()

            mix_images = images
            mix_images, _, _, _ = mix_aug(images, args)



            pred_label = student_model(images)
            soft_mix_label = torch.zeros(images.shape[0], 1000).cuda()
            for teacher_model in teacher_models:
                soft_mix_label += teacher_model(mix_images)/len(teacher_models)
            soft_mix_label = F.softmax(soft_mix_label / args.temperature, dim=1)

        if batch_idx % args.accum_steps == 0:
            optimizer.zero_grad()

        # prec1, prec5 = accuracy(pred_label, labels, topk=(1))

        pred_mix_label = student_model(mix_images)

        soft_pred_mix_label = F.log_softmax(pred_mix_label / args.temperature, dim=1)
        loss = loss_function_kl(soft_pred_mix_label.squeeze(), soft_mix_label)
        # loss = -torch.sum(soft_mix_label * pred_mix_label) / pred_mix_label.shape[0] * (args.temperature**2)
        # loss += torch.sum(soft_mix_label * soft_pred_mix_label) / pred_mix_label.shape[0] * (args.temperature**2)

        loss = loss / args.accum_steps

        loss.backward()
        if batch_idx % args.accum_steps == (args.accum_steps - 1):
            optimizer.step()

        n = images.size(0)
        objs.update(loss.item(), n)
        # top1.update(prec1.item(), n)
        # top5.update(prec5.item(), n)

    printInfo = (
        "TRAIN Iter {}: loss = {:.6f},\t".format(epoch, objs.avg)
        # + "Top-1 err = {:.6f},\t".format(100 - top1.avg)
        # + "Top-5 err = {:.6f},\t".format(100 - top5.avg)
        + "train_time = {:.6f}".format((time.time() - t1))
    )
    if args.verbose:
        print(printInfo)
    t1 = time.time()


def validate(model, args, epoch=None):
    # objs = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()
    # loss_function = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        for data, target in args.val_loader:
            target = target.type(torch.LongTensor)
            data, target = data.cuda(), target.cuda()
            output = model(data)
            # loss = loss_function(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            # objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            # top5.update(prec5.item(), n)

    # logInfo = (
    #     "TEST:\nIter {}: loss = {:.6f},\t".format(epoch, objs.avg)
    #     + "Top-1 err = {:.6f},\t".format(100 - top1.avg)
    #     + "Top-5 err = {:.6f},\t".format(100 - top5.avg)
    #     + "val_time = {:.6f}".format(time.time() - t1)
    # )
    # if args.verbose:
    #     print(logInfo)
    return top1.avg


if __name__ == "__main__":
    main(args)
