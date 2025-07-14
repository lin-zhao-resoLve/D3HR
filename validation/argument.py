import os
import argparse


def str2bool(v):
    """Cast string to boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser("EEF")
parser.add_argument(
    "--arch-name",
    type=str,
    default="resnet18",
    help="arch name from pretrained torchvision models",
)
parser.add_argument(
    "--subset",
    type=str,
    default="imagenet-1k",
)
parser.add_argument(
    "--spec",
    type=str,
    default="none",
)
parser.add_argument(
    "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
)
parser.add_argument(
    "--data-dir",
    nargs='+',
    default=["../data/imagenet"],
    help="path to imagenet dataset",
)
parser.add_argument(
    "--nclass",
    type=int,
    default=10,
    help="number of classes for synthesis or validation",
)
parser.add_argument(
    "--ipc",
    type=int,
    default=10,
    help="number of images per class for synthesis or validation",
)
parser.add_argument(
    "--target-ipc",
    type=int,
    default=50,
    help="number of images per class for synthesis or validation",
)
parser.add_argument(
    "--phase",
    type=int,
    default=0,
)
parser.add_argument(
    "--input-size",
    default=224,
    type=int,
    metavar="S",
)
parser.add_argument(
    "--save-size",
    default=224,
    type=int,
    metavar="S",
)
parser.add_argument(
    "--repeat",
    default=1,
    type=int,
    help="Repeat times for the validation"
)
parser.add_argument(
    "--factor",
    default=2,
    type=int,
)
parser.add_argument(
    "--batch-size", default=0, type=int, metavar="N"
)
parser.add_argument(
    "--accum-steps",
    type=int,
    default=1,
    help="gradient accumulation steps for small gpu memory",
)
parser.add_argument(
    "--mix-type",
    default="cutmix",
    type=str,
    choices=["mixup", "cutmix", None],
    help="mixup or cutmix or None",
)
parser.add_argument(
    "--stud-name",
    type=str,
    default="resnet18",
    help="arch name from torchvision models",
)
parser.add_argument(
    "--workers",
    default=24,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--temperature",
    type=float,
    help="temperature for distillation loss",
)
parser.add_argument(
    "--min-scale-crops", type=float, default=0.08, help="argument in RandomResizedCrop"
)
parser.add_argument(
    "--max-scale-crops", type=float, default=1, help="argument in RandomResizedCrop"
)
parser.add_argument("--epochs", default=300, type=int)
parser.add_argument(
    "--results-dir",
    type=str,
    default="results",
    help="where to store synthetic data",
)
parser.add_argument(
    "--seed", default=42, type=int, help="seed for initializing training. "
)
parser.add_argument(
    "--mixup",
    type=float,
    default=0.8,
    help="mixup alpha, mixup enabled if > 0. (default: 0.8)",
)
parser.add_argument(
    "--cutmix",
    type=float,
    default=1.0,
    help="cutmix alpha, cutmix enabled if > 0. (default: 1.0)",
)
parser.add_argument("--cos", default=True, help="cosine lr scheduler")
parser.add_argument("--verbose", type=str2bool, default=False)
parser.add_argument("--mapping_file", default="ds_inf/imagenet_1k_mapping.json", type=str)
parser.add_argument("--txt_file", default='/home/zhao.lin1/DD-DDIM-inversion/ds_inf/imagenet-1k/biggest_20%_ipc_for_all_1k.txt', type=str)
parser.add_argument("--val_txt_file", default='/home/zhao.lin1/CONCORD/val.txt', type=str)
# diffusion
parser.add_argument("--dit-model", default='DiT-XL/2')
parser.add_argument("--ckpt", type=str, default='pretrained_models/DiT-XL-2-256x256.pt',
                    help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
parser.add_argument("--dit-image-size", default=256, type=int)
parser.add_argument("--num-dit-classes", default=1000, type=int)
parser.add_argument("--diffusion-steps", default=1000, type=int)
parser.add_argument("--cfg-scale", type=float, default=4.0)

parser.add_argument("--vae-path", default='stabilityai/sd-vae-ft-ema')

# distillation
parser.add_argument("--save-path", default='./results/test')
parser.add_argument("--description-path", default='./misc/class_description.json')
parser.add_argument("--clip-alpha", type=float, default=10.0)
parser.add_argument("--cls-alpha", type=float, default=10.0)
parser.add_argument("--num-neg-samples", type=int, default=5)
parser.add_argument("--neg-policy", type=str, default="weighted")

# sgd
parser.add_argument("--sgd", default=False, action="store_true", help="sgd optimizer")
parser.add_argument(
    "-lr",
    "--learning-rate",
    type=float,
    default=0.1,
    help="sgd init learning rate",
)
parser.add_argument("--momentum", type=float, default=0.9, help="sgd momentum")
parser.add_argument("--weight-decay", type=float, default=1e-4, help="sgd weight decay")

# adamw
parser.add_argument("--adamw-lr", type=float, default=0, help="adamw learning rate")
parser.add_argument(
    "--adamw-weight-decay", type=float, default=0.01, help="adamw weight decay"
)
parser.add_argument(
    "--exp-name",
    type=str,
    help="name of the experiment, subfolder under syn_data_path",
)
args = parser.parse_args()

# temperature
if args.mix_type == "mixup":
    args.temperature = 4
elif args.mix_type == "cutmix":
    args.temperature = 20

if args.subset == "imagenet_1k":
    args.nclass = 1000
    args.classes = range(args.nclass)
    args.val_ipc = 50
    args.input_size = 224

elif args.subset == "imagewoof":
    args.nclass = 10
    args.classes = range(args.nclass)
    args.val_ipc = 50
    args.input_size = 224
    if args.ipc == 10:
        args.epochs = 2000
    elif args.ipc == 50:
        args.epochs = 1500
    else:
        args.epochs = 1000


elif args.subset == "cifar10":
    args.nclass = 10
    args.classes = range(args.nclass)
    args.val_ipc = 1000
    args.input_size = 32
    args.epochs = 1000

elif args.subset == "cifar100":
    args.nclass = 100
    args.classes = range(args.nclass)
    args.val_ipc = 100
    args.input_size = 32
    args.epochs = 400

elif args.subset == "tinyimagenet":
    args.nclass = 200
    args.classes = range(args.nclass)
    args.val_ipc = 50
    args.input_size = 64
    args.epochs = 300

# set up batch size
if args.batch_size == 0:
    if args.ipc >= 50:
        args.batch_size = 100
    elif args.ipc >= 10:
        args.batch_size = 50
    elif args.ipc > 0:
        args.batch_size = 15
    elif args.ipc == -1:
        args.batch_size = 100

    if args.nclass == 10:
        args.batch_size *= 1
    if args.nclass == 100:
        args.batch_size *= 2
    # if args.nclass == 1000:
    #     args.batch_size *= 2

# reset batch size below ipc * nclass
if args.ipc != -1 and args.batch_size > args.ipc * args.nclass:
    args.batch_size = int(args.ipc * args.nclass)

# reset batch size with accum_steps
if args.accum_steps != 1:
    args.batch_size = int(args.batch_size / args.accum_steps)

# result dir for saving
args.exp_name = f"{args.spec}_{args.arch_name}_f{args.factor}_ipc{args.ipc}"
if not os.path.exists(f"./exp/{args.exp_name}"):
    os.makedirs(f"./exp/{args.exp_name}")


# adamw learning rate
if args.stud_name == "vgg11":
    args.adamw_lr = 0.0005
elif args.stud_name == "conv3":
    args.adamw_lr = 0.001
elif args.stud_name == "conv4":
    args.adamw_lr = 0.001
elif args.stud_name == "conv5":
    args.adamw_lr = 0.001
elif args.stud_name == "conv6":
    args.adamw_lr = 0.001
elif args.stud_name == "resnet18":
    args.adamw_lr = 0.001
elif args.stud_name == "resnet18_modified":
    args.adamw_lr = 0.001
elif args.stud_name == "efficientnet_b0":
    args.adamw_lr = 0.002
elif args.stud_name == "mobilenet_v2":
    args.adamw_lr = 0.0025
elif args.stud_name == "alexnet":
    args.adamw_lr = 0.0001
elif args.stud_name == "resnet50":
    args.adamw_lr = 0.001
elif args.stud_name == "resnet50_modified":
    args.adamw_lr = 0.001
elif args.stud_name == "resnet101":
    args.adamw_lr = 0.001
elif args.stud_name == "resnet101_modified":
    args.adamw_lr = 0.001
elif args.stud_name == "vit_b_16":
    args.adamw_lr = 0.0001
elif args.stud_name == "swin_v2_t":
    args.adamw_lr = 0.0001
