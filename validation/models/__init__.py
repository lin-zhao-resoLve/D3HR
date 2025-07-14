import torch
import torch.nn as nn
import torchvision.models as thmodels

from .convnet import ConvNet
from .resnet import resnet18, resnet50, resnet101, resnet152
from .mobilenet_v2 import mobilenetv2
# import timm



def load_model(model_name="resnet18", dataset="cifar10", spec='full', pretrained=True, input_size=224, classes=[]):
    def get_model(model_name="resnet18"):
        if "conv" in model_name:
            size = input_size
            nclass = 1000

            model = ConvNet(
                num_classes=nclass,
                net_norm="batch",
                net_act="relu",
                net_pooling="avgpooling",
                net_depth=int(model_name[-1]),
                net_width=128,
                channel=3,
                im_size=(size, size),
            )
        elif model_name == 'resnet18':
            model = resnet18(weights=None)
        elif model_name == 'resnet50':
            model = resnet50(weights=None)
        elif model_name == 'resnet101':
            model = resnet101(weights=None)
        elif model_name == 'resnet152':
            model = resnet152(weights=None)
        elif model_name == 'mobilenet_v2':
            model = mobilenetv2()
        elif model_name == 'efficientnet_b0':
            model = timm.create_model('efficientnet_b0.ra_in1k', pretrained=False)
        elif model_name == "resnet18_modified":
            model = thmodels.__dict__["resnet18"](pretrained=False)
            model.conv1 = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
            model.maxpool = nn.Identity()
        elif model_name == "resnet50_modified":
            model = thmodels.__dict__["resnet50"](pretrained=False)
            model.conv1 = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
            model.maxpool = nn.Identity()
        elif model_name == "resnet101_modified":
            model = thmodels.__dict__["resnet101"](pretrained=False)
            model.conv1 = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
            model.maxpool = nn.Identity()
        else:
            model = thmodels.__dict__[model_name](weights=None)

        return model

    def pruning_classifier(model=None, classes=[]):
        try:
            model_named_parameters = [name for name, x in model.named_parameters()]
            for name, x in model.named_parameters():
                if (
                    name == model_named_parameters[-1]
                    or name == model_named_parameters[-2]
                ):
                    x.data = x[classes]
        except:
            print("ERROR in changing the number of classes.")

        return model

    model = get_model(model_name)
    model = pruning_classifier(model, classes)

    if pretrained:
        if dataset == 'imagenet_1k':
            if model_name == "efficientnet_b0":
                checkpoint = timm.create_model('efficientnet_b0.ra_in1k', pretrained=True).state_dict()
                model.load_state_dict(checkpoint)
            elif model_name == 'conv4':
                state_dict = torch.load('/home/linz/CONCORD/pretrained_models/imagenet-1k_conv4.pth')
                model.load_state_dict(state_dict['model'])
            elif model_name == 'resnet18':
                model = resnet18(weights='DEFAULT')
            elif model_name == 'mobilenet_v2':
                model.load_state_dict(torch.load('/home/zhao.lin1/CONCORD/pretrained_models/mobilenetv2_1.0-0c6065bc.pth'))
            else:
                raise AttributeError(f'{model_name} is not supported in the pre-trained pool')
        else:
            checkpoint = torch.load(
                f"pretrain_models/{dataset}_{model_name}.pth", map_location="cpu"
            )
            model.load_state_dict(checkpoint["model"])


    return model


# def load_model(model_name="resnet18", dataset="cifar10", pretrained=True, classes=[]):
#     def get_model(model_name="resnet18"):
#         if "conv" in model_name:
#             if dataset in ["cifar10", "cifar100"]:
#                 size = 32
#             elif dataset == "tinyimagenet":
#                 size = 64
#             elif dataset in ["imagenet-nette", "imagenet-woof", "imagenet-100"]:
#                 size = 128
#             else:
#                 size = 224

#             nclass = len(classes)

#             model = ConvNet(
#                 num_classes=nclass,
#                 net_norm="batch",
#                 net_act="relu",
#                 net_pooling="avgpooling",
#                 net_depth=int(model_name[-1]),
#                 net_width=128,
#                 channel=3,
#                 im_size=(size, size),
#             )
#         elif model_name == "resnet18_modified":
#             model = thmodels.__dict__["resnet18"](pretrained=False)
#             model.conv1 = nn.Conv2d(
#                 3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
#             )
#             model.maxpool = nn.Identity()
#         elif model_name == "resnet101_modified":
#             model = thmodels.__dict__["resnet101"](pretrained=False)
#             model.conv1 = nn.Conv2d(
#                 3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
#             )
#             model.maxpool = nn.Identity()
#         else:
#             model = thmodels.__dict__[model_name](pretrained=False)

#         return model

#     def pruning_classifier(model=None, classes=[]):
#         try:
#             model_named_parameters = [name for name, x in model.named_parameters()]
#             for name, x in model.named_parameters():
#                 if (
#                     name == model_named_parameters[-1]
#                     or name == model_named_parameters[-2]
#                 ):
#                     x.data = x[classes]
#         except:
#             print("ERROR in changing the number of classes.")

#         return model

#     # "imagenet-100" "imagenet-10" "imagenet-first" "imagenet-nette" "imagenet-woof"
#     model = get_model(model_name)
#     model = pruning_classifier(model, classes)
#     if pretrained:
#         if dataset in [
#             "imagenet-100",
#             "imagenet-10",
#             "imagenet-nette",
#             "imagenet-woof",
#             "tinyimagenet",
#             "cifar10",
#             "cifar100",
#         ]:
#             checkpoint = torch.load(
#                 f"./data/pretrain_models/{dataset}_{model_name}.pth", map_location="cpu"
#             )
#             model.load_state_dict(checkpoint["model"])
#         elif dataset in ["imagenet-1k"]:
#             if model_name == "efficientNet-b0":
#                 # Specifically, for loading the pre-trained EfficientNet model, the following modifications are made
#                 from torchvision.models._api import WeightsEnum
#                 from torch.hub import load_state_dict_from_url

#                 def get_state_dict(self, *args, **kwargs):
#                     kwargs.pop("check_hash")
#                     return load_state_dict_from_url(self.url, *args, **kwargs)

#                 WeightsEnum.get_state_dict = get_state_dict

#             model = thmodels.__dict__[model_name](pretrained=True)

#     return model
