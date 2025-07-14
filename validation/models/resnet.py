import torch
import torch.nn as nn
from torchvision.models.resnet import (
    ResNet, ResNet18_Weights, ResNet50_Weights, ResNet101_Weights,
    BasicBlock, Bottleneck,
    _ovewrite_named_param
)


class FeatResNet(ResNet):
    def __init__(self, block, layers, **kwargs):
        super(FeatResNet, self).__init__(block, layers, **kwargs)
    
    def get_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


def resnet18(*, weights=None, progress=True, **kwargs):
    weights = ResNet18_Weights.verify(weights)
    if weights is not None:
        _ovewrite_named_param(kwargs, 'num_classes', len(weights.meta['categories']))

    model = FeatResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
    
    return model


def resnet50(*, weights=None, progress=True, **kwargs):
    weights = ResNet50_Weights.verify(weights)
    if weights is not None:
        _ovewrite_named_param(kwargs, 'num_classes', len(weights.meta['categories']))

    model = FeatResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
    
    return model


def resnet101(*, weights=None, progress=True, **kwargs):
    weights = ResNet101_Weights.verify(weights)
    if weights is not None:
        _ovewrite_named_param(kwargs, 'num_classes', len(weights.meta['categories']))

    model = FeatResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
    
    return model


def resnet152(*, weights=None, progress=True, **kwargs):
    weights = ResNet101_Weights.verify(weights)
    if weights is not None:
        _ovewrite_named_param(kwargs, 'num_classes', len(weights.meta['categories']))

    model = FeatResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
    
    return model
