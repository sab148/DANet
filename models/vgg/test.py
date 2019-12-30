import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import numpy as np
import random

import sys

sys.path.append('../')

__all__ = [
    'model',
]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=200):
        super(VGG, self).__init__()
        self.features = features[:-1]
        self.num_classes = num_classes        
        self.fc1 = self.classify(64)
        self.fc2 = self.classify(128)
        self.fc3 = self.classify(256)
        self.fc4 = self.classify(512)
        self.fc5 = self.classify(512)
        self.cls = self.classifier(512, num_classes)
        self._initialize_weights()

        self.loss_cross_entropy = nn.CrossEntropyLoss()

    def classifier(self, in_planes, out_planes):
        return nn.Sequential(
            # nn.Dropout(0.5),
            nn.Conv2d(in_planes, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
            # nn.Dropout(0.5),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
            nn.Conv2d(1024, out_planes, kernel_size=1, padding=0)  # fc8
        )

    def classify(self,in_planes):
        return nn.Conv2d(in_planes, self.num_classes, kernel_size=1, padding=0)



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def forward(self, x, label=None):
        batch_size = x.size(0)

        # First block
        x = self.features[:4](x)
        out = self.fc1(x)
        self.cam_map1 = out.view(batch_size, self.num_classes, 224, 224)
        logits0 = torch.mean(torch.mean(self.cam_map1, dim=2), dim=2)

        # Second block
        # x = self.features[4:9](x)
        # out = self.fc2(x)
        # self.cam_map2 = out.view(batch_size, self.num_classes, 112, 112)
        # logits1 = torch.mean(torch.mean(self.cam_map2, dim=2), dim=2)

        # Third block
        x = self.features[4:16](x)
        out = self.fc3(x)
        self.cam_map3 = out.view(batch_size, self.num_classes, 56, 56)
        logits2 = torch.mean(torch.mean(self.cam_map3, dim=2), dim=2)

        # Forth block 
        # x = self.features[16:23](x)
        # out = self.fc4(x)
        # self.cam_map4 = out.view(batch_size, self.num_classes, 28, 28)
        # logits3 = torch.mean(torch.mean(self.cam_map4, dim=2), dim=2)

        # Fifth block
        x = self.features[16:30](x)
        # out = self.fc5(x)
        # self.cam_map5 = out.view(batch_size, self.num_classes, 14, 14)
        # logits4 = torch.mean(torch.mean(self.cam_map5, dim=2), dim=2)

        #x = self.features[28](x)

        out = self.cls(x)
        batch_size = x.size(0)
        self.cam_map6 = out.view(batch_size, self.num_classes, 14, 14)
        logits5 = torch.mean(torch.mean(self.cam_map6, dim=2), dim=2)

        return [logits0, logits2, logits5]

    def get_loss(self, logits, gt_labels):
        logits0 = logits
        loss_cls0 = 0.1 * self.loss_cross_entropy(logits0[0], gt_labels.long())
        loss_cls1 = 0.2 * self.loss_cross_entropy(logits0[1], gt_labels.long())
        loss_cls2 = 0.7 * self.loss_cross_entropy(logits0[2], gt_labels.long())
        # loss_cls3 = 0.1 * self.loss_cross_entropy(logits0[3], gt_labels.long())
        # loss_cls4 = 0.1 * self.loss_cross_entropy(logits0[4], gt_labels.long())
        # loss_cls5 = 0.5 * self.loss_cross_entropy(logits0[5], gt_labels.long())


        loss_val = loss_cls0 + loss_cls1 + loss_cls2 

        return loss_val


    def get_cam_maps(self):
        return torch.mean(torch.stack([self.cam_map1, self.cam_map2, self.cam_map3, self.cam_map4, self.cam_map5, self.cam_map6]),dim=0)

def make_layers(cfg, dilation=None, batch_norm=False):
    layers = []
    in_channels = 3
    for v, d in zip(cfg, dilation):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        elif v == 'L':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, padding=0)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d, dilation=d)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512, 'N'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'O': [64, 64, 'L', 128, 128, 'L', 256, 256, 256, 'L', 512, 512, 512, 'L', 512, 512, 512, 'L']
}

dilation = {
    'D1': [1, 1, 'M', 1, 1, 'M', 1, 1, 1, 'M', 1, 1, 1, 'N', 1, 1, 1, 'N']
}

model = VGG(make_layers(cfg['O'], dilation=dilation['D1']))
print model

def model(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    """
    model = VGG(make_layers(cfg['O'], dilation=dilation['D1']), **kwargs)
    # print(model)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['vgg16'])
        # print(pretrained_dict)
        print('load pretrained model from {}'.format(model_urls['vgg16']))
        for k in pretrained_dict.keys():
            if k not in model_dict:
                print('Key {} is removed from vgg16'.format(k))
        for k in model_dict.keys():
            if k not in pretrained_dict:
                print('Key {} is new added for DA Net'.format(k))
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model
