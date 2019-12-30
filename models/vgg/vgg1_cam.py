import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import numpy as np
import random
from utils.l1 import L1_norm
import sys
import utils.fusion as fusion 
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

    def __init__(self, features, num_classes=1000, threshold=0.6):
        super(VGG, self).__init__()
        self.features = features#[:-1]
        self.num_classes = num_classes
        self.cls = self.classifier(512, num_classes)
        self.cls_erase = self.classifier(512, num_classes)
        self._initialize_weights()
        self.threshold = threshold

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
        x = self.features(x)
        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        # Branch A
        out = self.cls(x)
        batch_size = x.size(0)
        self.cam_map = out.view(batch_size, self.num_classes, 28, 28)
        logits_1 = torch.mean(torch.mean(self.cam_map, dim=2), dim=2)

        logits0 = F.softmax(logits_1, dim=1)
        logits0 = logits0.cpu().data.numpy()
        
        indices = np.argmax(logits0, axis=1)
        indices = torch.from_numpy(indices)
        
        localization_map_normed = self.get_atten_map(out, indices, True)
        self.attention = localization_map_normed
        localization_map_normed = 1 - localization_map_normed
        feat_erase = self.erase_feature_maps(localization_map_normed, x, self.threshold)
        
        # Branch B
        out_erase = self.cls_erase(feat_erase)

        self.map_erase = out_erase.view(batch_size, self.num_classes, 28, 28)
        logits_ers = torch.mean(torch.mean(out_erase, dim=2), dim=2)

        logits0 = F.softmax(logits_ers, dim=1)
        logits0 = logits0.cpu().data.numpy()
        
        indices = np.argmax(logits0, axis=1)
        indices = torch.from_numpy(indices)
        



        localization_map_normed2 = self.get_atten_map(out_erase, indices, True)        
        self.attention2 = localization_map_normed2


        return [logits_1, logits_ers]
        

    def get_loss(self, logits, gt_labels):
        logits0 = logits
        loss_cls = self.loss_cross_entropy(logits0[0], gt_labels.long())
        loss_cls_ers = self.loss_cross_entropy(logits0[1], gt_labels.long())

        loss_val = loss_cls + loss_cls_ers

        return loss_val



    def get_localization_maps(self):
        #return fusion.attention_fusion_weight(self.cam_map,self.map_erase)
        #return L1_norm (self.cam_map,self.map_erase)
        map1 = self.normalize_atten_maps(self.cam_map)
        map_erase = self.normalize_atten_maps(self.map_erase)
        return torch.max(map1, map_erase)
    
    def get_cam_maps(self):
        return self.cam_map


    def get_atten_map(self, feature_maps, gt_labels, normalize=True):
        label = gt_labels.long()

        feature_map_size = feature_maps.size()
        batch_size = feature_map_size[0]

        atten_map = torch.zeros([feature_map_size[0], feature_map_size[2], feature_map_size[3]])
        atten_map = Variable(atten_map.cuda())
        for batch_idx in range(batch_size):
            #print 'torch.squeeze',torch.squeeze(feature_maps[batch_idx, label.data[batch_idx], :,:]).size()
            atten_map[batch_idx,:,:] = torch.squeeze(feature_maps[batch_idx, label.data[batch_idx], :,:])

        if normalize:
            atten_map = self.normalize_atten_maps(atten_map)

        return atten_map


    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()

        #--------------------------
        batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,))-batch_mins,
                                 batch_maxs - batch_mins)
        atten_normed = atten_normed.view(atten_shape)

        return atten_normed

    def erase_feature_maps(self, atten_map_normed, feature_maps, threshold):
        # atten_map_normed = torch.unsqueeze(atten_map_normed, dim=1)
        # atten_map_normed = self.up_resize(atten_map_normed)
        if len(atten_map_normed.size())>3:
            atten_map_normed = torch.squeeze(atten_map_normed)
        atten_shape = atten_map_normed.size()

        pos = torch.ge(atten_map_normed, threshold)
        mask = torch.ones(atten_shape).cuda()
        mask[pos.data] = 0.0
        mask = torch.unsqueeze(mask, dim=1)
        #erase
        atten_map_normed = torch.unsqueeze(atten_map_normed, dim=1)
        erased_feature_maps = feature_maps * atten_map_normed # Variable(mask)

        return erased_feature_maps  

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

 
def model(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    """
    model = VGG(make_layers(cfg['D1'], dilation=dilation['D1']), **kwargs)
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
