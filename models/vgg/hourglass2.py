'''
Hourglass network inserted in the pre-activated Resnet
Use lr=0.01 for current version
(c) YANG, Wei
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
# from .preresnet import BasicBlock, Bottleneck


__all__ = ['hg_gn']

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

# hardcode group number
gn = 32

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.GroupNorm(gn, inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.GroupNorm(gn, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.GroupNorm(gn, planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual

        return out

# dummy = torch.ones((1, 128, 224, 224))

# block = Bottleneck(128, 64)
# block(dummy).shape
# print(block)


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)
        
        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)

# dummy = torch.ones((1, 256, 224, 224))

# block = Hourglass(Bottleneck, 4, 128, 4)
# block(dummy).shape
# print(block)

class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, features, block=Bottleneck, num_stacks=2, num_blocks=4, num_classes=16, threshold=0.6, args=None):
        super(HourglassNet, self).__init__()


        self.features = features[:-1]
        self.num_classes = num_classes
        self._initialize_weights()
        self.threshold = threshold
        self.onehot = False
        self.loss_cross_entropy = nn.CrossEntropyLoss()
        self.inplanes = 1024
        self.num_feats = 512
        self.num_stacks = num_stacks
        self.conv2 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, dilation=1)
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=True)
        # self.bn1 = nn.GroupNorm(gn, self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.layer1 = self._make_residual(block, self.inplanes, 1)
        # self.layer2 = self._make_residual(block, self.inplanes, 1)
        # self.layer3 = self._make_residual(block, self.num_feats, 1)
        # self.maxpool = nn.MaxPool2d(2, stride=2)
        self.branchA = self.hourglass_module(block, num_stacks, num_blocks, num_classes)
        self.branchB = self.hourglass_module(block, num_stacks, num_blocks, num_classes)
        self.threshold = threshold
        self.onehot = False #args.onehot
        self.loss_cross_entropy = nn.CrossEntropyLoss()

    def hourglass_module(self, block, num_stacks, num_blocks, num_classes):
        # build hourglass modules
        ch = self.num_feats*block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            if i < num_stacks-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))
        return nn.ModuleDict({        
            'hg' : nn.ModuleList(hg).cuda(),
            'res' : nn.ModuleList(res).cuda(),
            'fc' : nn.ModuleList(fc).cuda(),
            'score' : nn.ModuleList(score).cuda(),
            'fc_' : nn.ModuleList(fc_).cuda(),
            'score_' : nn.ModuleList(score_).cuda()
        })

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.GroupNorm(gn, inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )



    def get_loss(self, logits, gt_labels):
        if self.onehot == 'True':
            gt = gt_labels.float()
        else:
            gt = gt_labels.long()

        if type(logits) == list:  # multiple logits
            loss_val = 0
            loss_val1 = 0
            loss_val2 = 0
            for j, o in enumerate(logits[0]):
                
                loss_val1 += self.loss_cross_entropy(o, gt)
                loss_val2 += self.loss_cross_entropy(logits[1][j], gt)
                
            loss_val = loss_val1 +loss_val2
            #logits = logits[-1]
        else:  # single logits
            loss_val = model.module.get_loss(o, label_var)

        
        #loss_cls_ers = self.loss_cross_entropy(logits[1], gt)

        #loss_val = loss_cls + loss_cls_ers
        return loss_val

    def branch_forward(self, branch, x):

        out = []
        for i in range(self.num_stacks):
            y = branch['hg'][i](x)
            y = branch['res'][i](y)
            y = branch['fc'][i](y)
            score = branch['score'][i](y)
            
            
            
            logits_1 = torch.mean(torch.mean(score, dim=2), dim=2)
            out.append(logits_1)
            
            if i < self.num_stacks-1:
                fc_ = branch['fc_'][i](y)
                score_ = branch['score_'][i](score)
                x = x + fc_ + score_
        
        return out, score
        #return out

    def forward(self, x, label=None):
        
        x = self.features(x)
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)

        # x = self.layer1(x)
        # x = self.maxpool(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.maxpool(x)
        x = self.conv2(x)
        out, map1 = self.branch_forward(self.branchA, x)
        
        self.map1 = map1
        
        
        localization_map_normed = self.get_atten_map(map1, label, True)
        self.attention = localization_map_normed
        feat_erase = self.erase_feature_maps(localization_map_normed, x, self.threshold)
        
        
        out_erase, map_ers = self.branch_forward(self.branchB, x)
        self.map_ers = map_ers
        
        return [out, out_erase]




    def get_localization_maps(self,maps):
    
        map1 = self.normalize_atten_maps(maps[0])
        map_erase = self.normalize_atten_maps(maps[1])
        return torch.max(map1, map_erase)


    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()

        #--------------------------
        batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,))-batch_mins,
                                 batch_maxs - batch_mins)
        atten_normed = atten_normed.view(atten_shape)

        return atten_normed

    def get_atten_map(self, feature_maps, gt_labels, normalize=True):
        label = gt_labels.long()
        feature_map_size = feature_maps.size()
        batch_size = feature_map_size[0]

        atten_map = torch.zeros([feature_map_size[0], feature_map_size[2], feature_map_size[3]])
        atten_map = Variable(atten_map.cuda())
        for batch_idx in range(batch_size):        
            #print 'label.data[batch_idx] ', feature_maps[batch_idx, label.data[batch_idx], :,:].size()
            atten_map[batch_idx,:,:] = torch.squeeze(feature_maps[batch_idx, label.data[batch_idx], :,:])

        if normalize:
            atten_map = self.normalize_atten_maps(atten_map)

        return atten_map

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
        erased_feature_maps = feature_maps * Variable(mask)

        #print 'erased_feature_maps ', erased_feature_maps
        return erased_feature_maps



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


def make_layers(cfg, dilation=None, batch_norm=False):
    layers = []
    in_channels = 3
    for v, d in zip(cfg, dilation):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
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



def model(pretrained=False,**kwargs):

    model = HourglassNet(make_layers(cfg['D'], dilation=dilation['D1']), Bottleneck, num_stacks=kwargs["num_stacks"], num_blocks=kwargs["num_blocks"],
                         num_classes=kwargs["num_classes"], args=kwargs)
    #model = VGG(make_layers(cfg['O'], dilation=dilation['D1']), **kwargs)
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
    
        print model
    return model


