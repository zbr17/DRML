import pretrainedmodels as ptm
import torch
import torch.nn as nn 
import logging

# Code from: Pytorch Metric BaseLine
class resnet50(nn.Module):
    """
    Container for ResNet50 s.t. it can be used for metric learning.
    The Network has been broken down to allow for higher modularity, if one wishes
    to target specific layers/blocks directly.
    """
    def __init__(self, pretrained=True, list_style=False, no_norm=False):
        super(resnet50, self).__init__()

        if pretrained:
            logging.info('Getting pretrained weights...')
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')
        else:
            logging.info('Not utilizing pretrained weights!')
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained=None)

        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None
            
        # self.last_linear = self.model.last_linear
        self.last_linear = self.model.last_linear
        self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, 128)

        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

    def forward(self, x, is_init_cluster_generation=False):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))

        for layerblock in self.layer_blocks:
            x = layerblock(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0),-1)
        return x
