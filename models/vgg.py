import torch.nn as nn

__all__ = ['VGG_', 'vgg11_', 'vgg11bn_', 'vgg13_', 'vgg13bn_', 'vgg16_', 'vgg16bn_', 'vgg19bn_', 'vgg19_']
__all__ += ['VGG', 'vgg11', 'vgg11bn', 'vgg13', 'vgg13bn', 'vgg16', 'vgg16bn', 'vgg19bn', 'vgg19']


class VGG_(nn.Module):
    def __init__(self, features, num_classes=10):
        super(VGG_, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes)
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=1e-3)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers_(config, batch_norm=False):
    layers = []
    in_channels = 3
    for v in config:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    # layers = layers[:-1]  # <<-- optional VGG desing number #2
    # layers += [nn.AvgPool2d(kernel_size=2, stride=2)]  # <<-- optional VGG desing number #2
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]  # <<-- optional VGG desing number #1
    return nn.Sequential(*layers)


cfg_ = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11_(**kwargs):
    return VGG_(make_layers_(cfg_['A']), **kwargs)


def vgg11bn_(**kwargs):
    return VGG_(make_layers_(cfg_['A'], batch_norm=True), **kwargs)


def vgg13_(**kwargs):
    return VGG_(make_layers_(cfg_['B']), **kwargs)


def vgg13bn_(**kwargs):
    return VGG_(make_layers_(cfg_['B'], batch_norm=True), **kwargs)


def vgg16_(**kwargs):
    return VGG_(make_layers_(cfg_['D']), **kwargs)


def vgg16bn_(**kwargs):
    return VGG_(make_layers_(cfg_['D'], batch_norm=True), **kwargs)


def vgg19_(**kwargs):
    return VGG_(make_layers_(cfg_['E']), **kwargs)


def vgg19bn_(**kwargs):
    return VGG_(make_layers_(cfg_['E'], batch_norm=True), **kwargs)

# VGG for ImageNet...


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
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
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(**kwargs):
    return VGG(make_layers(cfg['A']), **kwargs)


def vgg11bn(**kwargs):
    return VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)


def vgg13(**kwargs):
    return VGG(make_layers(cfg_['B']), **kwargs)


def vgg13bn(**kwargs):
    return VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)


def vgg16(**kwargs):
    return VGG(make_layers(cfg['D']), **kwargs)


def vgg16bn(**kwargs):
    return VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)


def vgg19(**kwargs):
    return VGG(make_layers(cfg['E']), **kwargs)


def vgg19bn(**kwargs):
    return VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
