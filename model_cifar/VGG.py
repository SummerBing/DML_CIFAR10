'''
VGG16 for CIFAR-10/100 Dataset.

Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

'''

import torch
import torch.nn as nn

__all__ = ['vgg16', 'vgg19']


cfg = {
   16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
   19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, num_classes=100, depth=16, dropout=0.0, KD=False):
        super(VGG, self).__init__()
        self.KD = KD
        self.inplances = 64
        self.conv1 = nn.Conv2d(3, self.inplances, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.inplances)
        self.conv2 = nn.Conv2d(self.inplances, self.inplances, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.inplances)
        self.relu = nn.ReLU(True)
        self.layer1 = self._make_layers(128, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        if depth == 16:
            num_layer = 3
        elif depth == 19:
            num_layer = 4

        self.layer2 = self._make_layers(256, num_layer)
        self.layer3 = self._make_layers(512, num_layer)
        self.layer4 = self._make_layers(512, num_layer)

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, input, num_layer):
        layers = []
        for i in range(num_layer):
            conv2d = nn.Conv2d(self.inplances, input, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(input), nn.ReLU(inplace=True)]
            self.inplances = input
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x_f = x.view(x.size(0), -1)
        x = self.classifier(x_f)
        if self.KD:
            return x_f, x
        else:
            return x


class VGG_feature(nn.Module):
    def __init__(self, depth=16):
        super(VGG_feature, self).__init__()
        self.inplances = 64
        self.conv1 = nn.Conv2d(3, self.inplances, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.inplances)
        self.conv2 = nn.Conv2d(self.inplances, self.inplances, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.inplances)
        self.relu = nn.ReLU(True)
        self.layer1 = self._make_layers(128, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        if depth == 16:
            num_layer = 3
        elif depth == 19:
            num_layer = 4

        self.layer2 = self._make_layers(256, num_layer)
        self.layer3 = self._make_layers(512, num_layer)
        self.layer4 = self._make_layers(512, num_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, input, num_layer):
        layers = []
        for i in range(num_layer):
            conv2d = nn.Conv2d(self.inplances, input, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(input), nn.ReLU()]
            self.inplances = input
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class VGG_classifier(nn.Module):
    def __init__(self, num_class=100, dropout=0.0):
        super(VGG_classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_class),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, input, num_layer):
        layers = []
        for i in range(num_layer):
            conv2d = nn.Conv2d(self.inplances, input, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(input), nn.ReLU(inplace=True)]
            self.inplances = input
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x_f = x.view(x.size(0), -1)
        x = self.classifier(x_f)
        return x


class VGG_DML(nn.Module):
    def __init__(self, num_class, depth, num_clf=2):
        super(VGG_DML, self).__init__()
        assert num_clf <=4, 'do not support more than 4 classifiers yet'
        self.feature_extractor = VGG_feature(depth=depth)
        self.clf1 = VGG_classifier(num_class=num_class)
        self.clf2 = VGG_classifier(num_class=num_class)

        if num_clf >= 3:
            self.clf3 = VGG_classifier(num_class=num_class)
        if num_clf >= 4:
            self.clf4 = VGG_classifier(num_class=num_class)

        if num_clf == 2:
            self.clfs = [self.clf1, self.clf2]
        if num_clf == 3:
            self.clfs = [self.clf1, self.clf2, self.clf3]
        if num_clf == 4:
            self.clfs = [self.clf1, self.clf2, self.clf3, self.clf4]

    def forward(self, x):
        feature = self.feature_extractor(x)
        outs = [clf(feature) for clf in self.clfs]
        return outs


def vgg16(pretrained=False, path=None, **kwargs):
    """
    Constructs a VGG16 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = VGG(depth=16, **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model


def vgg19(pretrained=False, path=None, **kwargs):
    """
    Constructs a VGG19 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = VGG(depth=19, **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model


if __name__ == '__main__':
    import torch
    input = torch.FloatTensor(1, 3, 32, 32)
    net = VGG()
    from util import print_network

    print_network(net, name='1')
    out = net(input)
    print(out.size())

