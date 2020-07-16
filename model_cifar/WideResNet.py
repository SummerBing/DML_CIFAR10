import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_class, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_class)
        self.nChannels = nChannels

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)  # after conv2
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels[3])
        return self.fc(out)


class WideResNet_feature(nn.Module):
    def __init__(self, depth, widen_factor=1, dropRate=0.0):
        super(WideResNet_feature, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)  # after conv2
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        return out


class WideResNet_classifier(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet_classifier, self).__init__()
        self.nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        self.fc = nn.Linear(self.nChannels[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = x.view(-1, self.nChannels[3])
        return self.fc(out)


# class WideResNet_DML(nn.Module):
#     def __init__(self, num_class, depth, widen_factor):
#         super(WideResNet_DML, self).__init__()
#         self.feature_extractor = WideResNet_feature(depth=depth, widen_factor=widen_factor)
#         self.clf1 = WideResNet_classifier(depth=depth, num_classes=num_class, widen_factor=widen_factor)
#         self.clf2 = WideResNet_classifier(depth=depth, num_classes=num_class, widen_factor=widen_factor)

#     def forward(self, x):
#         feature = self.feature_extractor(x)
#         out1 = self.clf1(feature)
#         out2 = self.clf2(feature)
#         return out1, out2

class WideResNet_DML(nn.Module):
    def __init__(self, num_class, depth, widen_factor, num_clf=2):
        super(WideResNet_DML, self).__init__()
        assert num_clf <=4, 'do not support more than 4 classifiers yet'
        self.feature_extractor = WideResNet_feature(depth=depth, widen_factor=widen_factor)
        self.clf1 = WideResNet_classifier(depth=depth, num_classes=num_class, widen_factor=widen_factor)
        self.clf2 = WideResNet_classifier(depth=depth, num_classes=num_class, widen_factor=widen_factor)
        if num_clf >= 3:
            self.clf3 = WideResNet_classifier(depth=depth, num_classes=num_class, widen_factor=widen_factor)
        if num_clf >= 4:
            self.clf4 = WideResNet_classifier(depth=depth, num_classes=num_class, widen_factor=widen_factor)

        if num_clf == 2:
            self.clfs = [self.clf1, self.clf2]
        if num_clf == 3:
            self.clfs = [self.clf1, self.clf2, self.clf3]
        if num_clf == 4:
            self.clfs = [self.clf1, self.clf2, self.clf3, self.clf4]

    def forward(self, x, separate=False):
        if not separate:
            feature = self.feature_extractor(x)
            outs = [clf(feature) for clf in self.clfs]
        else:
            feature = self.feature_extractor(x.view(x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
            # print(feature.shape)
            features = feature.view(x.shape[0], x.shape[1], feature.shape[1])
            outs = [clf(feature) for clf, feature in zip(self.clfs, features)]
        return outs


class WideResNet_DML_12(nn.Module):
    def __init__(self, num_class, depth, widen_factor):
        super(WideResNet_DML_12, self).__init__()
        self.feature_extractor = WideResNet_feature(depth=depth, widen_factor=widen_factor)
        self.clf1 = WideResNet_classifier(depth=depth, num_classes=num_class, widen_factor=widen_factor)
        self.clf2 = WideResNet_classifier(depth=depth, num_classes=num_class, widen_factor=widen_factor)

    def forward(self, x1, x2):
        feature1 = self.feature_extractor(x1)
        feature2 = self.feature_extractor(x2)
        out1 = self.clf1(feature1)
        out2 = self.clf2(feature2)
        return out1, out2


class WideResNet_DML_3C(nn.Module):
    def __init__(self, num_class, depth, widen_factor):
        super(WideResNet_DML_3C, self).__init__()
        self.feature_extractor = WideResNet_feature(depth=depth, widen_factor=widen_factor)
        self.clf1 = WideResNet_classifier(depth=depth, num_classes=num_class, widen_factor=widen_factor)
        self.clf2 = WideResNet_classifier(depth=depth, num_classes=num_class, widen_factor=widen_factor)
        self.clf3 = WideResNet_classifier(depth=depth, num_classes=num_class, widen_factor=widen_factor)

    def forward(self, x):
        feature = self.feature_extractor(x)
        out1 = self.clf1(feature)
        out2 = self.clf2(feature)
        out3 = self.clf3(feature)
        return out1, out2, out3


if __name__ == '__main__':
    net = WideResNet(depth=16, num_classes=10)
    oo = torch.FloatTensor(1, 3, 32, 32)
    out = net(oo)
    # print(net)
    print(net.get_bn_before_relu())
    print(net.get_channel_num())
