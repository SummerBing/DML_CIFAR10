import torch
import torch.nn as nn
import util as util


def conv3x3_1(in_channel, out_channel):
    return nn.Sequential(*[
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channel)
    ])


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv3x3_2(in_channel, out_channel):
    return nn.Sequential(*[
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_channel)
    ])


def conv1x1(in_channel, out_channel):
    return nn.Sequential(*[
        nn.ReLU(),
        nn.Conv2d(in_channel, out_channel, kernel_size=1),
        nn.BatchNorm2d(out_channel)
    ])


class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.downsample = (in_channels != out_channels)
        if self.downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            self.ds = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels)
            ])
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.ds = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.ds(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def cfg(depth):
    depth_lst = [20, 32, 56, 110]
    assert (depth in depth_lst), "Error : Resnet depth is not expected."
    cf_dict = {
        '20': (resblock, [3, 3, 3]),
        '32': (resblock, [5, 5, 5]),
        '56': (resblock, [9, 9, 9]),
        '110': (resblock, [18, 18, 18])}
    return cf_dict[str(depth)]


class ResNet(nn.Module):
    def __init__(self, num_class, depth):
        super(ResNet, self).__init__()
        resblock, channel_list = cfg(depth=depth)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        self.res1 = self.make_layer(resblock, channel_list[0], 16, 16)
        self.res2 = self.make_layer(resblock, channel_list[1], 16, 32)
        self.res3 = self.make_layer(resblock, channel_list[2], 32, 64)

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def make_layer(self, block, num, in_channels, out_channels):
        layers = [block(in_channels, out_channels)]
        for i in range(num - 1):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        pre = self.conv1(x)
        pre = self.bn1(pre)
        pre = self.relu(pre)

        h1 = self.res1(pre)
        h2 = self.res2(h1)
        h3 = self.res3(h2)

        out = self.avgpool(h3)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResNet_DML(nn.Module):
    def __init__(self, num_class, depth):
        super(ResNet_DML, self).__init__()
        self.feature_extractor = ResNet_feature(depth=depth)
        self.clf1 = ResNet_classifier(num_class=num_class)
        self.clf2 = ResNet_classifier(num_class=num_class)

    def forward(self, x):
        feature = self.feature_extractor(x)
        out1 = self.clf1(feature)
        out2 = self.clf2(feature)
        return out1, out2


class ResNet_DML_3C(nn.Module):
    def __init__(self, num_class, depth):
        super(ResNet_DML_3C, self).__init__()
        self.feature_extractor = ResNet_feature(depth=depth)
        self.clf1 = ResNet_classifier(num_class=num_class)
        self.clf2 = ResNet_classifier(num_class=num_class)
        self.clf3 = ResNet_classifier(num_class=num_class)

    def forward(self, x):
        feature = self.feature_extractor(x)
        out1 = self.clf1(feature)
        out2 = self.clf2(feature)
        out3 = self.clf3(feature)
        return out1, out2, out3


class ResNet_feature(nn.Module):
    def __init__(self, depth):
        super(ResNet_feature, self).__init__()
        resblock, channel_list = cfg(depth=depth)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        self.res1 = self.make_layer(resblock, channel_list[0], 16, 16)
        self.res2 = self.make_layer(resblock, channel_list[1], 16, 32)
        self.res3 = self.make_layer(resblock, channel_list[2], 32, 64)

        self.avgpool = nn.AvgPool2d(8)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def make_layer(self, block, num, in_channels, out_channels):
        layers = [block(in_channels, out_channels)]
        for i in range(num - 1):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        pre = self.conv1(x)
        pre = self.bn1(pre)
        pre = self.relu(pre)

        h1 = self.res1(pre)
        h2 = self.res2(h1)
        h3 = self.res3(h2)

        out = self.avgpool(h3)
        out = out.view(out.size(0), -1)
        return out


class ResNet_classifier(nn.Module):
    def __init__(self, num_class):
        super(ResNet_classifier, self).__init__()
        self.fc1 = nn.Linear(64, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def forward(self, x):
        out = self.fc1(x)
        return out


def define_tsnet(name, num_class, depth, cuda=True):
    if name == 'resnet':
        net = ResNet(num_class, depth)
    elif name == 'RN_dml':
        net = ResNet_DML(num_class, depth)
    else:
        raise Exception('model name does not exist.')
    if cuda:
        net = torch.nn.DataParallel(net).cuda()
    util.print_network(net, name)
    return net


if __name__ == '__main__':
    ii = torch.FloatTensor(1,3,32,32)
    net = ResNet_DML(depth=110, num_class=100)
    o1, o2 = net(ii)

