from collections import OrderedDict

import torch.nn as nn

from .l0_layers import *


class L0LeNet(nn.Module):
    def __init__(self, num_classes=10, device=0, input_size=(1, 32, 32), conv_dims=[64, 128], fc_dims=[256, 256],
                 kernel_sizes=[5, 5], layer_sparsity=(0., 0., 0., 0.)):
        super(L0LeNet, self).__init__()

        self.num_classes = num_classes
        self.input_size = input_size
        self.conv_dims = conv_dims
        self.fc_dims = fc_dims
        self.kernel_sizes = kernel_sizes
        self.layer_sparsity = layer_sparsity

        if torch.cuda.is_available():
            self.device = 'cuda:%d' % device
        else:
            self.device = 'cpu'

        # model structure
        self.model_construct()

    def lamb(self):
        l = []
        for layer in self.layers:
            l.append(layer.get_lamb())
        return l

    def model_construct(self, sparsity=None):
        if sparsity is not None:
            self.layer_sparsity = sparsity

        convs = OrderedDict([
            ('conv1',
             nn.Conv2d(self.input_size[0], self.conv_dims[0], self.kernel_sizes[0])),
            ('relu1', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(2)),
            ('conv2',
             nn.Conv2d(self.conv_dims[0], self.conv_dims[1], self.kernel_sizes[1])),
            ('relu2', nn.ReLU()),
            ('maxpool2', nn.MaxPool2d(2))
        ])

        # , sparse_ratio = self.layer_sparsity[0],
        # device = self.device

        self.convs = nn.Sequential(convs)

        # h_out = lambda: int(self.convs.conv2.h_out((self.convs.conv1.h_out(self.input_size[1]) / 2)) / 2)
        # w_out = lambda: int(self.convs.conv2.w_out((self.convs.conv1.w_out(self.input_size[1]) / 2)) / 2)
        # self.output_len = self.conv_dims[-1] * h_out() * w_out()

        if torch.cuda.is_available():
            self.convs = self.convs.to(self.device)

        fcs = OrderedDict([
            ('fc1', nn.Linear(25*self.conv_dims[-1], self.fc_dims[0])),
            ('relu3', nn.ReLU()),
            ('fc2', nn.Linear(self.fc_dims[0], self.fc_dims[1])),
            ('relu4', nn.ReLU()),
            (
            'fc3', nn.Linear(self.fc_dims[1], self.num_classes)),
        ])

        # , sparse_ratio=self.layer_sparsity[2], device=self.device

        self.fcs = nn.Sequential(fcs)

        if torch.cuda.is_available():
            self.fcs = self.fcs.to(self.device)

        self.layers = []
        for m in self.modules():
            if isinstance(m, L0Dense) or isinstance(m, L0Conv2d):
                self.layers.append(m)

    def forward(self, input):
        out = self.convs(input)
        out = out.view(out.size(0), -1)
        out = self.fcs(out)
        return out


class LeNet_3C(nn.Module):
    def __init__(self, num_classes=10, device=0, input_size=(1, 32, 32), conv_dims=[32, 64, 128], fc_dims=[256, 256],
                 kernel_sizes=[4, 4, 4], layer_sparsity=(0., 0., 0., 0.)):
        super(LeNet_3C, self).__init__()

        self.num_classes = num_classes
        self.input_size = input_size
        self.conv_dims = conv_dims
        self.fc_dims = fc_dims
        self.kernel_sizes = kernel_sizes
        self.layer_sparsity = layer_sparsity

        if torch.cuda.is_available():
            self.device = 'cuda:%d' % device
        else:
            self.device = 'cpu'

        # model structure
        self.model_construct()

    def lamb(self):
        l = []
        for layer in self.layers:
            l.append(layer.get_lamb())
        return l

    def model_construct(self, sparsity=None):
        if sparsity is not None:
            self.layer_sparsity = sparsity

        convs = OrderedDict([
            ('conv1',
            nn.Conv2d(self.input_size[0], self.conv_dims[0], self.kernel_sizes[0], 2)),
            ('relu1', nn.ReLU()),
            ('conv2',
             nn.Conv2d(self.conv_dims[0], self.conv_dims[1], self.kernel_sizes[1], 2)),
            ('relu2', nn.ReLU()),
            ('conv3',
             nn.Conv2d(self.conv_dims[1], self.conv_dims[2], self.kernel_sizes[2], 2)),
            ('relu3', nn.ReLU()),
        ])

        # , sparse_ratio = self.layer_sparsity[0],
        # device = self.device

        self.convs = nn.Sequential(convs)

        # h_out = lambda: int(self.convs.conv2.h_out((self.convs.conv1.h_out(self.input_size[1]) / 2)) / 2)
        # w_out = lambda: int(self.convs.conv2.w_out((self.convs.conv1.w_out(self.input_size[1]) / 2)) / 2)
        # self.output_len = self.conv_dims[-1] * h_out() * w_out()

        if torch.cuda.is_available():
            self.convs = self.convs.to(self.device)

        fcs = OrderedDict([
            ('fc1', nn.Linear(4*self.conv_dims[-1], self.fc_dims[0])),
            ('relu3', nn.ReLU()),
            ('fc2', nn.Linear(self.fc_dims[0], self.fc_dims[1])),
            ('relu4', nn.ReLU()),
            ('fc3', nn.Linear(self.fc_dims[1], self.num_classes)),
        ])

        # , sparse_ratio=self.layer_sparsity[2], device=self.device

        self.fcs = nn.Sequential(fcs)

        if torch.cuda.is_available():
            self.fcs = self.fcs.to(self.device)

        self.layers = []
        for m in self.modules():
            if isinstance(m, L0Dense) or isinstance(m, L0Conv2d):
                self.layers.append(m)

    def forward(self, input):
        out = self.convs(input)
        # print(out.size())
        out = out.view(out.size(0), -1)
        out = self.fcs(out)
        return out


class LeNet1D(nn.Module):
    def __init__(self, in_channel, input_len, output_len):
        super(LeNet1D, self).__init__()

        self.padding = 0
        self.kernel_size = 5
        self.stride = 2

        l_out = lambda l_in, padding, kernel_size, stride: int(
            (l_in + 2 * padding - 1 * (kernel_size - 1) - 1) / stride + 1)

        self.convs = nn.Sequential(
            nn.Conv1d(in_channel, 6, self.kernel_size, self.stride, self.padding),
            nn.LeakyReLU(),
            nn.Conv1d(6, 16, self.kernel_size, self.stride, self.padding),
            nn.LeakyReLU(),
            nn.Conv1d(16, 120, self.kernel_size, self.stride, self.padding),
            nn.LeakyReLU()
        )
        conv_out_len = l_out(input_len, self.padding, self.kernel_size, self.stride)
        conv_out_len = l_out(conv_out_len, self.padding, self.kernel_size, self.stride)
        self.conv_out_len = l_out(conv_out_len, self.padding, self.kernel_size, self.stride)

        self.fcs = nn.Sequential(
            nn.Linear(self.conv_out_len * 120, 84),
            nn.LeakyReLU(),
            nn.Linear(84, output_len)
        )

        # model structure

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        out = self.convs(input)
        out = out.view(out.size(0), -1)
        out = self.fcs(out)
        return out
