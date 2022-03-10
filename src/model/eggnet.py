#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   src\model\eggnet.py
# @Time    :   2022-03-10 10:51:01
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    def __init__(self, classes_num, in_channels=30, in_feature=1500, dropout=0.5):
        super(EEGNet, self).__init__()
        self.drop_out = dropout
        self.in_channel = in_channels

        self.mlp = nn.Sequential(
            nn.Linear(in_feature, 64 * 64),
            nn.ReLU(),
            nn.Dropout(self.drop_out),
        )

        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channel,  # input shape (1, 64, 64)
                out_channels=8,  # num_filters
                kernel_size=(1, 5),  # filter size
                padding=(0, 2),  # padding
                bias=False),  # output shape (8, 64, 64)
            nn.BatchNorm2d(8)  # output shape (8, 64, 64)
        )

        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,  # input shape (8, 64, 64)
                out_channels=16,  # num_filters
                kernel_size=(64, 1),  # filter size
                groups=8,
                bias=False),  # output shape (16, 1, 64)
            nn.BatchNorm2d(16),  # output shape (16, 1, 64)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (16, 1, 16)
            nn.Dropout(self.drop_out)  # output shape (16, 1, 16)
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, 16)
                out_channels=16,  # num_filters
                kernel_size=(1, 3),  # filter size
                groups=16,
                padding=(0, 1),  # padding
                bias=False),  # output shape (16, 1, 16)
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, 16)
                out_channels=16,  # num_filters
                kernel_size=(1, 1),  # filter size
                bias=False),  # output shape (16, 1, 16)
            nn.BatchNorm2d(16),  # output shape (16, 1, 16)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (16, 1, 2)
            nn.Dropout(self.drop_out))

        self.out = nn.Linear((16 * 2), classes_num)

    def forward(self, x):
        x = self.mlp(x)
        x = x.view(-1, self.in_channel, 64, 64)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x  # return x for visualization


if __name__ == '__main__':
    net = EEGNet(classes_num=2)
    print(net)
    x = torch.randn(32, 30, 1500)
    y = net(x)
    print(y.size())