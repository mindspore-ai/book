# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""ResNet."""
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import TruncatedNormal, Normal

def weight_variable(fan_in):
    """Init weight."""
    stddev = (1.0/fan_in)**0.5
    return TruncatedNormal(stddev)

def dense_weight_variable():
    """The weight for dense."""
    return Normal(0.01)

def _conv3x3(in_channels, out_channels, stride=1, padding=0, pad_mode='same'):
    """Get a conv2d layer with 3x3 kernel size."""
    init_value = weight_variable(in_channels)
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, stride=stride, padding=padding, pad_mode=pad_mode, weight_init=init_value)

def _conv1x1(in_channels, out_channels, stride=1, padding=0, pad_mode='same'):
    """Get a conv2d layer with 1x1 kernel size."""
    init_value = weight_variable(in_channels)
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=1, stride=stride, padding=padding, pad_mode=pad_mode, weight_init=init_value)

def _conv7x7(in_channels, out_channels, stride=1, padding=0, pad_mode='same'):
    """Get a conv2d layer with 7x7 kernel size."""
    init_value = weight_variable(in_channels)
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=7, stride=stride, padding=padding, pad_mode=pad_mode, weight_init=init_value)

def _fused_bn(channels, momentum=0.9):
    """Get a fused batchnorm"""
    return nn.BatchNorm2d(channels, eps=1e-4, momentum=momentum, gamma_init=1, beta_init=0)

def _fused_bn_last(channels, momentum=0.9):
    """Get a fused batchnorm"""
    return nn.BatchNorm2d(channels, eps=1e-4, momentum=momentum, gamma_init=0, beta_init=0)

class BasicBlock(nn.Cell):
    """
    ResNet V1 basic block definition.

    Args:
        in_channels: Integer. Input channel.
        out_channels: Integer. Output channel.
        stride: Integer. Stride size for the initial convolutional layer. Default:1.
        momentum: Float. Momentum for batchnorm layer. Default:0.1.

    Returns:
        Tensor, output tensor.

    Examples:
        BasicBlock(3,256,stride=2,down_sample=True)
    """
    expansion = 1

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 momentum=0.9):
        super(BasicBlock, self).__init__()

        self.conv1 = _conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = _fused_bn(out_channels, momentum=momentum)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = _fused_bn(out_channels, momentum=momentum)
        self.relu = ops.ReLU()
        self.down_sample_layer = None
        self.downsample = (in_channels != out_channels)
        if self.downsample:
            self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channels,
                                                                 out_channels,
                                                                 stride=stride,
                                                                 padding=0),
                                                        _fused_bn(out_channels,
                                                                  momentum=momentum)])
        self.add = ops.TensorAdd()

    def construct(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample:
            identity = self.down_sample_layer(identity)

        out = self.add(x, identity)
        out = self.relu(out)

        return out


class ResidualBlock(nn.Cell):
    """
    ResNet V1 residual block definition.

    Args:
        in_channels: Integer. Input channel.
        out_channels: Integer. Output channel.
        stride: Integer. Stride size for the initial convolutional layer. Default:1.
        momentum: Float. Momentum for batchnorm layer. Default:0.1.

    Returns:
        Tensor, output tensor.

    Examples:
        ResidualBlock(3,256,stride=2,down_sample=True)
    """
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 momentum=0.9):
        super(ResidualBlock, self).__init__()

        out_chls = out_channels // self.expansion
        self.conv1 = _conv1x1(in_channels, out_chls, stride=1)
        self.bn1 = _fused_bn(out_chls, momentum=momentum)

        self.conv2 = _conv3x3(out_chls, out_chls, stride=stride)
        self.bn2 = _fused_bn(out_chls, momentum=momentum)

        self.conv3 = _conv1x1(out_chls, out_channels, stride=1)
        self.bn3 = _fused_bn_last(out_channels, momentum=momentum)

        self.relu = ops.ReLU()
        self.downsample = (in_channels != out_channels)
        self.stride = stride
        if self.downsample:
            self.conv_down_sample = _conv1x1(in_channels, out_channels,
                                             stride=stride)
            self.bn_down_sample = _fused_bn(out_channels, momentum=momentum)
        elif self.stride != 1:
            self.maxpool_down = nn.MaxPool2d(kernel_size=1, stride=2, pad_mode='same')

        self.add = ops.TensorAdd()

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            identity = self.conv_down_sample(identity)
            identity = self.bn_down_sample(identity)
        elif self.stride != 1:
            identity = self.maxpool_down(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    """
    ResNet V1 network.

    Args:
        block: Cell. Block for network.
        layer_nums: List. Numbers of different layers.
        in_channels: Integer. Input channel.
        out_channels: Integer. Output channel.
        num_classes: Integer. Class number. Default:100.

    Returns:
        Tensor, output tensor.

    Examples:
        ResNet(ResidualBlock,
               [3, 4, 6, 3],
               [64, 256, 512, 1024],
               [256, 512, 1024, 2048],
               100)
    """

    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides=(1, 2, 2, 2),
                 num_classes=100):
        super(ResNet, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of "
                             "layer_num, inchannel, outchannel list must be 4!")

        self.conv1 = _conv7x7(3, 64, stride=2)
        self.bn1 = _fused_bn(64)
        self.relu = ops.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0])
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1])
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2])
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3])

        self.mean = ops.ReduceMean(keep_dims=True)
        self.end_point = nn.Dense(out_channels[3], num_classes, has_bias=True,
                                  weight_init=dense_weight_variable())
        self.squeeze = ops.Squeeze()
        self.cast = ops.Cast()

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride):
        """
        Make Layer for ResNet.

        Args:
            block: Cell. Resnet block.
            layer_num: Integer. Layer number.
            in_channel: Integer. Input channel.
            out_channel: Integer. Output channel.
            stride:Integer. Stride size for the initial convolutional layer.

        Returns:
            SequentialCell, the output layer.

        Examples:
            _make_layer(BasicBlock, 3, 128, 256, 2)
        """
        layers = []

        resblk = block(in_channel, out_channel, stride=1)
        layers.append(resblk)

        for _ in range(1, layer_num - 1):
            resblk = block(out_channel, out_channel, stride=1)
            layers.append(resblk)

        resblk = block(out_channel, out_channel, stride=stride)
        layers.append(resblk)

        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        out = self.mean(c5, (2, 3))
        out = self.squeeze(out)
        out = self.end_point(out)

        return out


def resnet50(class_num=10):
    """
    Get ResNet50 neural network.

    Args:
        class_num: Integer. Class number.

    Returns:
        Cell, cell instance of ResNet50 neural network.

    Examples:
        resnet50(100)
    """
    return ResNet(ResidualBlock,
                  [3, 4, 6, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [2, 2, 2, 1],
                  class_num)


def resnet101(class_num=10):
    """
    Get ResNet101 neural network.

    Args:
        class_num: Integer. Class number.

    Returns:
        Cell, cell instance of ResNet101 neural network.

    Examples:
        resnet101(100)
    """
    return ResNet(ResidualBlock,
                  [3, 4, 23, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  class_num)


def resnet34(class_num=10):
    """
    Get ResNet34 neural network.

    Args:
        class_num: Integer. Class number.

    Returns:
        Cell, cell instance of ResNet34 neural network.

    Examples:
        resnet34(100)
    """
    return ResNet(BasicBlock,
                  [3, 4, 6, 3],
                  [64, 64, 128, 256],
                  [64, 128, 256, 512],
                  class_num)


def resnet18(class_num=10):
    """
    Get ResNet18 neural network.

    Args:
        class_num: Integer. Class number.

    Returns:
        Cell, cell instance of ResNet18 neural network.

    Examples:
        resnet18(100)
    """
    return ResNet(BasicBlock,
                  [2, 2, 2, 2],
                  [64, 64, 128, 256],
                  [64, 128, 256, 512],
                  class_num)
