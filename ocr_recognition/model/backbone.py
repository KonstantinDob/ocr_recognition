import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DummyModule(nn.Module):
    """Return same tensor."""

    def forward(self, data):
        return data


class VGGBackBone(nn.Module):
    """FeatureExtractor of CRNN

    Based on https://arxiv.org/pdf/1507.05717.pdf
    """

    def __init__(self, input_channel: int, output_channel: int = 512):
        """VGG BackBone constructor.

        Args:
            input_channel (int): Number of imput channels.
            output_channel (int): Number of output channels.
        """
        super(VGGBackBone, self).__init__()
        # [64, 128, 256, 512]
        self.output_channel = [
            int(output_channel / 8),
            int(output_channel / 4),
            int(output_channel / 2),
            output_channel
        ]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            # 64x16x50
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
            # 128x8x25
            nn.MaxPool2d(2, 2),
            # 256x8x25
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            # 256x4x25
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            # 512x2x25
            nn.MaxPool2d((2, 1), (2, 1)),
            # 512x1x24
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True))

    def forward(self, data: torch.tensor) -> torch.tensor:
        """Forward method.

        Args:
            data (torch.tensor): Data to froward.

        Returns:
            torch.tensor: Data after forward.
        """
        return self.ConvNet(data)


class RCNNBackBone(nn.Module):
    """FeatureExtractor of GRCNN

    Based on:
    https://papers.nips.cc/paper/6637-gated-recurrent-convolution-neural-network-for-ocr.pdf)
    """

    def __init__(self, input_channel: int, output_channel: int = 512):
        """RCNN BackBone constructor.

        Args:
            input_channel (int): Number of imput channels.
            output_channel (int): Number of output channels.
        """
        super(RCNNBackBone, self).__init__()
        # [64, 128, 256, 512]
        self.output_channel = [
            int(output_channel / 8),
            int(output_channel / 4),
            int(output_channel / 2),
            output_channel
        ]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64 x 16 x 50
            GRCL(self.output_channel[0], self.output_channel[0],
                 num_iteration=5, kernel_size=3, pad=1),
            nn.MaxPool2d(2, 2),  # 64 x 8 x 25
            GRCL(self.output_channel[0], self.output_channel[1],
                 num_iteration=5, kernel_size=3, pad=1),
            nn.MaxPool2d(2, (2, 1), (0, 1)),  # 128 x 4 x 26
            GRCL(self.output_channel[1], self.output_channel[2],
                 num_iteration=5, kernel_size=3, pad=1),
            nn.MaxPool2d(2, (2, 1), (0, 1)),  # 256 x 2 x 27
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 2, 1, 0, bias=False),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(True))  # 512 x 1 x 26

    def forward(self, data: torch.tensor) -> torch.tensor:
        """Forward method.

        Args:
            data (torch.tensor): Data to froward.

        Returns:
            torch.tensor: Data after forward.
        """
        return self.ConvNet(data)


class ResNetBackBone(nn.Module):
    """FeatureExtractor of FAN

    Based on: (http://openaccess.thecvf.com/content_ICCV_2017/papers/
    Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf)
    """

    def __init__(self, input_channel: int, output_channel: int = 512):
        """ResNet BackBone constructor.

        Args:
            input_channel (int): Number of imput channels.
            output_channel (int): Number of output channels.
        """
        super(ResNetBackBone, self).__init__()
        self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [1, 2, 5, 3])

    def forward(self, data: torch.tensor) -> torch.tensor:
        """Forward method.

        Args:
            data (torch.tensor): Data to froward.

        Returns:
            torch.tensor: Data after forward.
        """
        return self.ConvNet(data)


class GRCL(nn.Module):
    """Implemented for Gated RCNN."""

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        num_iteration: int,
        kernel_size: int,
        pad: bool
    ):
        """GRC constructor.

        Args:
            input_channel (int): Number of imput channels.
            output_channel (int): Number of output channels.
            num_iteration (int): Number of iteration to Add GRCLUnit.
            kernel_size (int): Convolutions kernel sizes.
            pad (bool): Ahould padding be applied.
        """
        super(GRCL, self).__init__()
        self.wgf_u = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False)
        self.wgr_x = nn.Conv2d(output_channel, output_channel, 1, 1, 0, bias=False)
        self.wf_u = nn.Conv2d(input_channel, output_channel, kernel_size, 1, pad, bias=False)
        self.wr_x = nn.Conv2d(output_channel, output_channel, kernel_size, 1, pad, bias=False)

        self.BN_x_init = nn.BatchNorm2d(output_channel)

        self.num_iteration = num_iteration
        self.GRCL = [GRCLUnit(output_channel)
                     for _ in range(num_iteration)]
        self.GRCL = nn.Sequential(*self.GRCL)

    def forward(self, data: torch.tensor) -> torch.tensor:
        """Compute GRCL forward.

        The input of GRCL is consistant over time t, which is denoted
        by u(0) thus wgf_u / wf_u is also consistant over time t.

        Args:
            data (torch.tensor): Data to froward.

        Returns:
            torch.tensor: Data after forward.
        """
        wgf_u = self.wgf_u(data)
        wf_u = self.wf_u(data)
        x = F.relu(self.BN_x_init(wf_u))

        for i in range(self.num_iteration):
            x = self.GRCL[i](wgf_u, self.wgr_x(x), wf_u, self.wr_x(x))

        return x


class GRCLUnit(nn.Module):
    """GRCl Unit for RCNN."""

    def __init__(self, output_channel: int):
        """ResNet BackBone constructor.

        Args:
            output_channel (int): Number of output channels.
        """
        super(GRCLUnit, self).__init__()
        self.BN_gfu = nn.BatchNorm2d(output_channel)
        self.BN_grx = nn.BatchNorm2d(output_channel)
        self.BN_fu = nn.BatchNorm2d(output_channel)
        self.BN_rx = nn.BatchNorm2d(output_channel)
        self.BN_Gx = nn.BatchNorm2d(output_channel)

    def forward(
        self,
        wgf_u: torch.tensor,
        wgr_x: torch.tensor,
        wf_u: torch.tensor,
        wr_x: torch.tensor
    ) -> torch.tensor:
        """Compute GRCL forward.

        The input of GRCL is consistant over time t, which is denoted
        by u(0) thus wgf_u / wf_u is also consistant over time t.

        Args:
            wgf_u (torch.tensor): wgf_u.
            wgr_x (torch.tensor): wgr_x.
            wf_u (torch.tensor): wf_u.
            wr_x (torch.tensor): wr_x.

        Returns:
            torch.tensor: Data after forward.
        """
        g_first_term = self.BN_gfu(wgf_u)
        g_second_term = self.BN_grx(wgr_x)
        g = F.sigmoid(g_first_term + g_second_term)

        x_first_term = self.BN_fu(wf_u)
        x_second_term = self.BN_Gx(self.BN_rx(wr_x) * g)
        x = F.relu(x_first_term + x_second_term)

        return x


class BasicBlock(nn.Module):
    """Basic module."""

    EXPANSION: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[bool] = None
    ):
        """Basic block constructor.

        Args:
            inplanes (int): Inplanes.
            planes (int): Planes.
            stride (int): Stride. Defaults to 1.
            downsample (bool, optional): Should use downsample. Defaults to None.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes: int, out_planes: int, stride: int = 1) -> torch.tensor:
        """3x3 convolution with padding.

        Args:
            in_planes (int): Input planes.
            out_planes (int): Output planes.
            stride (int): Stride.

        Returns:
            torch.tensor: Convolution result.
        """
        return nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

    def forward(self, data: torch.tensor) -> torch.tensor:
        """Forward method.

        Args:
            data (torch.tensor): Data to froward.

        Returns:
            torch.tensor: Data after forward.
        """
        residual = data

        out = self.conv1(data)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(data)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet implementation."""

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        block: Optional[nn.Module],
        layers: Optional[nn.Module]
    ):
        """ResNet constructor.

        Args:
            input_channel (int): Channels of input data. For RGB images
                input_channel == 3.
            output_channel (int): Channels of output tensor. It's important
                ro connect with other modules.
            block (nn.Module, optional): ResNet block.
            layers (nn.Module, optional): ResNet layer.
        """
        super(ResNet, self).__init__()

        self.output_channel_block = [
            int(output_channel / 4),
            int(output_channel / 2),
            output_channel, output_channel
        ]

        self.inplanes = int(output_channel / 8)
        self.conv0_1 = nn.Conv2d(
            input_channel,
            int(output_channel / 16),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))
        self.conv0_2 = nn.Conv2d(
            int(output_channel / 16),
            self.inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = nn.Conv2d(
            self.output_channel_block[0],
            self.output_channel_block[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
        self.conv2 = nn.Conv2d(
            self.output_channel_block[1],
            self.output_channel_block[1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.conv3 = nn.Conv2d(
            self.output_channel_block[2],
            self.output_channel_block[2],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

        self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)
        self.conv4_1 = nn.Conv2d(
            self.output_channel_block[3],
            self.output_channel_block[3],
            kernel_size=2,
            stride=(2, 1),
            padding=(0, 1),
            bias=False
        )
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(
            self.output_channel_block[3],
            self.output_channel_block[3],
            kernel_size=2,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

    def _make_layer(self, block: int, planes: int, blocks: int, stride=1) -> nn.Sequential:
        """Make Conv + BatchNorm layer.

        Args:
            block (int): Number of module in block.
            planes (int): Planes.
            blocks (int): Number of blocks.
            stride (int): Stride. Defaults to 1.

        Returns:
            nn.Sequential: Sequential layer.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.EXPANSION:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.EXPANSION,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.EXPANSION),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.EXPANSION
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, data: torch.tensor) -> torch.tensor:
        """Forward method.

        Args:
            data (torch.tensor): Data to froward.

        Returns:
            torch.tensor: Data after forward.
        """
        data = self.conv0_1(data)
        data = self.bn0_1(data)
        data = self.relu(data)
        data = self.conv0_2(data)
        data = self.bn0_2(data)
        data = self.relu(data)

        data = self.maxpool1(data)
        data = self.layer1(data)
        data = self.conv1(data)
        data = self.bn1(data)
        data = self.relu(data)

        data = self.maxpool2(data)
        data = self.layer2(data)
        data = self.conv2(data)
        data = self.bn2(data)
        data = self.relu(data)

        data = self.maxpool3(data)
        data = self.layer3(data)
        data = self.conv3(data)
        data = self.bn3(data)
        data = self.relu(data)

        data = self.layer4(data)
        data = self.conv4_1(data)
        data = self.bn4_1(data)
        data = self.relu(data)
        data = self.conv4_2(data)
        data = self.bn4_2(data)
        data = self.relu(data)

        return data


class SmallNet(nn.Module):
    """Small model without complicated modules.

    Allows to more simple convert to ONNX.
    """
    def __init__(self, input_channel: int, output_channel: int):
        """SmallNet Constructor.

        Args:
            input_channel (int): Channels of input data. For RGB images
                input_channel == 3.
            output_channel (int): Channels of output tensor. It's important
                to connect with other modules.
        """
        super(SmallNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=(3, 3))
        self.norm1 = nn.InstanceNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2)
        self.norm2 = nn.InstanceNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.norm3 = nn.InstanceNorm2d(64)
        self.conv4 = nn.Conv2d(64, output_channel, kernel_size=(3, 3), stride=2)
        self.norm4 = nn.InstanceNorm2d(output_channel)

    def forward(self, data: torch.tensor) -> torch.tensor:
        """Forward method.

        Args:
            data (torch.tensor): Data to froward.

        Returns:
            torch.tensor: Data after forward.
        """
        out = self.conv1(data)
        out = self.norm1(out)
        out = F.leaky_relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.leaky_relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        out = F.leaky_relu(out)
        out = self.conv4(out)
        out = self.norm4(out)
        out = F.leaky_relu(out)

        return out
