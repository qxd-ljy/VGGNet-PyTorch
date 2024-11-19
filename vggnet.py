import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, num_classes=10, input_channels=1):
        """
        VGG 网络的初始化方法，包含卷积层和全连接层。

        参数：
        - num_classes (int): 分类的类别数量，默认 10 (适用于 MNIST)
        - input_channels (int): 输入图片的通道数，默认 1 (适用于灰度图像)
        """
        super(VGG, self).__init__()

        # 构建卷积层部分
        self.features = self._make_layers(input_channels)

        # 构建分类器部分
        self.classifier = self._make_classifier(num_classes)

    def _make_layers(self, input_channels):
        """
        构建卷积层部分，通过堆叠卷积层、ReLU 激活和池化层来构建特征提取部分

        参数：
        - input_channels (int): 输入图像的通道数，默认为 1（灰度图）

        返回：
        - features (nn.Sequential): 包含卷积层和池化层的神经网络模块
        """
        layers = []
        # 卷积块 1
        layers += self._conv_block(input_channels, 64)
        # 卷积块 2
        layers += self._conv_block(64, 128)
        # 卷积块 3
        layers += self._conv_block(128, 256)
        # 卷积块 4
        layers += self._conv_block(256, 512)

        # 将所有卷积块和池化层堆叠在一起
        return nn.Sequential(*layers)

    def _conv_block(self, in_channels, out_channels):
        """
        创建一个卷积块，包含两个卷积层和一个最大池化层

        参数：
        - in_channels (int): 输入通道数
        - out_channels (int): 输出通道数

        返回：
        - block (list): 卷积块 [卷积层 + ReLU + 卷积层 + ReLU + 最大池化层]
        """
        block = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        return block

    def _make_classifier(self, num_classes):
        """
        构建全连接层部分，最后的输出层为分类层。

        参数：
        - num_classes (int): 分类类别数

        返回：
        - classifier (nn.Sequential): 包含全连接层和 Dropout 层的网络模块
        """
        return nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        """
        前向传播方法，输入图像通过卷积层提取特征后再通过全连接层进行分类。

        参数：
        - x (Tensor): 输入的图像数据

        返回：
        - x (Tensor): 分类结果
        """
        # 通过卷积层提取特征
        x = self.features(x)

        # 将特征图展平为一维向量
        x = x.view(x.size(0), -1)  # 这里将 4D 张量转换为 2D，保留 batch_size

        # 通过分类器进行最终分类
        x = self.classifier(x)

        return x
