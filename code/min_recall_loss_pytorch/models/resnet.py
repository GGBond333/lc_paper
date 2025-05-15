import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    实现ResNet的残差块
    残差块通过跳跃连接帮助解决深度网络的梯度消失问题
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        """
        初始化残差块
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 卷积步长
        """
        super(ResidualBlock, self).__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # 第二个卷积层
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, 
                              padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 如果输入和输出维度不同，则使用1x1卷积进行调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征
            
        返回:
            残差块的输出
        """
        # 保存输入用于跳跃连接
        identity = x
        
        # 第一个卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # 第二个卷积层
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 跳跃连接
        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out

class ResNet(nn.Module):
    """
    实现ResNet模型
    ResNet是一种使用残差连接的深度卷积神经网络，适用于时间序列分类
    """
    def __init__(self, input_dim, num_classes):
        """
        初始化ResNet模型
        
        参数:
            input_dim: 输入特征维度
            num_classes: 分类类别数
        """
        super(ResNet, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # 残差层
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """
        创建包含多个残差块的层
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            blocks: 残差块数量
            stride: 第一个残差块的步长
            
        返回:
            包含多个残差块的Sequential模块
        """
        layers = []
        
        # 第一个残差块可能改变维度
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
        
        # 后续残差块保持维度不变
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入时间序列 [batch_size, time_steps, features]
            
        返回:
            分类预测
        """
        # 转置输入以适应卷积层
        x = x.permute(0, 2, 1)  # [batch_size, features, time_steps]
        
        # 初始层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 全局平均池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 全连接层
        x = self.fc(x)
        
        return x