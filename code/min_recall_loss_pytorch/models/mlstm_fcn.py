import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExciteBlock(nn.Module):
    """
    实现Squeeze-and-Excitation (SE)块
    SE块通过自适应地重新校准通道特征响应来提高模型性能
    """
    def __init__(self, channel, reduction=16):
        """
        初始化SE块
        
        参数:
            channel: 输入特征通道数
            reduction: 降维比例
        """
        super(SqueezeExciteBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征 [batch_size, channel, time_steps]
            
        返回:
            重新校准后的特征
        """
        b, c, _ = x.size()
        # 全局平均池化 - Squeeze操作
        y = self.avg_pool(x).view(b, c)
        # 全连接层 - Excitation操作
        y = self.fc(y).view(b, c, 1)
        # 重新校准特征 - Scale操作
        return x * y.expand_as(x)

class MLSTM_FCN(nn.Module):
    """
    实现MLSTM-FCN模型
    MLSTM-FCN结合了LSTM和全卷积网络，适用于时间序列分类
    """
    def __init__(self, input_dim, num_classes, hidden_dim=128, dropout=0.8):
        """
        初始化MLSTM-FCN模型
        
        参数:
            input_dim: 输入特征维度
            num_classes: 分类类别数
            hidden_dim: LSTM隐藏层维度
            dropout: Dropout比例
        """
        super(MLSTM_FCN, self).__init__()
        
        # LSTM分支
        self.lstm = nn.LSTM(input_size=input_dim, 
                           hidden_size=hidden_dim, 
                           batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
        # FCN分支
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=8, padding='same')
        self.bn1 = nn.BatchNorm1d(128)
        
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm1d(256)
        
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm1d(128)
        
        # Squeeze-and-Excitation块
        self.se = SqueezeExciteBlock(128)
        
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # 输出层
        self.fc = nn.Linear(hidden_dim + 128, num_classes)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入时间序列 [batch_size, time_steps, features]
            
        返回:
            分类预测
        """
        batch_size, time_steps, n_features = x.size()
        
        # LSTM分支
        lstm_out, _ = self.lstm(x)  # [batch_size, time_steps, hidden_dim]
        lstm_out = self.dropout(lstm_out[:, -1, :])  # 只取最后一个时间步的输出
        
        # FCN分支 - 需要转置输入以适应卷积层
        x_conv = x.permute(0, 2, 1)  # [batch_size, features, time_steps]
        
        # 第一个卷积块
        x_conv = self.conv1(x_conv)
        x_conv = self.bn1(x_conv)
        x_conv = F.relu(x_conv)
        
        # 第二个卷积块
        x_conv = self.conv2(x_conv)
        x_conv = self.bn2(x_conv)
        x_conv = F.relu(x_conv)
        
        # 第三个卷积块
        x_conv = self.conv3(x_conv)
        x_conv = self.bn3(x_conv)
        x_conv = F.relu(x_conv)
        
        # 应用SE块
        x_conv = self.se(x_conv)
        
        # 全局平均池化
        x_conv = self.gap(x_conv).view(batch_size, -1)
        
        # 合并LSTM和FCN分支
        x_combined = torch.cat([lstm_out, x_conv], dim=1)
        
        # 输出层
        output = self.fc(x_combined)
        
        return output