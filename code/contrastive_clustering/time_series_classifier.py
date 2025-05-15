import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from cocl_loss import ContrastiveClusteringLoss
import torch.nn.functional as F


class MLSTMFCNEncoder(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim=64, conv_out_channels=128):
        super(MLSTMFCNEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_dim, batch_first=True)

        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=conv_out_channels, kernel_size=8, padding=4)
        self.bn1 = nn.BatchNorm1d(conv_out_channels)

        self.conv2 = nn.Conv1d(in_channels=conv_out_channels, out_channels=conv_out_channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(conv_out_channels)

        self.conv3 = nn.Conv1d(in_channels=conv_out_channels, out_channels=conv_out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(conv_out_channels)

        self.global_pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):  # x: [batch_size, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim]
        lstm_feat = lstm_out[:, -1, :]  # 最后时间步的隐藏状态

        # FCN 分支: 需要将 input transpose 成 [batch_size, input_dim, seq_len]
        x_permute = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x_permute)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        conv_feat = self.global_pooling(x).squeeze(-1)  # [batch_size, out_channels]

        out = torch.cat([lstm_feat, conv_feat], dim=1)  # 拼接 LSTM 与 FCN 特征
        return out  # [batch_size, lstm_hidden + conv_out]

class TimeSeriesEncoder(nn.Module):
    """
    时间序列数据的编码器网络。
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(TimeSeriesEncoder, self).__init__()
        # 双向LSTM层，用于捕获时间序列的时间依赖性
        self.lstm = nn.LSTM(
            input_size=input_dim,      # 输入特征维度
            hidden_size=hidden_dim,    # 隐藏层维度
            num_layers=num_layers,     # LSTM层数
            batch_first=True,          # 输入形状为[batch_size, seq_len, input_dim]
            bidirectional=True         # 使用双向LSTM，可以捕获前向和后向的时间依赖
        )
        # 全连接层，将LSTM的输出映射到所需的输出维度
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2是因为双向LSTM
        
    def forward(self, x):
        # x形状: [batch_size, seq_len, input_dim]
        output, (hidden, _) = self.lstm(x)
        
        # 连接最终的前向和后向隐藏状态
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # 投影到输出维度
        x = self.fc(hidden)
        return x

class TimeSeriesClassifier(nn.Module):
    """
    使用COCL损失的时间序列分类器。
    """
    def __init__(self, input_dim, hidden_dim, feature_dim, num_classes, temperature=0.5, alpha=1.0):
        super(TimeSeriesClassifier, self).__init__()
        # 编码器，用于提取时间序列特征
        self.encoder = TimeSeriesEncoder(input_dim, hidden_dim, feature_dim)
        # self.encoder = MLSTMFCNEncoder(input_dim=input_dim)
        # self.feature_dim = 64 + 128  # lstm_hidden_dim + conv_out_channels
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        # 分类器，用于预测类别
        # self.classifier = nn.Linear(feature_dim, num_classes)
        # COCL损失函数
        self.cocl_loss = ContrastiveClusteringLoss(temperature, alpha)
        
    def forward(self, x, labels=None):
        # 提取特征
        features = self.encoder(x)
        
        # 获取类别预测
        logits = self.classifier(features)
        
        # 如果在训练中且提供了标签，则计算COCL损失
        loss = None
        loss_components = {}
        if self.training and labels is not None:
            loss, loss_components = self.cocl_loss(features, labels)
        
        return logits, features, loss, loss_components

def train_model(model, train_loader, val_loader, optimizer, num_epochs=100, device='cuda'):
    """
    训练时间序列分类器。
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 使用的设备（'cuda'或'cpu'）
    
    返回:
        训练好的模型
    """
    model.to(device)
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()  # 清除梯度
            logits, features, cocl_loss_val, loss_components = model(data, target)
            loss_ce = F.cross_entropy(logits, target)
            loss = cocl_loss_val + loss_ce  # 总损失

            # 计算准确率
            _, predicted = torch.max(logits, 1)
            acc = (predicted == target).float().mean()
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += acc.item()
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                logits, features, _, _ = model(data)
                
                # 计算准确率
                _, predicted = torch.max(logits, 1)
                acc = (predicted == target).float().mean()
                
                val_acc += acc.item()
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        val_acc /= len(val_loader)
        
        # 计算最小召回率（对于不平衡数据集）
        val_targets_np = np.array(val_targets)
        val_preds_np = np.array(val_preds)
        class_recalls = []
        
        # 计算每个类别的召回率
        for c in np.unique(val_targets_np):
            class_mask = (val_targets_np == c)
            if np.sum(class_mask) > 0:
                class_recall = np.sum((val_preds_np == c) & class_mask) / np.sum(class_mask)
                class_recalls.append(class_recall)
        
        # 最小召回率是所有类别召回率中的最小值
        min_recall = np.min(class_recalls) if class_recalls else 0.0
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Acc: {val_acc:.4f}, Min Recall: {min_recall:.4f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')
    
    return model

def evaluate_model(model, test_loader, device='cuda'):
    """
    评估时间序列分类器。
    
    参数:
        model: 要评估的模型
        test_loader: 测试数据加载器
        device: 使用的设备（'cuda'或'cpu'）
    
    返回:
        包含评估指标的字典
    """
    model.to(device)
    model.eval()
    
    test_preds = []
    test_targets = []
    all_probs = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits, _, _, _ = model(data)

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred_labels = torch.argmax(logits, dim=1).cpu().numpy()

            all_probs.extend(probs)
            test_preds.extend(pred_labels)
            test_targets.extend(target.cpu().numpy())

    test_targets_np = np.array(test_targets)
    test_preds_np = np.array(test_preds)
    all_probs_np = np.array(all_probs)
    
    # Calculate metrics
    test_acc = accuracy_score(test_targets_np, test_preds_np)
    f1 = f1_score(test_targets_np, test_preds_np, average='binary')
    
    # Calculate AUC using all probabilities
    auc = roc_auc_score(test_targets_np, all_probs_np)
    
    # 计算混淆矩阵
    cm = confusion_matrix(test_targets_np, test_preds_np, labels=[0, 1])
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        gmean = np.sqrt(recall * specificity)
    else:
        recall = specificity = gmean = 0.0

    # 最小召回率
    class_recalls = []
    for c in np.unique(test_targets_np):
        class_mask = (test_targets_np == c)
        if np.sum(class_mask) > 0:
            class_recall = np.sum((test_preds_np == c) & class_mask) / np.sum(class_mask)
            class_recalls.append(class_recall)
    min_recall = np.min(class_recalls) if class_recalls else 0.0

    return {
        'accuracy': test_acc,
        'min_recall': min_recall,
        'auc': auc,
        'f1': f1,
        'gmean': gmean,
        'predictions': test_preds,
        'targets': test_targets
    }