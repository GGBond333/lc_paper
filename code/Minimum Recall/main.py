# 更新 main.py 文件内容，移除下载部分，改为用户提供路径方式加载 Wafer 数据集
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import recall_score, accuracy_score
import numpy as np
import pandas as pd

# 使用用户提供路径加载数据集
def load_wafer_dataset(train_file, test_file):
    train_data = pd.read_csv(train_file, sep='\\t', header=None)
    y_train = train_data.iloc[:, 0].values
    X_train = train_data.iloc[:, 1:].values

    test_data = pd.read_csv(test_file, sep='\\t', header=None)
    y_test = test_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values

    print("\\n原始训练集类别分布:")
    unique_classes = np.unique(y_train)
    for c in unique_classes:
        print(f"  类别 {int(c)}: {np.sum(y_train == c)} 个样本")

    y_train = np.where(y_train > 0, 0, 1)
    y_test = np.where(y_test > 0, 0, 1)

    n_samples_train = X_train.shape[0]
    n_samples_test = X_test.shape[0]
    n_features = X_train.shape[1]

    X_train = X_train.reshape(n_samples_train, n_features, 1)
    X_test = X_test.reshape(n_samples_test, n_features, 1)

    print("\\n处理后的训练集类别分布:")
    for c in np.unique(y_train):
        print(f"  类别 {int(c)}: {np.sum(y_train == c)} 个样本")

    return X_train, y_train, X_test, y_test

# LSE最小召回率损失
class MinRecallLSELoss(nn.Module):
    def __init__(self, alpha=10.0, num_classes=2):
        super(MinRecallLSELoss, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, logits, targets):
        """
        logits: 模型输出，形状为 (batch, num_classes)
        targets: 标签，形状为 (batch,)
        """
        probs = F.softmax(logits, dim=1)  # 用softmax计算每类概率
        recalls = []

        for c in range(self.num_classes):
            mask = (targets == c).float()  # 取出类 c 的样本 mask, shape (batch,)
            if mask.sum() == 0:
                continue
            # 类 c 的 soft 预测概率
            prob_c = probs[:, c]  # shape (batch,)
            # 计算类 c 的“加权平均 recall”
            recall_c = (prob_c * mask).sum() / mask.sum()
            recalls.append(recall_c)

        recalls_tensor = torch.stack(recalls)
        lse = -1.0 / self.alpha * torch.logsumexp(-self.alpha * recalls_tensor, dim=0)
        return 1 - lse



# 简化MLSTM-FCN模型
class SimpleMLSTMFCN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleMLSTMFCN, self).__init__()
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True)
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=8)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_feat = lstm_out[:, -1, :]
        x_conv = x.permute(0, 2, 1)
        conv_feat = self.global_pool(F.relu(self.conv1(x_conv))).squeeze(-1)
        features = torch.cat([lstm_feat, conv_feat], dim=1)
        return self.fc(features)

# 训练和评估流程
def train_and_evaluate(X_train, y_train, X_test, y_test):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    model = SimpleMLSTMFCN(input_dim=1, num_classes=2)
    criterion = MinRecallLSELoss(alpha=10.0, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(100):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        acc = accuracy_score(y_test.numpy(), preds.numpy())
        recalls = [recall_score(y_test.numpy(), preds.numpy(), labels=[c], average='macro') for c in range(2)]
        print(f"Accuracy: {acc:.4f}, Min Recall: {min(recalls):.4f}")

# 主程序入口
if __name__ == "__main__":
    train_file = 'Wafer_TRAIN.tsv'
    test_file = 'Wafer_TEST.tsv'
    X_train, y_train, X_test, y_test = load_wafer_dataset(train_file, test_file)
    train_and_evaluate(X_train, y_train, X_test, y_test)
