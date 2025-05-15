import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
from time_series_classifier import TimeSeriesClassifier, train_model, evaluate_model
import pandas as pd
import os

class StandardCrossEntropyLoss(nn.Module):
    """标准交叉熵损失。"""
    def __init__(self):
        super(StandardCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, features, logits, labels):
        return self.criterion(logits, labels)

class WeightedCrossEntropyLoss(nn.Module):
    """加权交叉熵损失，用于不平衡数据。"""
    def __init__(self, class_weights):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
    def forward(self, features, logits, labels):
        return self.criterion(logits, labels)

class FocalLoss(nn.Module):
    """Focal损失，用于不平衡数据。"""
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma  # 聚焦参数，增加对难分类样本的关注
        self.alpha = alpha  # 类别权重
        
    def forward(self, features, logits, labels):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(logits, labels)
        pt = torch.exp(-ce_loss)  # 预测概率
        # Focal Loss公式: (1-pt)^gamma * ce_loss
        loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[labels]
            loss = alpha_t * loss
            
        return loss.mean()

def load_wafer_dataset(train_file, test_file):
    """
    加载Wafer数据集。
    
    参数:
        train_file: 训练数据文件路径
        test_file: 测试数据文件路径
    
    返回:
        X_train: 训练数据
        y_train: 训练标签
        X_test: 测试数据
        y_test: 测试标签
    """
    # 读取训练数据
    train_data = pd.read_csv(train_file, sep='\t', header=None)
    y_train = train_data.iloc[:, 0].values  # 第一列是标签
    X_train = train_data.iloc[:, 1:].values  # 后面是特征

    # 读取测试数据
    test_data = pd.read_csv(test_file, sep='\t', header=None)
    y_test = test_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values
    
    # 将类别重新映射为二分类问题
    # 将所有负类映射为0，所有正类映射为1
    y_train = np.where(y_train < 0, 0, 1)
    y_test = np.where(y_test < 0, 0, 1)
    
    # 重塑数据为3D张量 [samples, sequence_length, features]
    n_samples_train = X_train.shape[0]
    n_samples_test = X_test.shape[0]
    seq_len = 1  # 每个样本是一个时间点
    n_features = X_train.shape[1]  # 特征数
    
    X_train = X_train.reshape(n_samples_train, seq_len, n_features)
    X_test = X_test.reshape(n_samples_test, seq_len, n_features)
    
    return X_train, y_train, X_test, y_test

def compare_loss_functions(X_train, y_train, X_test, y_test, device='cuda'):
    """
    在同一数据集上比较不同的损失函数。
    """
    # 创建保存模型的目录
    model_dir = 'saved_models'
    os.makedirs(model_dir, exist_ok=True)
    
    # 分割训练集为训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_train,
        shuffle=True
    )
    
    # 打印数据集信息
    print("\n训练集类别分布:")
    for c in np.unique(y_train):
        print(f"  类别 {int(c)}: {np.sum(y_train == c)} 个样本")
    
    print("\n验证集类别分布:")
    for c in np.unique(y_val):
        print(f"  类别 {int(c)}: {np.sum(y_val == c)} 个样本")
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # 根据数据集大小调整batch_size
    batch_size = min(32, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 计算加权损失的类别权重
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts  # 反比于类别频率
    class_weights = class_weights / np.sum(class_weights) * len(class_counts)  # 归一化
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    # 定义要比较的损失函数
    loss_functions = {
        'Standard CE': StandardCrossEntropyLoss(),  # 标准交叉熵
        'Weighted CE': WeightedCrossEntropyLoss(class_weights_tensor),  # 加权交叉熵
        'Focal Loss': FocalLoss(gamma=2.0),  # Focal损失
        'COCL': None  # COCL集成在模型中
    }
    
    # 结果字典
    results = {name: {'accuracy': [], 'min_recall': [], 'auc': [], 'f1': [], 'gmean': []} for name in loss_functions.keys()}
    
    # 使用每个损失函数训练和评估
    for name, loss_fn in loss_functions.items():
        print(f"\n使用 {name} 损失函数训练...")
        
        # 初始化模型
        if name == 'COCL':
            model = TimeSeriesClassifier(
                input_dim=X_train.shape[2],
                hidden_dim=64,  # 使用原始维度
                feature_dim=32,  # 使用原始维度
                num_classes=2,  # 二分类问题
                temperature=0.5,
                alpha=1.0
            )
        else:
            # 对于其他损失函数，我们将使用自定义训练循环
            model = TimeSeriesClassifier(
                input_dim=X_train.shape[2],
                hidden_dim=64,  # 使用原始维度
                feature_dim=32,  # 使用原始维度
                num_classes=2,  # 二分类问题
                temperature=0.5,
                alpha=0.0  # 禁用COCL损失
            )
        
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用原始学习率
        
        # 模型保存路径
        model_path = os.path.join(model_dir, f'best_model_{name.replace(" ", "_")}.pt')
        
        if name == 'COCL':
            # 对于COCL，使用内置的训练函数
            model = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                num_epochs=50,
                device=device
            )
            # 保存COCL模型
            torch.save(model.state_dict(), model_path)
        else:
            # 对于其他损失函数，使用自定义训练循环
            best_val_acc = 0.0
            
            for epoch in range(50):
                # 训练阶段
                model.train()
                train_loss = 0.0
                train_acc = 0.0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    
                    optimizer.zero_grad()
                    logits, features, _, _ = model(data)
                    
                    # 使用特定的损失函数计算损失
                    loss = loss_fn(features, logits, target)
                    
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
                val_acc = 0.0
                val_preds = []
                val_targets = []
                
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(device), target.to(device)
                        
                        logits, _, _, _ = model(data)
                        
                        # 计算准确率
                        _, predicted = torch.max(logits, 1)
                        acc = (predicted == target).float().mean()
                        
                        val_acc += acc.item()
                        val_preds.extend(predicted.cpu().numpy())
                        val_targets.extend(target.cpu().numpy())
                
                val_acc /= len(val_loader)
                
                # 计算最小召回率
                val_targets_np = np.array(val_targets)
                val_preds_np = np.array(val_preds)
                class_recalls = []
                
                for c in np.unique(val_targets_np):
                    class_mask = (val_targets_np == c)
                    if np.sum(class_mask) > 0:
                        class_recall = np.sum((val_preds_np == c) & class_mask) / np.sum(class_mask)
                        class_recalls.append(class_recall)
                
                min_recall = np.min(class_recalls) if class_recalls else 0.0
                
                print(f'Epoch {epoch+1}/50:')
                print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
                print(f'  Val Acc: {val_acc:.4f}, Min Recall: {min_recall:.4f}')
                
                # 保存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), model_path)
        
        # 加载最佳模型
        try:
            model.load_state_dict(torch.load(model_path))
            print(f"成功加载模型: {model_path}")
        except FileNotFoundError:
            print(f"警告: 找不到模型文件 {model_path}，使用当前模型状态")
        
        # 在测试集上评估
        print(f"\n在测试集上评估 {name}...")
        test_results = evaluate_model(model, test_loader, device)
        
        results[name]['accuracy'] = test_results['accuracy']
        results[name]['min_recall'] = test_results['min_recall']
        results[name]['auc'] = test_results.get('auc', 0.0)
        results[name]['f1'] = test_results.get('f1', 0.0)
        results[name]['gmean'] = test_results.get('gmean', 0.0)
        print(f"测试准确率: {test_results['accuracy']:.4f}")
        print(f"测试最小召回率: {test_results['min_recall']:.4f}")
        print(f"测试AUC: {test_results.get('auc', 0.0):.4f}")
        print(f"测试F1: {test_results.get('f1', 0.0):.4f}")
        print(f"测试G-mean: {test_results.get('gmean', 0.0):.4f}")
    
    # 绘制结果
    plt.figure(figsize=(20, 5))
    
    # 准确率图
    plt.subplot(1, 5, 1)
    names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in names]
    plt.bar(names, accuracies)
    plt.title('测试准确率')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # 最小召回率图
    plt.subplot(1, 5, 2)
    min_recalls = [results[name]['min_recall'] for name in names]
    plt.bar(names, min_recalls)
    plt.title('测试最小召回率')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # AUC图
    plt.subplot(1, 5, 3)
    aucs = [results[name]['auc'] for name in names]
    plt.bar(names, aucs)
    plt.title('测试AUC')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # F1图
    plt.subplot(1, 5, 4)
    f1s = [results[name]['f1'] for name in names]
    plt.bar(names, f1s)
    plt.title('测试F1')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # G-mean图
    plt.subplot(1, 5, 5)
    gmeans = [results[name]['gmean'] for name in names]
    plt.bar(names, gmeans)
    plt.title('测试G-mean')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('loss_comparison.png')
    plt.close()
    
    return results

def main():
    # 设置随机种子以确保可重现性
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 加载Wafer数据集
    print("加载Wafer数据集...")
    X_train, y_train, X_test, y_test = load_wafer_dataset(
        'Wafer_TRAIN.tsv',
        'Wafer_TEST.tsv'
    )
    
    # 打印数据集信息
    print("\n数据集信息:")
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    print(f"特征维度: {X_train.shape[2]}")
    print("\n类别分布:")
    for c in np.unique(y_train):
        print(f"  类别 {int(c)}: {np.sum(y_train == c)} 个样本")
    
    # 比较不同损失函数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    results = compare_loss_functions(X_train, y_train, X_test, y_test, device)
    
    # 打印最终结果
    print("\n最终结果:")
    for name in results:
        print(f"\n{name}:")
        print(f"  准确率: {results[name]['accuracy']:.4f}")
        print(f"  最小召回率: {results[name]['min_recall']:.4f}")
        print(f"  AUC: {results[name]['auc']:.4f}")
        print(f"  F1: {results[name]['f1']:.4f}")
        print(f"  G-mean: {results[name]['gmean']:.4f}")

if __name__ == "__main__":
    main()