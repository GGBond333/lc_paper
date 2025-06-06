import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

def calculate_recalls(y_true, y_pred):
    """
    计算每个类别的召回率
    
    参数:
        y_true: 真实标签，形状为 [样本数, 类别数] 或 [样本数]
        y_pred: 预测标签，形状为 [样本数, 类别数] 或 [样本数]
        
    返回:
        每个类别的召回率列表
    """
    # 将one-hot编码转换为类别索引
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
        
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 计算每个类别的召回率
    recalls = []
    for i in range(cm.shape[0]):
        # 如果该类别没有样本，则设置召回率为1
        if np.sum(cm[i, :]) == 0:
            recalls.append(1.0)
        else:
            # 召回率 = 真阳性 / (真阳性 + 假阴性)
            recalls.append(cm[i, i] / np.sum(cm[i, :]))
    
    return recalls

def calculate_min_recall(y_true, y_pred):
    """
    计算最小召回率
    
    参数:
        y_true: 真实标签，形状为 [样本数, 类别数] 或 [样本数]
        y_pred: 预测标签，形状为 [样本数, 类别数] 或 [样本数]
        
    返回:
        最小召回率
    """
    recalls = calculate_recalls(y_true, y_pred)
    return np.min(recalls)

def calculate_accuracy(y_true, y_pred):
    """
    计算准确率
    
    参数:
        y_true: 真实标签，形状为 [样本数, 类别数] 或 [样本数]
        y_pred: 预测标签，形状为 [样本数, 类别数] 或 [样本数]
        
    返回:
        准确率
    """
    # 将one-hot编码转换为类别索引
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
        
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    return accuracy_score(y_true, y_pred)

def evaluate_model(model, test_loader, device):
    """
    评估模型性能
    
    参数:
        model: PyTorch模型
        test_loader: 测试数据加载器
        device: 计算设备
        
    返回:
        (accuracy, min_recall, recalls, auc, f1, gmean, precision) 准确率、最小召回率、每个类别的召回率、AUC、F1分数、G-mean和精确率
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(torch.argmax(targets, dim=1).cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    accuracy = accuracy_score(all_targets, all_preds)
    recalls = calculate_recalls(all_targets, all_preds)
    min_recall = np.min(recalls)

    # 计算F1分数
    f1 = f1_score(all_targets, all_preds, average='binary')

    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        gmean = np.sqrt(recall * specificity)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    else:
        recall = specificity = gmean = precision = 0.0

    y_probs = []
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            prob = F.softmax(outputs, dim=1)
            y_probs.append(prob.cpu().numpy())
    y_probs = np.concatenate(y_probs)
    try:
        auc = roc_auc_score(all_targets, y_probs[:, 1])  # 假设类别0是负类，1是正类
    except ValueError:
        auc = 0.0  # 防止某类样本全为空时异常

    return accuracy, min_recall, recalls, auc, f1, gmean, precision

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=50, patience=10, val_ratio=0.2):
    """
    训练模型
    
    参数:
        model: PyTorch模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
        num_epochs: 训练轮数
        patience: 早停耐心值
        val_ratio: 验证集比例
        
    返回:
        训练历史记录
    """
    # 从训练集中划分出验证集
    dataset = train_loader.dataset
    n_samples = len(dataset)
    indices = list(range(n_samples))
    np.random.shuffle(indices)
    val_size = int(n_samples * val_ratio)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # 创建训练集和验证集的数据加载器
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=train_loader.batch_size,
        sampler=train_sampler
    )
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_loader.batch_size,
        sampler=val_sampler
    )
    
    # 初始化最佳验证损失和耐心计数器
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # 初始化历史记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_min_recall': []
    }
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        # 计算平均训练损失
        train_loss = train_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(torch.argmax(targets, dim=1).cpu().numpy())
        
        # 计算验证指标
        val_loss = val_loss / len(val_loader.dataset)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        val_accuracy = accuracy_score(all_targets, all_preds)
        val_min_recall = calculate_min_recall(all_targets, all_preds)
        
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_min_recall'].append(val_min_recall)
        
        # 打印训练进度
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Val Accuracy: {val_accuracy:.4f}')
        print(f'  Val Min Recall: {val_min_recall:.4f}')
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history

def plot_training_history(history, title='Training History', save_path=None):
    """
    绘制训练历史
    
    参数:
        history: 训练历史记录
        title: 图表标题
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 4))
    
    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率和最小召回率
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'], label='Val Accuracy')
    plt.plot(history['val_min_recall'], label='Val Min Recall')
    plt.title('Model Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()