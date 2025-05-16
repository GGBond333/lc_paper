import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from time_series_classifier import TimeSeriesClassifier, train_model, evaluate_model
import matplotlib.pyplot as plt
import pandas as pd

def generate_imbalanced_time_series(n_samples=1000, seq_len=50, n_features=1, n_classes=2, imbalance_ratio=0.1):
    """
    生成合成的不平衡时间序列数据集。
    
    参数:
        n_samples: 样本总数
        seq_len: 每个时间序列的长度
        n_features: 每个时间点的特征数
        n_classes: 类别数
        imbalance_ratio: 少数类样本与总样本的比例
    
    返回:
        X: 形状为[n_samples, seq_len, n_features]的时间序列数据
        y: 形状为[n_samples]的标签
    """
    X = np.zeros((n_samples, seq_len, n_features))
    y = np.zeros(n_samples)
    
    # 计算每个类别的样本数
    samples_per_class = np.ones(n_classes)
    samples_per_class[0] = int(n_samples * (1 - imbalance_ratio))  # 多数类
    samples_per_class[1:] = int(n_samples * imbalance_ratio / (n_classes - 1))  # 少数类
    samples_per_class = samples_per_class.astype(int)
    
    # 如果总数与n_samples不匹配，则调整
    samples_per_class[0] += n_samples - np.sum(samples_per_class)
    
    # 为每个类别生成数据
    current_idx = 0
    for c in range(n_classes):
        n_class_samples = samples_per_class[c]
        
        # 为每个类别生成不同的模式
        if c == 0:  # 多数类 - 正弦波
            for i in range(n_class_samples):
                freq = np.random.uniform(0.1, 0.5)
                phase = np.random.uniform(0, 2*np.pi)
                amplitude = np.random.uniform(0.5, 2.0)
                
                t = np.linspace(0, 10, seq_len)
                signal = amplitude * np.sin(2 * np.pi * freq * t + phase)
                
                # 添加噪声
                noise = np.random.normal(0, 0.1, seq_len)
                signal += noise
                
                X[current_idx, :, 0] = signal
                y[current_idx] = c
                current_idx += 1
        
        elif c == 1:  # 少数类 - 方波
            for i in range(n_class_samples):
                freq = np.random.uniform(0.1, 0.3)
                phase = np.random.uniform(0, 2*np.pi)
                amplitude = np.random.uniform(0.5, 2.0)
                
                t = np.linspace(0, 10, seq_len)
                signal = amplitude * np.sign(np.sin(2 * np.pi * freq * t + phase))
                
                # 添加噪声
                noise = np.random.normal(0, 0.1, seq_len)
                signal += noise
                
                X[current_idx, :, 0] = signal
                y[current_idx] = c
                current_idx += 1
        
        else:  # 其他少数类 - 不同的模式
            for i in range(n_class_samples):
                # 随机游走
                signal = np.cumsum(np.random.normal(0, 0.1, seq_len))
                
                # 归一化
                signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) * 2 - 1
                
                X[current_idx, :, 0] = signal
                y[current_idx] = c
                current_idx += 1
    
    # 打乱数据
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y

def visualize_dataset(X, y, n_samples=5):
    """
    可视化数据集中的一些示例。
    """
    plt.figure(figsize=(12, 8))
    classes = np.unique(y)
    
    for c in classes:
        class_indices = np.where(y == c)[0]
        samples_to_plot = min(n_samples, len(class_indices))
        
        for i in range(samples_to_plot):
            plt.subplot(len(classes), n_samples, int(c) * n_samples + i + 1)
            plt.plot(X[class_indices[i], :, 0])
            plt.title(f'Class {int(c)}')
            if i == 0:
                plt.ylabel('Amplitude')
            if int(c) == len(classes) - 1:
                plt.xlabel('Time')
    
    plt.tight_layout()
    plt.savefig('dataset_visualization.png')
    plt.close()

def visualize_tsne(features, labels):
    """
    使用t-SNE可视化学习到的特征。
    """
    from sklearn.manifold import TSNE
    
    # 应用t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    classes = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
    
    for c, color in zip(classes, colors):
        class_indices = np.where(labels == c)[0]
        plt.scatter(features_2d[class_indices, 0], features_2d[class_indices, 1], 
                   color=color, label=f'Class {int(c)}', alpha=0.7)
    
    plt.legend()
    plt.title('t-SNE Visualization of Learned Features')
    plt.savefig('tsne_visualization.png')
    plt.close()

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
    y_train = train_data.iloc[:, 0].values   # 第一列是标签
    X_train = train_data.iloc[:, 1:].values   # 后面是特征
    
    # 读取测试数据
    test_data = pd.read_csv(test_file, sep='\t', header=None)
    y_test = test_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values
    
    # 打印原始类别分布
    print("\n原始训练集类别分布:")
    unique_classes = np.unique(y_train)
    for c in unique_classes:
        print(f"  类别 {int(c)}: {np.sum(y_train == c)} 个样本")
    
    # 将类别重新映射为二分类问题
    # 将所有负类映射为0，所有正类映射为1
    # 显式将正类设置为 0，负类为 1（符合论文设定）
    #y_train = np.where(y_train > 0, 0, 1)
    #y_test = np.where(y_test > 0, 0, 1)
    # 指定要标记为1的类别列表
    #min_classes = [1]
    # 创建一个新的标签数组，将指定的类别标记为1，其他类别标记为0
    #y_train = np.where(np.isin(y_train, min_classes), 1, 0)
    #y_test = np.where(np.isin(y_test, min_classes), 1, 0)

    # 重塑数据为3D张量 [samples, sequence_length, features]
    n_samples_train = X_train.shape[0]
    n_samples_test = X_test.shape[0]
    seq_len = 1  # 每个样本是一个时间点
    n_features = X_train.shape[1]  # 特征数

    # 将每个样本看作长度为 n_features 的时间序列，单通道特征
    X_train = X_train.reshape(n_samples_train, n_features, 1)
    X_test = X_test.reshape(n_samples_test, n_features, 1)

    # 打印处理后的类别分布
    print("\n处理后的训练集类别分布:")
    for c in np.unique(y_train):
        print(f"  类别 {int(c)}: {np.sum(y_train == c)} 个样本")
    
    return X_train, y_train, X_test, y_test

def main():
    # 设置随机种子以确保可重现性
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 加载Wafer数据集
    print("加载数据集...")
    X_train, y_train, X_test, y_test = load_wafer_dataset(
        'dataset/UWaveGestureLibraryAll_TRAIN.tsv',
        'dataset/UWaveGestureLibraryAll_TEST.tsv'
    )
    
    # 分割训练集为训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_train,
        shuffle=True
    )
    
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
    
    # 打印类别分布
    print("\n最终训练集类别分布:")
    for c in np.unique(y_train):
        print(f"  类别 {int(c)}: {np.sum(y_train == c)} 个样本")
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 获取输入特征维度
    input_dim = X_train.shape[2]  # 特征维度
    
    model = TimeSeriesClassifier(
        input_dim=input_dim,      # 输入特征维度
        hidden_dim=64,            # LSTM隐藏层维度
        feature_dim=32,           # 特征维度
        num_classes=2,            # 二分类问题
        temperature=0.5,          # 温度参数
        alpha=1.0                 # 缩放因子
    )
    
    # 训练模型
    print("\n使用COCL损失函数训练模型...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=20,
        device=device
    )
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pt'))
    
    # 在测试集上评估
    print("\n在测试集上评估...")
    results = evaluate_model(model, test_loader, device)
    
    print(f"测试准确率: {results['accuracy']:.4f}")
    print(f"测试最小召回率: {results['min_recall']:.4f}")
    if results['auc'] > 0:
        print(f"测试AUC: {results['auc']:.4f}")
        print(f"测试F1: {results['f1']:.4f}")
        print(f"测试G-mean: {results['gmean']:.4f}")
    # 提取特征进行可视化
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            _, features, _, _ = model(data)
            all_features.append(features.cpu().numpy())
            all_labels.append(target.numpy())
    
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)
    
    # 可视化学习到的特征
    visualize_tsne(all_features, all_labels)
    print("\n可视化已保存到 'tsne_visualization.png'")

if __name__ == "__main__":
    main()