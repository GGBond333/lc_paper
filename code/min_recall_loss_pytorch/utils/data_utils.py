import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.datasets import make_classification

class TimeSeriesDataset(Dataset):
    """
    时间序列数据集类
    用于PyTorch的DataLoader
    """
    def __init__(self, X, y):
        """
        初始化数据集
        
        参数:
            X: 时间序列数据，形状为 [样本数, 时间步长, 特征数]
            y: 标签，形状为 [样本数, 类别数]（one-hot编码）
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.X)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        return self.X[idx], self.y[idx]

def load_and_preprocess_data(X_train, y_train, X_test=None, y_test=None, test_size=0.2, random_state=42, batch_size=32):
    """
    加载和预处理时间序列数据
    
    参数:
        X_train: 训练数据，形状为 [样本数, 时间步长, 特征数]
        y_train: 训练标签，形状为 [样本数]
        X_test: 测试数据，如果为None则从训练集划分
        y_test: 测试标签，如果为None则从训练集划分
        test_size: 测试集比例（仅当X_test为None时使用）
        random_state: 随机种子
        batch_size: 批量大小
        
    返回:
        (train_loader, test_loader, X_train, X_test, y_train_onehot, y_test_onehot)
    """
    # 如果没有提供测试集，则从训练集划分
    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y_train
        )
    
    # 标准化特征
    # 对每个特征维度分别标准化
    for i in range(X_train.shape[2]):
        scaler = StandardScaler()
        X_train[:, :, i] = scaler.fit_transform(X_train[:, :, i])
        X_test[:, :, i] = scaler.transform(X_test[:, :, i])
    
    # 将标签转换为one-hot编码
    encoder = OneHotEncoder(sparse=False)
    y_train_reshaped = y_train.reshape(-1, 1)
    y_test_reshaped = y_test.reshape(-1, 1)
    y_train_onehot = encoder.fit_transform(y_train_reshaped)
    y_test_onehot = encoder.transform(y_test_reshaped)
    
    # 创建PyTorch数据集
    train_dataset = TimeSeriesDataset(X_train, y_train_onehot)
    test_dataset = TimeSeriesDataset(X_test, y_test_onehot)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, X_train, X_test, y_train_onehot, y_test_onehot

def create_imbalanced_dataset(X, y, imbalance_ratio=0.1, minority_class=1, random_state=42):
    """
    创建不平衡数据集
    
    参数:
        X: 时间序列数据，形状为 [样本数, 时间步长, 特征数]
        y: 标签，形状为 [样本数]
        imbalance_ratio: 少数类与多数类的比例
        minority_class: 少数类的标签
        random_state: 随机种子
        
    返回:
        (X_imbalanced, y_imbalanced)
    """
    np.random.seed(random_state)
    
    # 找出少数类和多数类的索引
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y != minority_class)[0]
    
    # 计算需要保留的少数类样本数量
    n_majority = len(majority_indices)
    n_minority = int(n_majority * imbalance_ratio)
    
    # 随机选择少数类样本
    selected_minority_indices = np.random.choice(minority_indices, size=n_minority, replace=False)
    
    # 合并多数类和选择的少数类样本
    selected_indices = np.concatenate([majority_indices, selected_minority_indices])
    np.random.shuffle(selected_indices)
    
    X_imbalanced = X[selected_indices]
    y_imbalanced = y[selected_indices]
    
    return X_imbalanced, y_imbalanced


def hellinger(p, q):
    """
    计算两个概率分布之间的 Hellinger 距离。
    参数:
        p, q: 一维 numpy 数组，必须是归一化的概率分布（sum=1）
    返回:
        Hellinger 距离（0 到 1 之间）
    """
    p = np.asarray(p)
    q = np.asarray(q)

    # 若未归一化，则归一化
    p = p / p.sum()
    q = q / q.sum()

    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))

def calculate_imbalance_degree(y):
    """
    计算数据集的不平衡度 (ID)
    
    参数:
        y: 标签，形状为 [样本数]
        
    返回:
        不平衡度 (ID)
    """
    # from scipy.spatial.distance import hellinger
    
    # 计算类别分布
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    
    # 计算经验分布
    empirical_dist = counts / np.sum(counts)
    
    # 计算均匀分布
    uniform_dist = np.ones(n_classes) / n_classes
    
    # 计算少数类的数量
    minority_classes = np.sum(empirical_dist < 1/n_classes)
    
    # 计算Hellinger距离
    # 由于scipy的hellinger函数需要两个分布的平方根，我们需要先计算
    dist = hellinger(np.sqrt(empirical_dist), np.sqrt(uniform_dist))
    
    # 计算最大可能的Hellinger距离（当有k个少数类时）
    # 这部分在论文中有详细描述，但实现较为复杂
    # 这里我们使用一个简化的计算方式
    max_dist = np.sqrt(2 - 2 * np.sqrt(1/n_classes))
    
    # 计算不平衡度
    id_value = (dist / max_dist) + (minority_classes - 1)
    
    return id_value

def generate_synthetic_time_series(n_samples=1000, n_features=3, n_classes=2, n_timesteps=100, class_sep=1.0, imbalance_ratio=0.1):
    """
    生成合成的时间序列数据
    
    参数:
        n_samples: 样本数量
        n_features: 特征数量
        n_classes: 类别数量
        n_timesteps: 时间步长
        class_sep: 类别分离度
        imbalance_ratio: 不平衡比例
        
    返回:
        (X, y) 其中X的形状为 [n_samples, n_timesteps, n_features]，y的形状为 [n_samples]
    """
    # 生成基础特征
    X_base, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=class_sep,
        random_state=42
    )
    
    # 创建时间序列数据
    X = np.zeros((n_samples, n_timesteps, n_features))
    
    for i in range(n_samples):
        for j in range(n_features):
            # 生成随机时间序列
            trend = np.linspace(0, 1, n_timesteps) * X_base[i, j]
            seasonality = 0.1 * np.sin(np.linspace(0, 10 * np.pi, n_timesteps))
            noise = 0.1 * np.random.randn(n_timesteps)
            
            X[i, :, j] = trend + seasonality + noise
    
    # 创建不平衡数据集
    if imbalance_ratio < 1.0:
        X, y = create_imbalanced_dataset(X, y, imbalance_ratio=imbalance_ratio)
    
    return X, y

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
    y_train = np.where(y_train > 0, 0, 1)
    y_test = np.where(y_test > 0, 0, 1)

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