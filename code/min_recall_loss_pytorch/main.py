import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 导入自定义模块
from models import MLSTM_FCN, ResNet
from min_recall_losses import MeanRecallLoss, ProductRecallLoss, SoftmaxRecallLoss, LSERecallLoss, PNormRecallLoss
from utils.data_utils import generate_synthetic_time_series, load_and_preprocess_data, calculate_imbalance_degree, load_wafer_dataset
from utils.evaluation import train_model, evaluate_model, plot_training_history

# 设置随机种子以确保结果可重现
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def run_experiment(model_type='mlstm_fcn', loss_type='lse', alpha=10.0, epochs=50, batch_size=32):
    """
    运行实验

    参数:
        model_type: 模型类型 ('mlstm_fcn' 或 'resnet')
        loss_type: 损失函数类型 ('mean', 'product', 'softmax', 'lse', 'p_norm', 'cross_entropy')
        alpha: 损失函数的alpha参数
        epochs: 训练轮数
        batch_size: 批量大小

    返回:
        (accuracy, min_recall) 准确率和最小召回率
    """
    print(f"运行实验: 模型={model_type}, 损失函数={loss_type}, alpha={alpha}")

    # 加载Wafer数据集
    X_train, y_train, X_test, y_test = load_wafer_dataset(
        'Wafer_TRAIN.tsv',
        'Wafer_TEST.tsv'
    )

    # 计算不平衡度
    id_value = calculate_imbalance_degree(y_train)
    print(f"数据集不平衡度 (ID): {id_value:.4f}")

    # 预处理数据
    train_loader, test_loader, X_train, X_test, y_train, y_test = load_and_preprocess_data(
        X_train, y_train, X_test=X_test, y_test=y_test, batch_size=batch_size
    )

    # 打印数据集信息
    print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
    print(f"类别分布: {np.bincount(np.argmax(y_train, axis=1))}")

    # 创建模型
    input_dim = X_train.shape[2]  # 特征维度
    num_classes = y_train.shape[1]  # 类别数

    if model_type == 'mlstm_fcn':
        model = MLSTM_FCN(input_dim, num_classes)
    else:
        model = ResNet(input_dim, num_classes)

    model = model.to(device)
    criterion = LSERecallLoss(alpha=alpha)

    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    history = train_model(
        model, train_loader, criterion, optimizer, device,
        num_epochs=epochs, patience=10
    )

    # 评估模型
    accuracy, min_recall, recalls, auc, f1, gmean, precision = evaluate_model(model, test_loader, device)
    print(f"测试集 AUC: {auc:.4f}")

    print(f"测试集准确率: {accuracy:.4f}")
    print(f"测试集最小召回率: {min_recall:.4f}")
    print(f"各类别召回率: {recalls}")
    print(f"F1: {f1:.4f}")
    print(f"G-mean: {gmean:.4f}")
    print(f"precision: {precision:.4f}")

    # 绘制训练历史
    plot_training_history(
        history,
        title=f'{model_type}_{loss_type}_alpha{alpha}',
        save_path=f'{model_type}_{loss_type}_alpha{alpha}_history.png'
    )

    return accuracy, min_recall

# def compare_loss_functions():
#     """
#     比较不同损失函数的性能
#     """
#     loss_types = ['cross_entropy', 'mean', 'product', 'softmax', 'lse', 'p_norm']
#     results = []
#
#     for loss_type in loss_types:
#         accuracy, min_recall = run_experiment(model_type='mlstm_fcn', loss_type=loss_type)
#         results.append((loss_type, accuracy, min_recall))
#
#     # 打印结果
#     print("\n比较不同损失函数的性能:")
#     print("损失函数\t准确率\t最小召回率")
#     for loss_type, accuracy, min_recall in results:
#         print(f"{loss_type}\t{accuracy:.4f}\t{min_recall:.4f}")
#
#     # 绘制结果
#     plt.figure(figsize=(10, 6))
#
#     loss_names = [r[0] for r in results]
#     accuracies = [r[1] for r in results]
#     min_recalls = [r[2] for r in results]
#
#     x = np.arange(len(loss_names))
#     width = 0.35
#
#     plt.bar(x - width/2, accuracies, width, label='准确率')
#     plt.bar(x + width/2, min_recalls, width, label='最小召回率')
#
#     plt.xlabel('损失函数')
#     plt.ylabel('性能')
#     plt.title('不同损失函数的性能比较')
#     plt.xticks(x, loss_names, rotation=45)
#     plt.legend()
#
#     plt.tight_layout()
#     plt.savefig('loss_functions_comparison.png')
#     plt.show()
#
# def analyze_alpha_sensitivity():
#     """
#     分析alpha参数的敏感性
#     """
#     alpha_values = [1, 5, 10, 20, 40, 60, 80, 100, 120]
#     results = []
#
#     for alpha in alpha_values:
#         accuracy, min_recall = run_experiment(model_type='mlstm_fcn', loss_type='lse', alpha=alpha)
#         results.append((alpha, accuracy, min_recall))
#
#     # 打印结果
#     print("\n分析alpha参数的敏感性:")
#     print("alpha\t准确率\t最小召回率")
#     for alpha, accuracy, min_recall in results:
#         print(f"{alpha}\t{accuracy:.4f}\t{min_recall:.4f}")
#
#     # 绘制结果
#     plt.figure(figsize=(10, 6))
#
#     alphas = [r[0] for r in results]
#     accuracies = [r[1] for r in results]
#     min_recalls = [r[2] for r in results]
#
#     plt.plot(alphas, accuracies, 'o-', label='准确率')
#     plt.plot(alphas, min_recalls, 's-', label='最小召回率')
#
#     plt.xlabel('alpha值')
#     plt.ylabel('性能')
#     plt.title('alpha参数敏感性分析')
#     plt.legend()
#     plt.grid(True)
#
#     plt.tight_layout()
#     plt.savefig('alpha_sensitivity.png')
#     plt.show()

def main():
    """
    主函数
    """
    print("基于最小召回率的不平衡时间序列分类损失函数实现 (PyTorch版)")
    print("=" * 60)

    # 创建结果目录
    os.makedirs('results', exist_ok=True)

    # 运行单个实验
    print("\n运行单个实验 (MLSTM-FCN + LSE损失函数):")
    run_experiment(model_type='mlstm_fcn', loss_type='lse', alpha=10.0)
    
    # 比较不同损失函数
    # print("\n比较不同损失函数:")
    # compare_loss_functions()
    #
    # # 分析alpha参数的敏感性
    # print("\n分析alpha参数的敏感性:")
    # analyze_alpha_sensitivity()

if __name__ == "__main__":
    main()