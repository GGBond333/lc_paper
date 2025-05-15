import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score

class ContrastiveClusteringLoss(nn.Module):
    """
    实现对比聚类(COCL)损失函数，用于解决时间序列分类中的类别不平衡问题。
    """
    def __init__(self, temperature=0.5, alpha=1.0):
        """
        初始化COCL损失函数。
        
        参数:
            temperature: 控制相似度缩放的温度参数
            alpha: 平衡无监督和有监督组件的缩放因子
        """
        super(ContrastiveClusteringLoss, self).__init__()
        self.temperature = temperature  # 温度参数，控制相似度分数的缩放
        self.alpha = alpha  # 缩放因子，平衡无监督和有监督损失
        
    def forward(self, features, labels=None):
        """
        计算COCL损失。
        
        参数:
            features: 特征向量批次 [batch_size, feature_dim]
            labels: 可选的标签张量，用于有监督组件
            
        返回:
            总损失，结合相似性最大化、差异性最小化、聚类损失和可选的交叉熵损失
        """
        batch_size = features.size(0)
        
        # 归一化特征向量
        features_norm = F.normalize(features, dim=1)  # 对特征进行L2归一化，确保余弦相似度计算的准确性
        
        # 计算余弦相似度矩阵
        similarity_matrix = torch.matmul(features_norm, features_norm.T) / self.temperature
        
        # 创建正样本对的掩码（相同类别）
        mask = torch.zeros_like(similarity_matrix)
        if labels is not None:
            # 如果提供了标签，则创建一个掩码，其中相同类别的样本对为1，不同类别的样本对为0
            mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        
        # 计算相似度概率
        exp_sim = torch.exp(similarity_matrix)  # 对相似度矩阵进行指数化
        row_sum = exp_sim.sum(dim=1, keepdim=True)  # 计算每行的和
        prob_matrix = exp_sim / row_sum  # 归一化，得到概率矩阵
        
        # 计算平均类内和类间概率
        p_i = prob_matrix.mean(dim=1)  # 每个样本的平均类内概率
        p_j = prob_matrix.mean(dim=0)  # 每个样本的平均类间概率
        
        # 1. 相似性最大化损失 (ℒSM)
        # 这部分损失鼓励相同类别的样本在特征空间中更接近
        l_sm = -torch.mean(torch.log(p_i) + torch.log(p_j * mask + (1 - mask)))
        
        # 2. 差异性最小化损失 (ℒDM)
        # 这部分损失惩罚样本被过度自信地分配到某个聚类中
        l_dm = -torch.mean(torch.log(1 - p_i))
        
        # 3. 聚类损失 (ℒCL)
        # 计算欧几里得距离矩阵
        x_square = features_norm.pow(2).sum(dim=1, keepdim=True)  # 计算每个特征向量的平方和
        # 计算距离矩阵：||a-b||^2 = ||a||^2 + ||b||^2 - 2(a·b)
        distance_matrix = x_square + x_square.T - 2 * torch.matmul(features_norm, features_norm.T)
        
        # 为每个样本找到最近的聚类
        cluster_assignments = torch.argmin(distance_matrix, dim=1)
        
        # 统计每个聚类中的样本数
        unique_clusters = torch.unique(cluster_assignments)
        cluster_counts = torch.zeros(len(unique_clusters), device=features.device)
        
        for i, c in enumerate(unique_clusters):
            cluster_counts[i] = (cluster_assignments == c).sum().float()
        
        # 计算聚类概率
        cluster_probs = cluster_counts / batch_size
        
        # 计算聚类分布的熵
        # 这部分损失确保样本在各个聚类中的分布更加均衡，防止某些聚类被忽略
        l_cl = -torch.sum(cluster_probs * torch.log(cluster_probs + 1e-10))
        
        # 组合三个损失组件
        cocl_loss = l_sm + l_dm + l_cl
        
        # 如果提供了标签，则添加交叉熵损失
        ce_loss = 0
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            logits = similarity_matrix  # 为简单起见，使用相似度作为logits
            ce_loss = criterion(logits, labels)
        
        # 总损失
        total_loss = self.alpha * cocl_loss + ce_loss
        
        return total_loss, {'l_sm': l_sm.item(), 'l_dm': l_dm.item(), 'l_cl': l_cl.item(), 'ce': ce_loss}