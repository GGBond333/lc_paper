import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MeanRecallLoss(nn.Module):
    """
    均值召回率损失函数
    
    参数:
        y_pred: 预测概率，形状为 [batch_size, num_classes]
        y_true: 真实标签，形状为 [batch_size, num_classes]（one-hot编码）
        
    返回:
        均值召回率的负值（用于最大化召回率）
    """
    def __init__(self):
        super(MeanRecallLoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        # 获取预测的类别和真实的类别
        y_pred_classes = torch.argmax(y_pred, dim=1)
        y_true_classes = torch.argmax(y_true, dim=1)
        
        # 获取类别数量
        num_classes = y_true.size(1)
        
        # 计算每个类别的召回率
        recalls = []
        for i in range(num_classes):
            # 找出属于当前类别的样本
            true_class_mask = (y_true_classes == i)
            
            # 如果没有该类别的样本，则跳过
            if torch.sum(true_class_mask) == 0:
                continue
                
            # 计算这些样本中被正确分类的比例
            class_recall = torch.mean((y_pred_classes[true_class_mask] == i).float())
            recalls.append(class_recall)
        
        # 计算所有召回率的平均值
        if len(recalls) > 0:
            mean_recall = torch.stack(recalls).mean()
        else:
            mean_recall = torch.tensor(0.0, device=y_pred.device)
        
        # 返回负值，因为我们要最大化召回率（而PyTorch默认最小化损失）
        return -mean_recall

class ProductRecallLoss(nn.Module):
    """
    乘积召回率损失函数
    
    参数:
        y_pred: 预测概率，形状为 [batch_size, num_classes]
        y_true: 真实标签，形状为 [batch_size, num_classes]（one-hot编码）
        
    返回:
        所有类别召回率乘积的负值
    """
    def __init__(self):
        super(ProductRecallLoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        # 获取预测的类别和真实的类别
        y_pred_classes = torch.argmax(y_pred, dim=1)
        y_true_classes = torch.argmax(y_true, dim=1)
        
        # 获取类别数量
        num_classes = y_true.size(1)
        
        # 计算每个类别的召回率
        recalls = []
        for i in range(num_classes):
            # 找出属于当前类别的样本
            true_class_mask = (y_true_classes == i)
            
            # 如果没有该类别的样本，则跳过
            if torch.sum(true_class_mask) == 0:
                continue
                
            # 计算这些样本中被正确分类的比例
            class_recall = torch.mean((y_pred_classes[true_class_mask] == i).float())
            # 添加一个小的epsilon值避免乘积为0
            recalls.append(class_recall + 1e-7)
        
        # 计算所有召回率的乘积
        if len(recalls) > 0:
            product_recall = torch.prod(torch.stack(recalls))
        else:
            product_recall = torch.tensor(0.0, device=y_pred.device)
        
        # 返回负值用于最小化损失
        return -product_recall

class SoftmaxRecallLoss(nn.Module):
    """
    Softmax近似的最小召回率损失函数
    
    参数:
        y_pred: 预测概率，形状为 [batch_size, num_classes]
        y_true: 真实标签，形状为 [batch_size, num_classes]（one-hot编码）
        alpha: 控制近似程度的参数，越大越接近最小值函数
        
    返回:
        Softmax近似的最小召回率的负值
    """
    def __init__(self, alpha=10.0):
        super(SoftmaxRecallLoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, y_pred, y_true):
        # 获取预测的类别和真实的类别
        y_pred_classes = torch.argmax(y_pred, dim=1)
        y_true_classes = torch.argmax(y_true, dim=1)
        
        # 获取类别数量
        num_classes = y_true.size(1)
        
        # 计算每个类别的召回率
        recalls = []
        for i in range(num_classes):
            # 找出属于当前类别的样本
            true_class_mask = (y_true_classes == i)
            
            # 如果没有该类别的样本，则跳过
            if torch.sum(true_class_mask) == 0:
                continue
                
            # 计算这些样本中被正确分类的比例
            class_recall = torch.mean((y_pred_classes[true_class_mask] == i).float())
            recalls.append(class_recall)
        
        # 如果没有有效的召回率，返回0
        if len(recalls) == 0:
            return torch.tensor(0.0, device=y_pred.device)
            
        # 将召回率转换为张量
        recalls_tensor = torch.stack(recalls)
        
        # 使用softmax近似计算最小召回率
        # 论文中的公式: Softmax(α) = Σ(recall_Ci * exp(-α * recall_Ci)) / Σ(exp(-α * recall_Ci))
        numerator = torch.sum(recalls_tensor * torch.exp(-self.alpha * recalls_tensor))
        denominator = torch.sum(torch.exp(-self.alpha * recalls_tensor))
        softmax_recall = numerator / denominator
        
        # 返回负值用于最小化损失
        return -softmax_recall

class LSERecallLoss(nn.Module):
    """
    LogSumExp (LSE) 召回率损失函数
    论文中表现最好的损失函数
    
    参数:
        y_pred: 预测概率，形状为 [batch_size, num_classes]
        y_true: 真实标签，形状为 [batch_size, num_classes]（one-hot编码）
        alpha: 控制近似程度的参数，越大越接近最小值函数
        
    返回:
        LSE近似的最小召回率的负值
    """
    def __init__(self, alpha=10.0):
        super(LSERecallLoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, y_pred, y_true):
        # 获取类别数量
        num_classes = y_true.size(1)
        
        # 计算每个类别的召回率
        recalls = []
        for i in range(num_classes):
            # 获取当前类别的真实标签
            true_class = y_true[:, i]
            
            # 如果没有该类别的样本，则跳过
            if torch.sum(true_class) == 0:
                continue
            
            # 计算当前类别的预测概率
            pred_probs = y_pred[:, i]
            
            # 计算召回率：正确预测的概率之和除以该类的样本数
            class_recall = torch.sum(pred_probs * true_class) / (torch.sum(true_class) + 1e-7)
            recalls.append(class_recall)
        
        # 如果没有有效的召回率，返回0
        if len(recalls) == 0:
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
            
        # 将召回率转换为张量
        recalls_tensor = torch.stack(recalls)
        
        # 使用LSE近似计算最小召回率
        # 论文中的公式: LSE(α) = (1/-α) * log(Σ(exp(-α * recall_Ci)))
        lse_recall = (1.0 / -self.alpha) * torch.log(torch.sum(torch.exp(-self.alpha * recalls_tensor)))
        
        # 返回负值用于最小化损失
        return -lse_recall

class PNormRecallLoss(nn.Module):
    """
    P-norm召回率损失函数
    
    参数:
        y_pred: 预测概率，形状为 [batch_size, num_classes]
        y_true: 真实标签，形状为 [batch_size, num_classes]（one-hot编码）
        alpha: 控制近似程度的参数，越大越接近最小值函数
        
    返回:
        P-norm近似的最小召回率的负值
    """
    def __init__(self, alpha=10.0):
        super(PNormRecallLoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, y_pred, y_true):
        # 获取预测的类别和真实的类别
        y_pred_classes = torch.argmax(y_pred, dim=1)
        y_true_classes = torch.argmax(y_true, dim=1)
        
        # 获取类别数量
        num_classes = y_true.size(1)
        
        # 计算每个类别的召回率
        recalls = []
        for i in range(num_classes):
            # 找出属于当前类别的样本
            true_class_mask = (y_true_classes == i)
            
            # 如果没有该类别的样本，则跳过
            if torch.sum(true_class_mask) == 0:
                continue
                
            # 计算这些样本中被正确分类的比例
            class_recall = torch.mean((y_pred_classes[true_class_mask] == i).float())
            recalls.append(class_recall)
        
        # 如果没有有效的召回率，返回0
        if len(recalls) == 0:
            return torch.tensor(0.0, device=y_pred.device)
            
        # 将召回率转换为张量
        recalls_tensor = torch.stack(recalls)
        
        # 使用p-norm近似计算最小召回率
        # 论文中的公式: P-norm(α) = (Σ(recall_Ci^(-α)))^(-1/α)
        p_norm_recall = torch.pow(torch.sum(torch.pow(recalls_tensor, -self.alpha)), -1.0/self.alpha)
        
        # 返回负值用于最小化损失
        return -p_norm_recall