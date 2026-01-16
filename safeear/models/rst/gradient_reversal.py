# Copyright (c) 2023 Amphion.
# Modified for Residual-Stripping Tower

"""
梯度反转层 (Gradient Reversal Layer)
用于对抗训练，确保特定层不包含某种信息
"""

from torch.autograd import Function
import torch
from torch import nn


class GradientReversalFunction(Function):
    """
    梯度反转函数
    前向传播时直接传递输入，反向传播时将梯度乘以负的alpha
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -alpha * grad_output
        return grad_input, None


revgrad = GradientReversalFunction.apply


class GradientReversal(nn.Module):
    """
    梯度反转层模块
    
    在对抗训练中使用：
    - 前向传播时不改变特征
    - 反向传播时反转梯度方向
    
    这样可以训练网络使特定层"不包含"某种信息
    例如：确保残差不包含语义/说话人/韵律信息
    
    Args:
        alpha: 梯度反转系数，默认1.0
               alpha越大，对抗训练越强
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha.to(x.device))
    
    def set_alpha(self, alpha: float):
        """动态调整alpha值"""
        self.alpha = torch.tensor(alpha, requires_grad=False)


class ScheduledGradientReversal(nn.Module):
    """
    带调度的梯度反转层
    
    alpha随训练进度动态调整：
    alpha(p) = 2 / (1 + exp(-gamma * p)) - 1
    
    其中 p 是训练进度 (0到1)
    
    Args:
        gamma: 调度参数，控制alpha变化的速度
        max_alpha: 最大alpha值
    """
    def __init__(self, gamma: float = 10.0, max_alpha: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.max_alpha = max_alpha
        self.register_buffer('progress', torch.tensor(0.0))
        
    def forward(self, x):
        # 计算当前alpha
        alpha = 2.0 / (1.0 + torch.exp(-self.gamma * self.progress)) - 1.0
        alpha = alpha * self.max_alpha
        return revgrad(x, alpha.to(x.device))
    
    def update_progress(self, progress: float):
        """更新训练进度 (0到1之间)"""
        self.progress = torch.tensor(min(max(progress, 0.0), 1.0))
