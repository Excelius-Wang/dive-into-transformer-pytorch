import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config.config import TransformerConfig

class AttentionHead(nn.Module):
    """单头自注意力机制"""
    
    def __init__(self, head_size, config):
        """初始化注意力头
        
        Args:
            head_size: 注意力头的维度大小
            config: 配置对象，包含模型参数
        """
        super().__init__()
        self.key = nn.Linear(config.EMBEDDING_DIM, head_size, bias=False)
        self.query = nn.Linear(config.EMBEDDING_DIM, head_size, bias=False)
        self.value = nn.Linear(config.EMBEDDING_DIM, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(config.BLOCK_SIZE, config.BLOCK_SIZE)))
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        self.device = config.DEVICE
        
    def forward(self, x):
        """前向传播计算注意力
        
        Args:
            x: 输入张量 [batch_size, seq_len, embedding_dim]
            
        Returns:
            注意力输出 [batch_size, seq_len, head_size]
        """
        batch_size, seq_len, embedding_dim = x.shape
        
        # 计算key, query, value
        key = self.key(x)  # [batch_size, seq_len, head_size]
        query = self.query(x)  # [batch_size, seq_len, head_size]
        
        # 计算注意力权重
        attention_scores = (query @ key.transpose(-2, -1) * key.shape[-1] ** -0.5).to(self.device)
        # 应用因果掩码
        attention_scores = attention_scores.masked_fill(self.tril == 0, float("-inf")).to(self.device)
        # softmax归一化
        attention_probs = F.softmax(attention_scores, dim=-1)
        # dropout正则化
        attention_probs = self.dropout(attention_probs)
        
        # 计算输出
        value = self.value(x).to(self.device)
        output = attention_probs @ value
        return output


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, config):
        """初始化多头注意力
        
        Args:
            config: 配置对象，包含模型参数
        """
        super().__init__()
        assert config.EMBEDDING_DIM % config.NUM_HEADS == 0, "嵌入维度必须能被注意力头数整除"
        
        # 创建多个注意力头
        self.heads = nn.ModuleList([
            AttentionHead(config.HEAD_SIZE, config) for _ in range(config.NUM_HEADS)
        ])
        self.projection = nn.Linear(config.HEAD_SIZE * config.NUM_HEADS, config.EMBEDDING_DIM)
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        
    def forward(self, x):
        """前向传播计算多头注意力
        
        Args:
            x: 输入张量 [batch_size, seq_len, embedding_dim]
            
        Returns:
            多头注意力输出 [batch_size, seq_len, embedding_dim]
        """
        # 将所有头的输出拼接起来
        attention_outputs = torch.cat([head(x) for head in self.heads], dim=-1)
        # 投影回原始维度
        output = self.dropout(self.projection(attention_outputs))
        return output


class FeedForward(nn.Module):
    """前馈神经网络"""
    
    def __init__(self, config):
        """初始化前馈网络
        
        Args:
            config: 配置对象，包含模型参数
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.EMBEDDING_DIM, config.EMBEDDING_DIM * 4),
            nn.ReLU(),
            nn.Linear(config.EMBEDDING_DIM * 4, config.EMBEDDING_DIM),
            nn.Dropout(config.DROPOUT_RATE),
        )
        
    def forward(self, x):
        """前向传播计算前馈网络
        
        Args:
            x: 输入张量 [batch_size, seq_len, embedding_dim]
            
        Returns:
            前馈网络输出 [batch_size, seq_len, embedding_dim]
        """
        return self.net(x) 