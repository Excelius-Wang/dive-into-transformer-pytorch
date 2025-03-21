import torch.nn as nn
from src.models.attention import MultiHeadAttention, FeedForward

class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    
    def __init__(self, config):
        """初始化Transformer块
        
        Args:
            config: 配置对象，包含模型参数
        """
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.layer_norm1 = nn.LayerNorm(config.EMBEDDING_DIM)
        self.layer_norm2 = nn.LayerNorm(config.EMBEDDING_DIM)
        
    def forward(self, x):
        """前向传播计算Transformer块
        
        Args:
            x: 输入张量 [batch_size, seq_len, embedding_dim]
            
        Returns:
            Transformer块输出 [batch_size, seq_len, embedding_dim]
        """
        # 自注意力 + 残差连接
        x = x + self.attention(self.layer_norm1(x))
        # 前馈网络 + 残差连接
        x = x + self.feed_forward(self.layer_norm2(x))
        return x 