"""GPT模型架构定义

包含所有模型组件的定义，包括注意力机制、前馈网络和完整的GPT模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import Config

class Head(nn.Module):
    """单个注意力头 - 改进版本"""

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(Config.n_embd, head_size, bias=False)
        self.query = nn.Linear(Config.n_embd, head_size, bias=False)
        self.value = nn.Linear(Config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(Config.block_size, Config.block_size)))
        self.dropout = nn.Dropout(Config.dropout)
        
        # 改进的权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化"""
        for module in [self.key, self.query, self.value]:
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # 输入形状: (batch, time-step, channels)
        # 输出形状: (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        
        # 计算注意力权重，使用缩放点积注意力
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # 执行值向量的加权聚合
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """多头注意力机制 - 改进版本"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        assert Config.n_embd == num_heads * head_size, "嵌入维度必须等于头数乘以头大小"
        
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(Config.n_embd, Config.n_embd, bias=False)
        self.dropout = nn.Dropout(Config.dropout)
        
        # 改进的权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化"""
        torch.nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # 并行计算所有注意力头
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # 投影到原始维度
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedFoward(nn.Module):
    """前馈神经网络 - 改进版本"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),  # 使用GELU激活函数，性能更好
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(Config.dropout),
        )
        
        # 改进的权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化"""
        for module in self.net:
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer块 - 改进版本（Pre-LayerNorm）"""

    def __init__(self, n_embd, n_head):
        # n_embd: 嵌入维度, n_head: 注意力头数
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
        # 改进的权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化"""
        # LayerNorm参数保持默认初始化
        pass

    def forward(self, x):
        # Pre-LayerNorm架构，训练更稳定
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    """GPT语言模型 - 改进版本"""

    def __init__(self, vocab_size):
        super().__init__()
        # 词嵌入表
        self.token_embedding_table = nn.Embedding(vocab_size, Config.n_embd)
        # 位置嵌入表
        self.position_embedding_table = nn.Embedding(Config.block_size, Config.n_embd)
        # Transformer块序列
        self.blocks = nn.Sequential(*[Block(Config.n_embd, n_head=Config.n_head) for _ in range(Config.n_layer)])
        # 最终层归一化
        self.ln_f = nn.LayerNorm(Config.n_embd)
        # 语言模型头（权重共享）
        self.lm_head = nn.Linear(Config.n_embd, vocab_size, bias=False)
        
        # 权重初始化
        self.apply(self._init_weights)
        
        # 特殊初始化：缩放残差投影
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * Config.n_layer))

    def _init_weights(self, module):
        """改进的权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # 词嵌入和位置嵌入
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        
        # 通过Transformer块处理
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        
        # 计算logits
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            # 计算交叉熵损失
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx是当前上下文中的索引数组，形状为(B, T)
        for _ in range(max_new_tokens):
            # 截取token以适应block_size
            idx_cond = idx[:, -Config.block_size:]
            # 获取预测
            logits, loss = self(idx_cond)
            # 关注最后一个时间步
            logits = logits[:, -1, :] # 变成 (B, C)
            # 应用softmax获取概率
            probs = F.softmax(logits, dim=-1) # (B, C)
            # 从分布中采样
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # 将采样的索引拼接到运行序列中
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx