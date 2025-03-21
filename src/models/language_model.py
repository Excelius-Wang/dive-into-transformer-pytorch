import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.blocks import TransformerBlock

class GPTLanguageModel(nn.Module):
    """GPT风格的自回归语言模型"""
    
    def __init__(self, vocab_size, config):
        """初始化语言模型
        
        Args:
            vocab_size: 词表大小
            config: 配置对象，包含模型参数
        """
        super().__init__()
        self.config = config
        
        # token嵌入
        self.token_embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM)
        # 位置嵌入
        self.position_embedding = nn.Embedding(config.BLOCK_SIZE, config.EMBEDDING_DIM)
        # Transformer块
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.NUM_LAYERS)]
        )
        # 最终层归一化
        self.final_layer_norm = nn.LayerNorm(config.EMBEDDING_DIM)
        # 语言模型头
        self.lm_head = nn.Linear(config.EMBEDDING_DIM, vocab_size, bias=False)
        
        # 设备配置
        self.device = config.DEVICE
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """权重初始化函数
        
        Args:
            module: 需要初始化权重的模块
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, targets=None):
        """前向传播函数
        
        Args:
            input_ids: 输入的token索引 [batch_size, seq_len]
            targets: 目标token索引 [batch_size, seq_len]，可选
            
        Returns:
            logits: 预测logits [batch_size, seq_len, vocab_size]
            loss: 如果提供了targets，则返回损失值，否则为None
        """
        batch_size, seq_len = input_ids.shape
        
        # 获取token嵌入
        token_embeddings = self.token_embedding(input_ids).to(self.device)
        # 获取位置嵌入
        position_indices = torch.arange(seq_len, device=self.device)
        position_embeddings = self.position_embedding(position_indices).to(self.device)
        
        # 组合嵌入
        x = token_embeddings + position_embeddings
        
        # 通过Transformer块
        x = self.transformer_blocks(x)
        # 最终层归一化
        x = self.final_layer_norm(x)
        # 预测下一个token
        logits = self.lm_head(x)
        
        # 计算损失
        if targets is None:
            loss = None
        else:
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.view(batch_size * seq_len, vocab_size)
            targets = targets.view(batch_size * seq_len)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    def generate(self, input_ids, max_new_tokens):
        """生成文本
        
        Args:
            input_ids: 输入的token索引 [batch_size, seq_len]
            max_new_tokens: 要生成的最大新token数
            
        Returns:
            生成的完整token序列 [batch_size, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # 截断context以处理最大长度
            input_context = input_ids[:, -self.config.BLOCK_SIZE:]
            # 预测
            logits, _ = self.forward(input_context)
            # 获取最后一个时间步的预测
            logits = logits[:, -1, :]
            # 应用softmax获取概率
            probs = F.softmax(logits, dim=-1)
            # 采样下一个token
            next_token = torch.multinomial(probs, num_samples=1)
            # 加入序列
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
        return input_ids 