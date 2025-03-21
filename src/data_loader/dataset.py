import torch
import jieba
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from src.config.config import TransformerConfig
from src.utils.logger import printlog

class TextDataset:
    """文本数据集处理类，负责加载和处理文本数据"""
    
    def __init__(self, config):
        """初始化数据集
        
        Args:
            config: 配置对象，包含数据路径和处理参数
        """
        self.config = config
        self.file_path = Path(Path.cwd().parent.parent, config.DATA_PATH)
        self.block_size = config.BLOCK_SIZE
        self.device = config.DEVICE
        
        # 读取文本
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.text = f.read()
        
        # 分词处理
        self.words = list(jieba.cut(self.text))
        printlog(f"分词后的词语数量: {len(self.words)}")
        
        # 构建词表
        self.vocab = sorted(list(set(self.words)))
        self.vocab_size = len(self.vocab)
        printlog(f"词表大小: {self.vocab_size}")
        
        # 构建映射
        self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}
        self.idx_to_word = {i: word for i, word in enumerate(self.vocab)}
        
        # 编码文本并分割训练集和验证集
        self.data = torch.tensor(self.encode_text(self.text), dtype=torch.long)
        split_factor = 0.8
        n = int(split_factor * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]
        printlog(f"文件 {self.file_path} 读取完成...")
    
    def encode_text(self, text):
        """将文本编码为索引序列
        
        Args:
            text: 需要编码的原始文本
            
        Returns:
            包含索引的列表，每个索引对应词表中的一个词
        """
        words = list(jieba.cut(text))
        return [self.word_to_idx[word] for word in words]
    
    def decode_text(self, indices):
        """将索引序列解码为文本
        
        Args:
            indices: 索引序列
            
        Returns:
            解码后的文本
        """
        return ''.join([self.idx_to_word[idx] for idx in indices])
    
    def get_batch(self, split):
        """获取批次数据
        
        Args:
            split: 'train'或'val'，指定获取训练集还是验证集的数据
            
        Returns:
            输入张量和目标张量的元组
        """
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.config.BATCH_SIZE,))
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])
        return x.to(self.device), y.to(self.device)
    
    def get_random_context(self, max_new_tokens):
        """从验证集获取随机上下文
        
        Args:
            max_new_tokens: 实际下文的最大长度
            
        Returns:
            上下文张量和真实下文张量的元组
        """
        start_idx = random.randint(0, len(self.val_data) - self.block_size - max_new_tokens)
        
        # 上文
        context = torch.zeros((1, self.block_size), dtype=torch.long, device=self.device)
        context[0, :] = self.val_data[start_idx:start_idx + self.block_size]
        
        # 真实下文
        real_next = torch.zeros((1, max_new_tokens), dtype=torch.long, device=self.device)
        real_next[0, :] = self.val_data[start_idx + self.block_size:start_idx + self.block_size + max_new_tokens]
        
        return context, real_next 