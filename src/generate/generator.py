import torch
import textwrap
import re
from src.utils.logger import printlog

class TextGenerator:
    """文本生成器，用于从训练好的模型生成新文本"""
    
    def __init__(self, model, dataset, config):
        """初始化生成器
        
        Args:
            model: 训练好的语言模型
            dataset: 用于获取编码和解码的数据集
            config: 配置对象，包含生成参数
        """
        self.model = model
        self.model.eval()  # 设置为评估模式
        self.dataset = dataset
        self.config = config
        self.device = config.DEVICE
    
    def generate_random_completion(self, max_new_tokens=None):
        """从随机初始上下文生成文本
        
        Args:
            max_new_tokens: 要生成的最大新token数，默认使用配置中的值
            
        Returns:
            生成的文本
        """
        if max_new_tokens is None:
            max_new_tokens = self.config.MAX_NEW_TOKENS
        
        # 从训练集随机选择一段文本作为上下文
        idx = torch.randint(0, len(self.dataset.train_data) - self.config.BLOCK_SIZE, (1,))
        context = self.dataset.train_data[idx:idx + self.config.BLOCK_SIZE].to(self.device)
        
        # 生成文本
        return self.generate_from_context(context, max_new_tokens)
        
    def generate_from_context(self, context, max_new_tokens=None):
        """从给定上下文生成文本
        
        Args:
            context: 上下文输入，形状为(batch_size, seq_len)
            max_new_tokens: 要生成的最大新token数，默认使用配置中的值
            
        Returns:
            生成的文本
        """
        if max_new_tokens is None:
            max_new_tokens = self.config.MAX_NEW_TOKENS
            
        # 保存原始上下文长度，以便只返回新生成的部分
        context_length = context.shape[1]
            
        # 模型设置为评估模式
        self.model.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 获取预测
                logits, _ = self.model(context)
                
                # 仅关注最后一个时间步
                logits = logits[:, -1, :]  # (batch_size, vocab_size)
                
                # 应用softmax获取概率
                probs = torch.softmax(logits, dim=-1)
                
                # 采样下一个token
                next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
                
                # 添加到上下文
                context = torch.cat([context, next_token], dim=1)  # (batch_size, seq_len+1)
        
        # 仅提取新生成的部分（不包含原始上下文）
        new_tokens = context[:, context_length:]
        
        # 解码为文本
        return self.decode_tokens(new_tokens)
    
    def generate_from_text(self, text, max_new_tokens=None):
        """从文本生成续写
        
        Args:
            text: 上下文文本
            max_new_tokens: 要生成的最大新token数，默认使用配置中的值
            
        Returns:
            生成的文本
        """
        if max_new_tokens is None:
            max_new_tokens = self.config.MAX_NEW_TOKENS
        
        # 编码输入文本
        context = self.encode_text(text)
        
        # 生成文本
        generated_text = self.generate_from_context(context, max_new_tokens)
        
        # 返回原始文本 + 生成的文本
        return text + generated_text
    
    def encode_text(self, text):
        """将文本编码为token ids
        
        Args:
            text: 要编码的文本
            
        Returns:
            编码后的tensor，形状为(1, seq_len)
        """
        # 使用数据集的分词和编码方法
        tokens = [self.dataset.stoi.get(ch, self.dataset.stoi.get('<UNK>')) for ch in self.dataset.encode_fn(text)]
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        return tokens
    
    def decode_tokens(self, tokens):
        """将token ids解码为文本
        
        Args:
            tokens: token id tensor，形状为(batch_size, seq_len)
            
        Returns:
            解码后的文本
        """
        tokens = tokens[0].tolist()  # 取第一个batch并转为列表
        text = ''.join([self.dataset.itos.get(i, '<UNK>') for i in tokens])
        return text
        
    def print_generation_result(self, text, context_text=None):
        """打印生成结果
        
        Args:
            text: 要打印的文本
            context_text: 上下文文本，若提供则以不同方式显示
        """
        wrap_width = self.config.WRAP_WIDTH
        
        if context_text:
            # 以不同方式显示上下文和生成文本
            printlog("\n=== 上下文 ===")
            # 对上下文文本进行包装显示
            for i in range(0, len(context_text), wrap_width):
                printlog(context_text[i:i+wrap_width])
                
            printlog("\n=== 生成文本 ===")
            # 对生成文本进行包装显示
            for i in range(0, len(text), wrap_width):
                printlog(text[i:i+wrap_width])
        else:
            # 直接显示生成文本
            printlog("\n=== 生成文本 ===")
            for i in range(0, len(text), wrap_width):
                printlog(text[i:i+wrap_width]) 