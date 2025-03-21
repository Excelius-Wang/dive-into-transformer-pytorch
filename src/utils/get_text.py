import random
import textwrap
from pathlib import Path
import jieba
import torch
import os
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.module import T

from src.utils.logger import printlog

# 移除固定GPU设置，改为自动检测可用GPU
torch.manual_seed(42)

# 超参数
batch_size = 64  # 同时平行处理多少条独立数据（batch）
block_size = 256  # 训练、验证的字符串长度
# 自动检测并设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu_count = torch.cuda.device_count()
printlog(f"当前使用设备: {device}")
printlog(f"可用GPU数量: {gpu_count}")
if gpu_count > 0 and device.type == 'cuda':
    printlog(f"GPU型号: {torch.cuda.get_device_name(0)}")
size = 16  # 几个值需要做嵌入
n_embedding = 512  # 嵌入的维度，尽量为 2 的整数次幂
n_heads = 8
n_layers = 12
head_size = n_embedding // n_heads
wrap_width = 50
learning_rate = 3e-4
max_iters = 3000
eval_interval = int(max_iters / 10)
eval_iters = 200
dropout_value = 0.2

# file_name = Path(Path.cwd().parent.parent, 'data/raw', 'HLM_Short.txt')
file_name = Path(Path.cwd().parent.parent, 'data/raw', 'Hong_Lou_Meng.txt')
printlog(Path.cwd())
printlog(str(file_name))

# 数据预处理
with open(file_name, "r", encoding="utf-8") as f:
    text = f.read()  # str

# 使用jieba进行分词
words = list(jieba.cut(text))
printlog(f"分词后的词语数量: {len(words)}")

# 生成词表
vocab = sorted(list(set(words)))
vocab_size = len(vocab)
printlog(f"词表大小: {vocab_size}")

# 获取词语与数字的投影
word_to_idx = {word: i for i, word in enumerate(vocab)}  # 词语到整数
idx_to_word = {i: word for i, word in enumerate(vocab)}  # 整数到词语


def encode_text(_text):
    # 对文本进行分词，然后将每个词转换为对应的索引
    _words = list(jieba.cut(_text))
    return [word_to_idx[word] for word in _words]


def decode_text(index_list):
    # 将索引列表转换回词语，然后连接成文本
    return ''.join([idx_to_word[i] for i in index_list])


# 训练、验证分组
data = torch.tensor(encode_text(text), dtype=torch.long)  # 整数表示词语
split_factor = 0.8
n = int(split_factor * len(data))  # 前 80%（比例系数）作为训练集
train_data = data[:n]
val_data = data[n:]
printlog(f"文件 {file_name} 读取完成...")


# 模型的预测：输入当前词的时候，要预测下一个词
def get_batch(split):
    _data = train_data if split == 'train' else val_data
    ix = torch.randint(len(_data) - block_size, (batch_size,))
    x = torch.stack([_data[i:i + block_size] for i in ix])  # 输入值 batch
    y = torch.stack([_data[i + 1:i + block_size + 1] for i in ix])  # 标签 batch，把 x 后移一个词，想要的输出值（target）
    x, y = x.to(device), y.to(device)
    return x, y


# ----- 损失评测 -----
@torch.no_grad()  # 不需要计算梯度，作用域为整个函数
def estimate_loss(model):
    out = {}
    model.eval()  # 模型进入评估模式
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)  # 建立一个初始值为 0 的容器，用于储存 loss 值
        for k in range(eval_iters):
            X, Y = get_batch(split)  # split 是一个字符串，用来控制 get_batch 函数的行为
            logits, loss = model(X, Y)  # model 的输入值一个是 index（以每个字符的序号表示的序列），一个是 target
            losses[k] = loss.item()
        out[split] = losses.mean()  # out 是含有两个元素的字典，一个是 train，一个是 val，每个元素对应一个 loss 的平均值
    model.train()  # 再转化为训练模式（如果之前没有转为 eval 模式，则不需要）
    return out


# Head 类
class Head(nn.Module):
    def __init__(self, head_size, _dropout=dropout_value):
        super().__init__()
        self.key = nn.Linear(n_embedding, head_size, bias=False)  # 线性变换层
        self.query = nn.Linear(n_embedding, head_size, bias=False)  # 线性变换层
        self.value = nn.Linear(n_embedding, head_size, bias=False)  # 线性变换层
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))  # 不可训练的结构，约为常量，下三角矩阵
        self.dropout = nn.Dropout(_dropout)

    def forward(self, x):
        B, T, C = x.shape  # B = batch_size, T = block_size, C = n_embedding

        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        wei = (q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5).to(device)  # T 和 head_size 需要转置一下，注意力矩阵
        wei = wei.masked_fill(self.tril == 0, float("-inf")).to(device)  # 掩码
        wei = F.softmax(wei, dim=-1)  # softmax
        wei = self.dropout(wei)  # 随即去掉（归零）一些值，增加网络的稳定性

        v = self.value(x).to(device)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, _n_heads, _head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(_head_size) for _ in range(_n_heads)])
        self.proj = nn.Linear(_n_heads * _head_size, n_embedding)
        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # 拼接
        out = self.dropout(self.proj(out))

        return out


class FeedForward(nn.Module):
    def __init__(self, _n_embedding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_n_embedding, _n_embedding * 4),
            nn.ReLU(),
            nn.Linear(_n_embedding * 4, _n_embedding),
            nn.Dropout(dropout_value),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, _n_embedding, _n_heads):
        super().__init__()
        self.sa = MultiHeadAttention(_n_heads, head_size)  # 多头自注意力
        self.ffwd = FeedForward(_n_embedding)
        self.ln1 = nn.LayerNorm(_n_embedding)
        self.ln2 = nn.LayerNorm(_n_embedding)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # 残差多头注意力网络
        x = x + self.ffwd(self.ln2(x))  # 残差前馈网络
        return x


# 语言模型
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embedding)
        self.position_embedding_table = nn.Embedding(block_size, n_embedding)
        self.blocks = nn.Sequential(*[Block(n_embedding, n_heads) for _ in range(n_layers)])  # 多残差多头注意力
        self.ln_f = nn.LayerNorm(n_embedding)  # final LayerNorm
        self.lm_head = nn.Linear(n_embedding, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape  # (B, T) B = batch_size, T = block_size, 数据为 token（整数）形式
        token_embd = self.token_embedding_table(idx).to(device)
        position_idx = torch.arange(T).to(device)
        position_embd = self.position_embedding_table(position_idx).to(device)

        x = token_embd + position_embd  # (B, T, n_embedding)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # B T 连接起来
            logits = logits.view(B * T, C)  # 摊平
            targets = targets.view(B * T)
            # 连成一串，去算交叉熵损失
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, token_sequ, max_new_tokens):  # token_sequ 已知的上文，max_new_tokens 是续写的长度
        for _ in range(max_new_tokens):
            token_input = token_sequ[:, -block_size:]  # 取最后 block_size 个 token
            logits, loss = self.forward(token_input)  # logits, (B, T, vocab_size)
            logits = logits[:, -1, :]  # 只取最后一个 token 的输出
            probs = F.softmax(logits, dim=-1)
            token_next = torch.multinomial(probs, num_samples=1)  # 概率分布向量 --> one-hot 向量 --> 整数 token
            token_sequ = torch.cat([token_sequ, token_next], dim=1)
        new_tokens = token_sequ[:, -max_new_tokens:]
        return new_tokens


# ----- 主函数 -----
def main():
    printlog(f"训练内容：{file_name}")
    model = LanguageModel().to(device)  # 实例化
    
    # 根据GPU数量自动适配多GPU训练
    if torch.cuda.device_count() > 1:
        printlog(f"使用 {torch.cuda.device_count()} 个GPU并行训练")
        model = nn.DataParallel(model)
    else:
        printlog("使用单GPU训练")
    
    printlog(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")  # 打印参数数量

    # 设定一个优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 训练循环
    for i in range(max_iters):
        if i % eval_interval == 0 or i == max_iters - 1:
            losses = estimate_loss(model)
            print(f"步骤 {i}: | 训练损失: {losses['train']:.4f} | 验证损失: {losses['val']:.4f}")
        # 取样
        xb, yb = get_batch('train')
        # 前向传播
        logits, loss = model(xb, yb)
        # 梯度重置
        optimizer.zero_grad(set_to_none=True)
        # 反向传播，计算新的梯度的过程
        loss.backward()
        # 做一步优化计算
        optimizer.step()

    print("==========> 训练结束，开始生成新的内容")

    max_new_tokens = 500
    start_idx = random.randint(0, len(val_data) - block_size - max_new_tokens)
    # 上文内容
    context = torch.zeros((1, block_size), dtype=torch.long, device=device)  # (B, T) = 1, T = block_size
    context[0, :] = val_data[start_idx:start_idx + block_size]
    context_str = decode_text(context[0].tolist())  # 一阶张量
    wrapped_context_str = textwrap.fill(context_str, wrap_width)

    # 真实上下文
    real_next_tokens = torch.zeros((1, max_new_tokens), dtype=torch.long, device=device)  # (B, T) = 1, T = block_size
    real_next_tokens[0, :] = val_data[start_idx + block_size:start_idx + block_size + max_new_tokens]
    real_next_tokens_str = decode_text(real_next_tokens[0].tolist())  # 一阶张量
    wrapped_real_next_tokens_str = textwrap.fill(real_next_tokens_str, width=wrap_width)

    # 生成下文
    generated_tokens = model.generate(context, max_new_tokens)
    generated_str = decode_text(generated_tokens[0].tolist())
    wrapped_generated_str = textwrap.fill(generated_str, width=wrap_width)

    print("=====> 上文内容：")
    print(wrapped_context_str)
    print("=====> 生成内容：")
    print(wrapped_generated_str)
    print("=====> 真实下文内容：")
    print(real_next_tokens_str)


# -------------执行-------------
main()
