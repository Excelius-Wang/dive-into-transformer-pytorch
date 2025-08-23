"""训练工具函数

包含数据处理、模型评估、分布式训练等工具函数。
"""

import torch
import torch.distributed as dist
import os
import math
from tqdm import tqdm
from loguru import logger
from config import Config

def setup_logging():
    """配置loguru日志系统"""
    if Config.rank == 0:
        logger.remove()  # 移除默认处理器
        
        # 创建日志目录
        os.makedirs(Config.log_dir, exist_ok=True)
        
        # 文件日志
        logger.add(
            f"{Config.log_dir}/training_{{time}}.log",
            rotation="100 MB",
            retention="10 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
        
        # 控制台日志
        logger.add(
            lambda msg: print(msg, end=""),
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}"
        )
    else:
        logger.remove()  # 非主进程不输出日志

def setup_distributed():
    """初始化分布式训练环境"""
    if Config.world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(Config.local_rank)
        if Config.rank == 0:
            logger.info(f"分布式训练初始化完成: {Config.world_size} GPUs")
            logger.info(f"主进程设备: {Config.get_device()}")
    else:
        if Config.rank == 0:
            logger.info(f"单GPU训练模式，设备: {Config.get_device()}")

def cleanup_distributed():
    """清理分布式训练环境"""
    if Config.world_size > 1:
        dist.destroy_process_group()
        if Config.rank == 0:
            logger.info("分布式训练环境已清理")

def load_and_process_data():
    """加载和预处理数据"""
    # 读取文本文件
    with open(Config.data_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 获取所有唯一字符
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # 创建字符到整数的映射
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    # 编码和解码函数
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    # 编码整个文本并转换为张量
    data = torch.tensor(encode(text), dtype=torch.long)
    
    # 分割训练和验证数据
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    if Config.rank == 0:
        logger.info(f"文件 {Config.data_file} 已读取和处理完成")
        logger.info(f"数据集大小: {len(data):,} 字符, 词汇表大小: {vocab_size}")
        logger.info(f"训练集: {len(train_data):,} 字符, 验证集: {len(val_data):,} 字符")
    
    return train_data, val_data, vocab_size, encode, decode

def get_batch(split, train_data, val_data, device):
    """生成一个批次的数据"""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - Config.block_size, (Config.batch_size,))
    x = torch.stack([data[i:i+Config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+Config.block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def get_lr(iter_num):
    """余弦退火学习率调度器"""
    if iter_num < Config.warmup_iters:
        return Config.learning_rate * iter_num / Config.warmup_iters
    if iter_num > Config.max_iters:
        return Config.min_lr
    decay_ratio = (iter_num - Config.warmup_iters) / (Config.max_iters - Config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return Config.min_lr + coeff * (Config.learning_rate - Config.min_lr)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, device):
    """评估模型在训练集和验证集上的损失"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(Config.eval_iters, device=device)
        if Config.rank == 0:
            pbar = tqdm(range(Config.eval_iters), desc=f"评估{split}集", leave=False)
        else:
            pbar = range(Config.eval_iters)
        
        for k in pbar:
            X, Y = get_batch(split, train_data, val_data, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        
        # 分布式训练中同步损失
        if Config.world_size > 1:
            dist.all_reduce(losses, op=dist.ReduceOp.SUM)
            losses /= Config.world_size
        
        out[split] = losses.mean()
        if Config.rank == 0:
            logger.info(f"{split}集平均损失: {out[split]:.4f}")
    
    model.train()
    return out

def print_model_info(model):
    """打印模型信息"""
    if Config.rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型参数总数: {total_params:,}")
        logger.info(f"可训练参数: {trainable_params:,}")
        logger.info(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")

def save_checkpoint(model, optimizer, iter_num, loss, filepath=None):
    """保存检查点"""
    if Config.rank == 0:
        if filepath is None:
            os.makedirs("checkpoints", exist_ok=True)
            filepath = f"checkpoints/checkpoint_iter_{iter_num}.pt"
        
        # 只保存可序列化的配置参数
        config_dict = {
            'learning_rate': Config.learning_rate,
            'batch_size': Config.batch_size,
            'block_size': Config.block_size,
            'max_iters': Config.max_iters,
            'eval_interval': Config.eval_interval,
            'eval_iters': Config.eval_iters,
            'n_embd': Config.n_embd,
            'n_head': Config.n_head,
            'n_layer': Config.n_layer,
            'dropout': Config.dropout,
            'weight_decay': Config.weight_decay,
            'grad_clip': Config.grad_clip,
            'warmup_iters': Config.warmup_iters,
            'min_lr': Config.min_lr,
            'max_new_tokens': Config.max_new_tokens
        }
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iter_num': iter_num,
            'loss': loss,
            'config': config_dict
        }
        torch.save(checkpoint, filepath)
        logger.info(f"检查点已保存: {filepath}")

def load_checkpoint(model, optimizer, filepath, device):
    """加载检查点"""
    checkpoint = torch.load(filepath, map_location=device)
    
    if Config.world_size > 1:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if Config.rank == 0:
        logger.info(f"检查点已加载: {filepath}")
    
    return checkpoint['iter_num'], checkpoint['loss']