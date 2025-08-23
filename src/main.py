"""基于PyTorch的分布式Transformer语言模型实现

支持多GPU分布式训练的GPT语言模型，使用loguru进行日志管理。
"""

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from loguru import logger
from tqdm import tqdm

# 导入自定义模块
from config import Config
from model import GPTLanguageModel
from utils import (
    setup_logging, setup_distributed, cleanup_distributed,
    load_and_process_data, get_batch, get_lr, estimate_loss,
    print_model_info, save_checkpoint
)
from monitor import monitor

def main():
    """主训练函数"""
    # 验证配置
    Config.validate()

    # 设置随机种子
    torch.manual_seed(42)

    # 设置日志和分布式训练
    setup_logging()
    setup_distributed()

    device = Config.get_device()

    # 加载和处理数据
    train_data, val_data, vocab_size, encode, decode = load_and_process_data()

    device = Config.get_device()

    # 创建模型
    model = GPTLanguageModel(vocab_size)
    model = model.to(device)

    if Config.is_main_process():
        print_model_info(model)

    # 包装模型为分布式数据并行
    if Config.world_size > 1:
        model = DDP(model, device_ids=[Config.local_rank], output_device=Config.local_rank)
        if Config.is_main_process():
            logger.info("模型已包装为分布式数据并行")

    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay
    )

    if Config.is_main_process():
        logger.info(f"优化器配置: lr={Config.learning_rate}, weight_decay={Config.weight_decay}")

    # 训练循环
    model.train()
    if Config.is_main_process():
        logger.info("开始训练循环")
        logger.info("开始训练监控")
        train_pbar = tqdm(range(Config.max_iters), desc="训练进度")
    else:
        train_pbar = None

    tokens_processed = 0

    for iter_num in range(Config.max_iters):
        iter_start_time = time.time()

        # 更新学习率
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 定期评估损失（跳过第一次迭代）
        if iter_num > 0 and (iter_num % Config.eval_interval == 0 or iter_num == Config.max_iters - 1):
            eval_model = model.module if Config.world_size > 1 else model
            losses = estimate_loss(eval_model, train_data, val_data, device)
            if Config.is_main_process():
                logger.info(f"步骤 {iter_num}: 训练损失 {losses['train']:.4f}, 验证损失 {losses['val']:.4f}, 学习率 {lr:.2e}")
                # 更新监控系统中的验证损失
                if len(monitor.metrics["iteration"]) > 0:
                    monitor.metrics["val_loss"][-1] = float(losses['val'])

        # 获取训练批次
        xb, yb = get_batch('train', train_data, val_data, device)
        tokens_processed += xb.numel()

        # 前向传播和反向传播
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.grad_clip)

        optimizer.step()

        # 记录训练指标
        if Config.is_main_process():
            # 更新进度条
            if train_pbar is not None:
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{lr:.2e}'
                })
                train_pbar.update(1)

            # 记录到监控系统
            monitor.log_iteration(
                iteration=iter_num,
                train_loss=loss.item(),
                learning_rate=lr,
                tokens_processed=tokens_processed
            )

        # 保存检查点和生成图表
        if iter_num % Config.save_interval == 0 and iter_num > 0:
            if Config.is_main_process():
                save_model = model.module if Config.world_size > 1 else model
                save_checkpoint(save_model, optimizer, iter_num, loss.item())
                # 生成训练曲线
                monitor.plot_training_curves()
                monitor.print_summary()

    # 训练完成后的文本生成
    if Config.is_main_process():
        logger.info("训练完成，开始生成文本样本")

        gen_model = model.module if Config.world_size > 1 else model

        # 生成文本样本
        context = torch.zeros((1, Config.block_size), dtype=torch.long, device=device)
        start_idx = torch.randint(0, len(val_data) - Config.block_size, (1,)).item()
        context[0, :] = val_data[start_idx:start_idx + Config.block_size]

        with torch.no_grad():
            generated_tokens = gen_model.generate(context, Config.max_new_tokens)

        context_str = decode(context[0].tolist())
        generated_str = decode(generated_tokens[0].tolist())

        logger.info("\n" + "="*80)
        logger.info("上下文内容:")
        logger.info("="*80)
        logger.info(context_str)
        logger.info("\n" + "="*80)
        logger.info("生成的文本:")
        logger.info("="*80)
        logger.info(generated_str)
        logger.info("="*80)

        logger.info("文本生成完成！")

    # 清理分布式训练环境
    cleanup_distributed()

if __name__ == "__main__":
    main()