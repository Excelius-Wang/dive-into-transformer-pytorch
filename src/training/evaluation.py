import torch

@torch.no_grad()
def evaluate_loss(model, dataset, config):
    """评估模型在训练集和验证集上的损失
    
    Args:
        model: 需要评估的模型
        dataset: 数据集对象
        config: 配置对象，包含评估参数
        
    Returns:
        包含训练集和验证集平均损失的字典
    """
    results = {}
    model.eval()  # 切换到评估模式
    
    for split in ['train', 'val']:
        losses = torch.zeros(config.EVAL_ITERS)
        for i in range(config.EVAL_ITERS):
            inputs, targets = dataset.get_batch(split)
            logits, loss = model(inputs, targets)
            losses[i] = loss.item()
        results[split] = losses.mean()
        
    model.train()  # 切换回训练模式
    return results 