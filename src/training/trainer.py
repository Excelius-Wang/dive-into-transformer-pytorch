import torch
import torch.nn as nn
import time
import os
from src.utils.logger import printlog
from src.training.evaluation import evaluate_loss
from src.utils.model_utils import ModelCheckpoint

class ModelTrainer:
    """模型训练器，负责模型训练过程"""
    
    def __init__(self, model, dataset, config):
        """初始化训练器
        
        Args:
            model: 要训练的模型
            dataset: 训练数据集
            config: 配置对象，包含训练参数
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = config.DEVICE
        
        # 配置多GPU训练
        if config.gpu_count > 1:
            printlog(f"使用 {config.gpu_count} 个GPU并行训练")
            self.model = nn.DataParallel(self.model)
        else:
            printlog("使用单GPU训练")
            
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.LEARNING_RATE
        )
        
        # 初始化检查点管理器
        self.checkpoints_enabled = not hasattr(config, 'DISABLE_CHECKPOINTS') or not config.DISABLE_CHECKPOINTS
        
        if self.checkpoints_enabled:
            printlog("检查点功能已启用")
            self.checkpoint_manager = ModelCheckpoint(config, config.CHECKPOINT_DIR)
            
            # 恢复训练（如果需要）
            self.start_step = 0
            if config.RESUME_TRAINING:
                self._resume_from_checkpoint()
        else:
            printlog("检查点功能已禁用")
            self.start_step = 0
        
        # 打印模型信息
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        printlog(f"模型参数量: {n_params:.2f} M")
    
    def _resume_from_checkpoint(self):
        """从检查点恢复训练"""
        if not self.checkpoints_enabled:
            printlog("检查点功能已禁用，无法恢复训练")
            return
            
        printlog("尝试从检查点恢复训练...")
        self.model, self.optimizer, self.start_step = self.checkpoint_manager.load_for_training(
            self.model, self.optimizer, 
            checkpoint_path=self.config.CHECKPOINT_PATH,
            load_best=self.config.LOAD_BEST_MODEL
        )
        if self.start_step > 0:
            printlog(f"成功恢复训练，从步骤 {self.start_step} 开始")
        else:
            printlog("没有找到有效的检查点，从头开始训练")
    
    def _cleanup_old_checkpoints(self):
        """清理旧的检查点文件，只保留最新的几个"""
        if not self.checkpoints_enabled:
            return
            
        if self.config.KEEP_LAST_CHECKPOINTS <= 0:
            return
            
        checkpoint_files = self.checkpoint_manager.list_checkpoints()
        # 排除best_model.pt
        checkpoint_files = [f for f in checkpoint_files if "best_model.pt" not in f]
        
        # 按创建时间排序（最新的在最后）
        checkpoint_files.sort(key=os.path.getctime)
        
        # 删除旧的检查点
        if len(checkpoint_files) > self.config.KEEP_LAST_CHECKPOINTS:
            for old_checkpoint in checkpoint_files[:-self.config.KEEP_LAST_CHECKPOINTS]:
                try:
                    os.remove(old_checkpoint)
                    printlog(f"删除旧检查点: {old_checkpoint}")
                except Exception as e:
                    printlog(f"无法删除检查点 {old_checkpoint}: {e}")
    
    def train(self):
        """训练模型
        
        Returns:
            训练完成的模型
        """
        start_time = time.time()
        best_val_loss = float('inf')
        
        for step in range(self.start_step, self.config.MAX_ITERS):
            # 定期评估损失
            if step % self.config.EVAL_INTERVAL == 0 or step == self.config.MAX_ITERS - 1:
                losses = evaluate_loss(self.model, self.dataset, self.config)
                val_loss = losses['val']
                printlog(f"步骤 {step}: | 训练损失: {losses['train']:.4f} | 验证损失: {val_loss:.4f}")
                
                # 如果启用了检查点功能，保存模型检查点
                if self.checkpoints_enabled:
                    # 保存模型检查点
                    self.checkpoint_manager.save_on_interval(
                        self.model, self.optimizer, self.dataset, 
                        step, val_loss, self.config.SAVE_INTERVAL
                    )
                    
                    # 清理旧检查点
                    self._cleanup_old_checkpoints()
                    
                    # 更新最佳验证损失
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        if self.config.SAVE_BEST_MODEL:
                            self.checkpoint_manager.save_checkpoint(
                                self.model, self.optimizer, self.dataset, 
                                step, val_loss, is_best=True
                            )
            
            # 获取一批数据
            inputs, targets = self.dataset.get_batch('train')
            
            # 前向传播
            logits, loss = self.model(inputs, targets)
            
            # 反向传播
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        
        # 保存最终模型
        if self.checkpoints_enabled:
            final_losses = evaluate_loss(self.model, self.dataset, self.config)
            self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.dataset, 
                self.config.MAX_ITERS - 1, final_losses['val'],
                additional_info={'training_time': time.time() - start_time}
            )
        
        train_time = time.time() - start_time
        printlog(f"训练完成! 用时: {train_time:.2f} 秒")
        return self.model 