import os
import torch
import json
import time
import glob
import pickle
from pathlib import Path
from src.utils.logger import printlog

class ModelCheckpoint:
    """模型检查点管理类，负责模型的保存和加载"""
    
    def __init__(self, config, save_dir=None):
        """初始化模型检查点管理器
        
        Args:
            config: 配置对象
            save_dir: 保存目录，如果未指定则使用默认目录
        """
        self.config = config
        
        # 设置保存目录
        if save_dir is None:
            # 使用当前目录下的checkpoints文件夹
            root_dir = Path.cwd()
            self.save_dir = root_dir / "checkpoints"
        else:
            self.save_dir = Path(save_dir)
            
        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 记录最佳验证损失
        self.best_val_loss = float('inf')
        
    def _get_checkpoint_filename(self, step=None, is_best=False, timestamp=None):
        """生成检查点文件名
        
        Args:
            step: 训练步骤
            is_best: 是否为最佳模型
            timestamp: 时间戳，默认为当前时间
            
        Returns:
            检查点文件名
        """
        if timestamp is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
        if is_best:
            return f"best_model.pt"
        elif step is not None:
            return f"model_step_{step:06d}_{timestamp}.pt"
        else:
            return f"model_{timestamp}.pt"
    
    def save_checkpoint(self, model, optimizer, dataset, step, val_loss, is_best=False, additional_info=None):
        """保存模型检查点
        
        Args:
            model: 要保存的模型
            optimizer: 优化器
            dataset: 数据集对象
            step: 训练步骤
            val_loss: 验证损失
            is_best: 是否为最佳模型
            additional_info: 要保存的其它信息
            
        Returns:
            保存的文件路径
        """
        # 如果是多GPU模型，获取内部模型
        if isinstance(model, torch.nn.DataParallel):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
        
        # 准备要保存的数据
        checkpoint = {
            'step': step,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'vocab_size': dataset.vocab_size,
            'word_to_idx': dataset.word_to_idx,
            'idx_to_word': dataset.idx_to_word,
            'val_loss': val_loss,
            'config': {attr: getattr(self.config, attr) for attr in dir(self.config) 
                      if not attr.startswith('__') and not callable(getattr(self.config, attr))},
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 添加额外信息
        if additional_info is not None:
            checkpoint.update(additional_info)
        
        # 保存检查点
        checkpoint_filename = self._get_checkpoint_filename(step=step, is_best=is_best)
        checkpoint_path = self.save_dir / checkpoint_filename
        
        # 保存模型
        torch.save(checkpoint, checkpoint_path)
        
        # 如果是最佳模型，更新最佳损失
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            # 保存最佳模型
            best_model_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_model_path)
            printlog(f"保存最佳模型: {best_model_path} (验证损失: {val_loss:.4f})")
        
        printlog(f"保存检查点: {checkpoint_path}")
        return checkpoint_path
    
    def save_on_interval(self, model, optimizer, dataset, step, val_loss, interval, additional_info=None):
        """按间隔保存模型
        
        Args:
            model: 要保存的模型
            optimizer: 优化器
            dataset: 数据集对象
            step: 当前步骤
            val_loss: 验证损失
            interval: 保存间隔
            additional_info: 要保存的其它信息
            
        Returns:
            如果保存了模型，返回保存路径，否则返回None
        """
        if step % interval == 0 or step == self.config.MAX_ITERS - 1:
            is_best = val_loss < self.best_val_loss
            return self.save_checkpoint(
                model, optimizer, dataset, step, val_loss, 
                is_best=is_best, additional_info=additional_info
            )
        return None
    
    def load_checkpoint(self, checkpoint_path=None, load_best=False, device=None):
        """加载模型检查点
        
        Args:
            checkpoint_path: 检查点文件路径，如果未指定则加载最新检查点
            load_best: 是否加载最佳模型
            device: 加载模型的设备
            
        Returns:
            加载的检查点数据
        """
        if device is None:
            device = self.config.DEVICE
            
        # 确定要加载的检查点文件
        if load_best:
            checkpoint_path = self.save_dir / "best_model.pt"
        elif checkpoint_path is None:
            # 加载最新的检查点
            checkpoint_files = glob.glob(str(self.save_dir / "model_step_*.pt"))
            if not checkpoint_files:
                printlog("没有找到可用的检查点文件")
                return None
            
            checkpoint_path = max(checkpoint_files, key=os.path.getctime)
            
        # 检查文件是否存在
        if not os.path.exists(checkpoint_path):
            printlog(f"检查点文件不存在: {checkpoint_path}")
            return None
            
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=device)
        printlog(f"加载检查点: {checkpoint_path}")
        printlog(f"模型训练步骤: {checkpoint['step']}, 验证损失: {checkpoint['val_loss']:.4f}")
        
        return checkpoint
    
    def load_for_inference(self, model, checkpoint_path=None, load_best=True):
        """加载模型用于推理
        
        Args:
            model: 模型对象
            checkpoint_path: 检查点文件路径
            load_best: 是否加载最佳模型
            
        Returns:
            加载了权重的模型和词表映射
        """
        checkpoint = self.load_checkpoint(checkpoint_path, load_best)
        if checkpoint is None:
            return None, None, None
            
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # 设置为评估模式
        
        # 返回模型、词表映射和加载的检查点
        word_to_idx = checkpoint['word_to_idx']
        idx_to_word = checkpoint['idx_to_word']
        
        return model, (word_to_idx, idx_to_word), checkpoint
    
    def load_for_training(self, model, optimizer, checkpoint_path=None, load_best=False):
        """加载模型用于继续训练
        
        Args:
            model: 模型对象
            optimizer: 优化器对象
            checkpoint_path: 检查点文件路径
            load_best: 是否加载最佳模型
            
        Returns:
            加载了权重的模型、优化器和起始步骤
        """
        checkpoint = self.load_checkpoint(checkpoint_path, load_best)
        if checkpoint is None:
            return model, optimizer, 0
            
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 设置最佳验证损失
        self.best_val_loss = checkpoint['val_loss']
        
        # 获取训练步骤
        start_step = checkpoint['step'] + 1
        
        return model, optimizer, start_step
    
    def list_checkpoints(self):
        """列出所有可用的检查点
        
        Returns:
            检查点文件列表
        """
        checkpoint_files = glob.glob(str(self.save_dir / "model_*.pt"))
        checkpoint_files.sort(key=os.path.getctime)
        
        if os.path.exists(self.save_dir / "best_model.pt"):
            checkpoint_files.append(str(self.save_dir / "best_model.pt"))
            
        return checkpoint_files 