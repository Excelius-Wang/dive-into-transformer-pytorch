import os
import json
import time
from typing import Dict, List, Optional
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from config import Config

class TrainingMonitor:
    """训练监控类，用于记录和可视化训练过程"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.metrics_file = os.path.join(log_dir, "training_metrics.json")
        self.plots_dir = os.path.join(log_dir, "plots")
        
        # 创建必要的目录
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # 初始化指标存储
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "iteration": [],
            "timestamp": [],
            "gpu_memory": [],
            "tokens_per_second": []
        }
        
        # 加载已有的指标（如果存在）
        self.load_metrics()
        
        # 训练开始时间
        self.start_time = time.time()
        self.last_log_time = time.time()
        
    def load_metrics(self):
        """加载已保存的训练指标"""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    saved_metrics = json.load(f)
                    self.metrics.update(saved_metrics)
                logger.info(f"已加载训练指标，共 {len(self.metrics['iteration'])} 个记录")
            except Exception as e:
                logger.warning(f"加载训练指标失败: {e}")
    
    def save_metrics(self):
        """保存训练指标到文件"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            logger.error(f"保存训练指标失败: {e}")
    
    def log_iteration(self, iteration: int, train_loss: float, val_loss: Optional[float] = None, 
                     learning_rate: float = 0.0, tokens_processed: int = 0):
        """记录单次迭代的指标"""
        current_time = time.time()
        
        # 计算tokens/秒
        time_diff = current_time - self.last_log_time
        tokens_per_second = tokens_processed / time_diff if time_diff > 0 else 0
        
        # 获取GPU内存使用情况
        gpu_memory = 0
        device = Config.get_device()
        if device.startswith('cuda'):
            try:
                import torch
                gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            except:
                pass
        
        # 记录指标（确保所有值都是JSON可序列化的）
        self.metrics["iteration"].append(int(iteration))
        self.metrics["train_loss"].append(float(train_loss) if train_loss is not None else None)
        self.metrics["val_loss"].append(float(val_loss) if val_loss is not None else None)
        self.metrics["learning_rate"].append(float(learning_rate))
        self.metrics["timestamp"].append(current_time)
        self.metrics["gpu_memory"].append(float(gpu_memory))
        self.metrics["tokens_per_second"].append(float(tokens_per_second))
        
        # 更新时间
        self.last_log_time = current_time
        
        # 定期保存指标
        if iteration % 100 == 0:
            self.save_metrics()
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """绘制训练曲线"""
        if len(self.metrics["iteration"]) == 0:
            logger.warning("没有训练数据可绘制")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Monitoring Dashboard', fontsize=16)
        
        iterations = self.metrics["iteration"]
        
        # 损失曲线
        ax1 = axes[0, 0]
        ax1.plot(iterations, self.metrics["train_loss"], label='Train Loss', color='blue', alpha=0.7)
        if any(v is not None for v in self.metrics["val_loss"]):
            val_iterations = [it for it, val in zip(iterations, self.metrics["val_loss"]) if val is not None]
            val_losses = [val for val in self.metrics["val_loss"] if val is not None]
            ax1.plot(val_iterations, val_losses, label='Validation Loss', color='red', alpha=0.7)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 学习率曲线
        ax2 = axes[0, 1]
        ax2.plot(iterations, self.metrics["learning_rate"], color='green', alpha=0.7)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        
        # GPU内存使用
        ax3 = axes[1, 0]
        ax3.plot(iterations, self.metrics["gpu_memory"], color='orange', alpha=0.7)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('GPU Memory (GB)')
        ax3.set_title('GPU Memory Usage')
        ax3.grid(True, alpha=0.3)
        
        # 训练速度
        ax4 = axes[1, 1]
        # 使用滑动平均平滑曲线
        if len(self.metrics["tokens_per_second"]) > 10:
            window_size = min(50, len(self.metrics["tokens_per_second"]) // 10)
            smoothed_speed = np.convolve(self.metrics["tokens_per_second"], 
                                       np.ones(window_size)/window_size, mode='valid')
            smoothed_iterations = iterations[window_size-1:]
            ax4.plot(smoothed_iterations, smoothed_speed, color='purple', alpha=0.7)
        else:
            ax4.plot(iterations, self.metrics["tokens_per_second"], color='purple', alpha=0.7)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Tokens/Second')
        ax4.set_title('Training Speed')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            save_path = os.path.join(self.plots_dir, f"training_curves_{int(time.time())}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"训练曲线已保存: {save_path}")
    
    def get_summary_stats(self) -> Dict:
        """获取训练统计摘要"""
        if len(self.metrics["iteration"]) == 0:
            return {}
        
        current_time = time.time()
        total_time = current_time - self.start_time
        
        stats = {
            "total_iterations": len(self.metrics["iteration"]),
            "current_iteration": self.metrics["iteration"][-1] if self.metrics["iteration"] else 0,
            "latest_train_loss": self.metrics["train_loss"][-1] if self.metrics["train_loss"] else 0,
            "best_train_loss": min(self.metrics["train_loss"]) if self.metrics["train_loss"] else 0,
            "latest_val_loss": next((v for v in reversed(self.metrics["val_loss"]) if v is not None), None),
            "current_lr": self.metrics["learning_rate"][-1] if self.metrics["learning_rate"] else 0,
            "avg_gpu_memory": np.mean(self.metrics["gpu_memory"]) if self.metrics["gpu_memory"] else 0,
            "avg_tokens_per_second": np.mean(self.metrics["tokens_per_second"]) if self.metrics["tokens_per_second"] else 0,
            "total_training_time": total_time,
            "training_time_formatted": self._format_time(total_time)
        }
        
        return stats
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def print_summary(self):
        """打印训练摘要"""
        stats = self.get_summary_stats()
        if not stats:
            logger.info("暂无训练数据")
            return
        
        logger.info("=== 训练摘要 ===")
        logger.info(f"总迭代次数: {stats['total_iterations']}")
        logger.info(f"当前迭代: {stats['current_iteration']}")
        logger.info(f"最新训练损失: {stats['latest_train_loss']:.4f}")
        logger.info(f"最佳训练损失: {stats['best_train_loss']:.4f}")
        if stats['latest_val_loss'] is not None:
            logger.info(f"最新验证损失: {stats['latest_val_loss']:.4f}")
        logger.info(f"当前学习率: {stats['current_lr']:.2e}")
        logger.info(f"平均GPU内存使用: {stats['avg_gpu_memory']:.2f} GB")
        logger.info(f"平均训练速度: {stats['avg_tokens_per_second']:.0f} tokens/秒")
        logger.info(f"总训练时间: {stats['training_time_formatted']}")
        logger.info("===============")

# 全局监控实例
monitor = TrainingMonitor()