"""
主程序入口，用于启动训练、评估或推理过程。
"""
import argparse
import os
import sys
import torch
import random
import numpy as np
from pathlib import Path

# 导入配置
from src.config.model_config import MODEL_CONFIGS


def set_seed(seed):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="深度学习项目命令行工具")
    
    # 基本参数
    parser.add_argument("--mode", type=str, default="train", 
                        choices=["train", "eval", "predict", "generate"],
                        help="运行模式")
    parser.add_argument("--config", type=str, default="classification",
                        help="配置名称或配置文件路径")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="模型检查点路径")
    
    # 数据相关参数
    parser.add_argument("--data_path", type=str, default=None,
                        help="数据路径")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录")
    
    # 训练相关参数
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="批量大小")
    parser.add_argument("--lr", type=float, default=None,
                        help="学习率")
    
    # 其他参数
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU ID，例如'0'或'0,1,2'")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子")
    parser.add_argument("--debug", action="store_true",
                        help="启用调试模式")
    
    return parser.parse_args()


def load_config(config_name_or_path):
    """
    加载配置
    
    参数:
        config_name_or_path: 配置名称或配置文件路径
        
    返回:
        配置字典
    """
    # 如果是配置名称
    if config_name_or_path in MODEL_CONFIGS:
        return MODEL_CONFIGS[config_name_or_path]
    
    # 如果是配置文件路径
    if os.path.exists(config_name_or_path):
        # TODO: 从文件加载配置
        pass
    
    # 默认使用分类配置
    print(f"未找到配置'{config_name_or_path}'，使用默认分类配置")
    return MODEL_CONFIGS["classification"]


def update_config_with_args(config, args):
    """
    使用命令行参数更新配置
    
    参数:
        config: 配置字典
        args: 命令行参数
        
    返回:
        更新后的配置字典
    """
    # 更新训练参数
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    
    if args.lr is not None:
        config["training"]["lr"] = args.lr
    
    # 更新硬件配置
    if args.gpu is not None:
        gpu_ids = [int(x) for x in args.gpu.split(",")]
        config["hardware"]["gpu_ids"] = gpu_ids
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # 更新其他参数
    if args.seed is not None:
        config["seed"] = args.seed
    
    if args.debug:
        config["debug"] = True
    
    # 更新路径参数
    if args.data_path is not None:
        config["data"]["raw_dir"] = args.data_path
    
    if args.output_dir is not None:
        config["checkpoints"]["save_dir"] = os.path.join(args.output_dir, "checkpoints")
        config["logging"]["log_dir"] = os.path.join(args.output_dir, "logs")
    
    return config


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 使用命令行参数更新配置
    config = update_config_with_args(config, args)
    
    # 设置随机种子
    set_seed(config["seed"])
    
    # 确保必要的目录存在
    os.makedirs(config["checkpoints"]["save_dir"], exist_ok=True)
    os.makedirs(config["logging"]["log_dir"], exist_ok=True)
    
    # 根据模式执行相应操作
    if args.mode == "train":
        # 导入训练模块
        from src.train import train
        train(config)
    
    elif args.mode == "eval":
        # 导入评估模块
        from src.inference import evaluate
        evaluate(config, args.checkpoint)
    
    elif args.mode == "predict":
        # 导入预测模块
        from src.inference import predict
        predict(config, args.checkpoint, args.data_path)
    
    elif args.mode == "generate":
        # 导入生成模块
        from src.generate.generator import generate
        generate(config, args.checkpoint, args.data_path)
    
    else:
        print(f"不支持的模式: {args.mode}")


if __name__ == "__main__":
    main() 