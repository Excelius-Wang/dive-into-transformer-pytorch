import argparse
import sys
import torch
from src.config.config import TransformerConfig
from src.data_loader.dataset import TextDataset
from src.models.language_model import GPTLanguageModel
from src.training.trainer import ModelTrainer
from src.generate.generator import TextGenerator
from src.utils.logger import printlog

def parse_args():
    """解析命令行参数
    
    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(description='Transformer语言模型训练与生成')
    
    # 检查是否有命令行参数
    if len(sys.argv) <= 1:
        printlog("未提供命令行参数，默认使用训练模式")
        sys.argv.append('--train')
    
    # 主要操作模式
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument('--train', action='store_true', help='训练模型')
    mode_group.add_argument('--generate', action='store_true', help='生成文本')
    
    # 生成参数
    parser.add_argument('--context', type=str, default=None, help='生成文本的上下文')
    parser.add_argument('--max_tokens', type=int, default=None, help='生成的最大token数')
    
    args = parser.parse_args()
    
    # 如果没有指定模式，默认为训练模式
    if not (args.train or args.generate):
        args.train = True
        printlog("未指定模式，默认使用训练模式")
    
    return args

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    config = TransformerConfig()
    
    # 设置配置
    if args.max_tokens is not None:
        config.MAX_NEW_TOKENS = args.max_tokens
    
    # 创建数据集
    printlog("加载数据集...")
    dataset = TextDataset(config)
    
    # 创建模型
    printlog("创建模型...")
    model = GPTLanguageModel(dataset.vocab_size, config).to(config.DEVICE)
    
    if args.train:
        # 训练模式
        printlog("开始训练模型...")
        trainer = ModelTrainer(model, dataset, config)
        model = trainer.train()
        printlog("训练完成!")
        
    elif args.generate:
        # 生成模式
        printlog("初始化生成器...")
        generator = TextGenerator(model, dataset, config)
        
        if args.context:
            # 使用用户提供的上下文
            printlog("使用提供的上下文生成文本...")
            context_text = args.context
            generated_text = generator.generate_from_text(context_text)
            # 只显示新生成的部分（不包含原始上下文）
            generator.print_generation_result(generated_text[len(context_text):], context_text)
        else:
            # 使用随机上下文
            printlog("使用随机上下文生成文本...")
            generated_text = generator.generate_random_completion()
            generator.print_generation_result(generated_text)

if __name__ == "__main__":
    main() 