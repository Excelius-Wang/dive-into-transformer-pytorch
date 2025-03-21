import os
import pathlib

import torch


class TransformerConfig:
    """Transformer模型的配置类"""
    # 训练参数
    BATCH_SIZE = 128  # 同时平行处理多少条独立数据
    BLOCK_SIZE = 256  # 训练、验证的字符串长度
    LEARNING_RATE = 3e-4
    MAX_ITERS = 300  # 减少迭代次数，加快训练速度
    EVAL_INTERVAL = MAX_ITERS // 10
    EVAL_ITERS = 200

    # 模型参数
    EMBEDDING_DIM = 512  # 嵌入的维度，尽量为 2 的整数次幂
    NUM_HEADS = 8
    NUM_LAYERS = 12
    HEAD_SIZE = EMBEDDING_DIM // NUM_HEADS
    DROPOUT_RATE = 0.2

    # 文本生成参数
    MAX_NEW_TOKENS = 500
    WRAP_WIDTH = 50

    # GPU设置
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 模型保存与加载参数
    CHECKPOINT_DIR = os.path.join(os.getcwd(), "checkpoints")  # 检查点保存目录
    SAVE_INTERVAL = 10000  # 设置一个非常大的值，实际上禁用了定期保存
    KEEP_LAST_CHECKPOINTS = 0  # 不保留任何检查点
    SAVE_BEST_MODEL = False  # 不保存最佳模型
    RESUME_TRAINING = False  # 不恢复训练
    CHECKPOINT_PATH = None  # 不指定检查点路径
    LOAD_BEST_MODEL = False  # 不加载最佳模型
    DISABLE_CHECKPOINTS = True  # 完全禁用检查点功能

    # 文件路径
    DATA_PATH = os.path.abspath(os.path.dirname(os.getcwd())) + '/data/raw/Hong_Lou_Meng.txt'

    def __init__(self):
        """初始化配置，检测可用GPU数量"""
        self.gpu_count = torch.cuda.device_count()

    def __str__(self):
        """将配置转换为字符串表示"""
        attrs = vars(self)
        return '\n'.join([f"{key}: {value}" for key, value in attrs.items()])
