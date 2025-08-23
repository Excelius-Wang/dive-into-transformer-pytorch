"""训练配置文件

包含所有模型和训练相关的超参数配置。
"""

import os

class Config:
    """训练配置类"""

    # 模型超参数 - 针对双3090优化
    batch_size = 16  # 进一步减小批次大小
    block_size = 256  # 序列长度
    max_iters = 5000  # 训练轮数
    eval_interval = 200  # 评估间隔
    learning_rate = 1e-4  # 初始学习率
    eval_iters = 100  # 评估迭代次数
    n_embd = 768  # 嵌入维度
    n_head = 12  # 注意力头数
    n_layer = 12  # Transformer层数
    dropout = 0.1  # Dropout率

    # 训练优化参数
    warmup_iters = 500  # 学习率预热步数
    min_lr = 1e-5  # 最小学习率
    weight_decay = 0.1  # 权重衰减
    grad_clip = 1.0  # 梯度裁剪

    # 数据和输出配置
    data_file = "data/raw/Hong_Lou_Meng.txt"
    log_dir = "logs"
    wrap_width = 80  # 文本换行宽度

    # 分布式训练配置
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # 文本生成配置
    max_new_tokens = 500  # 生成文本长度

    @classmethod
    def get_device(cls):
        """获取设备"""
        import torch
        return f'cuda:{cls.local_rank}' if torch.cuda.is_available() else 'cpu'

    # 检查点保存间隔
    save_interval = 1000

    @classmethod
    def is_main_process(cls):
        """判断是否为主进程"""
        return cls.rank == 0

    @classmethod
    def validate(cls):
        """验证配置参数"""
        assert cls.n_embd % cls.n_head == 0, "嵌入维度必须能被注意力头数整除"
        assert cls.block_size > 0, "序列长度必须大于0"
        assert cls.learning_rate > cls.min_lr, "初始学习率必须大于最小学习率"
        return True