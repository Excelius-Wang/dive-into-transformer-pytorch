
# 基于Transformer的中文文本生成模型

## 项目简介

这是一个专门针对中文文本的自回归语言模型，采用GPT风格的Transformer架构实现。项目以《红楼梦》作为训练语料，能够学习古典文学的语言风格并生成具有相似特征的文本内容。

该模型实现了完整的Transformer解码器架构，包括多头自注意力机制、位置编码、层归一化等核心组件，专门为中文文本处理进行了优化。通过大规模文本训练，模型能够捕捉到中文语言的语法结构和语义特征，实现高质量的文本续写功能。

## 技术特点

- **中文文本优化**：采用jieba分词器进行中文文本预处理，构建专门的中文词表
- **多头自注意力机制**：实现8个注意力头的并行计算，提升模型对长距离依赖的建模能力
- **多GPU并行训练**：支持DataParallel多GPU训练，显著提升训练效率
- **模块化架构设计**：清晰的代码结构，便于维护和扩展
- **灵活的文本生成**：支持从随机上下文或指定文本开始的续写生成
- **完整的训练流程**：包含数据预处理、模型训练、检查点管理和文本生成的完整pipeline

## 项目结构

```
dive-into-transformer-pytorch/
├── data/                     # 数据目录
│   ├── raw/                  # 原始训练数据
│   │   └── Hong_Lou_Meng.txt # 红楼梦文本语料
│   ├── processed/            # 预处理后的数据
│   └── splits/               # 数据集分割
├── src/                      # 源代码目录
│   ├── config/               # 模型配置文件
│   │   └── config.py         # 超参数和训练配置
│   ├── data_loader/          # 数据加载模块
│   │   └── dataset.py        # 文本数据集处理
│   ├── models/               # 模型架构定义
│   │   ├── language_model.py # GPT语言模型主体
│   │   ├── attention.py      # 注意力机制实现
│   │   └── blocks.py         # Transformer块
│   ├── training/             # 训练流程
│   │   ├── trainer.py        # 模型训练器
│   │   └── evaluation.py     # 模型评估
│   ├── generate/             # 文本生成
│   │   └── generator.py      # 文本生成器
│   ├── utils/                # 工具函数
│   │   ├── logger.py         # 日志记录
│   │   ├── model_utils.py    # 模型工具
│   │   └── text_processor.py # 文本处理
│   └── main.py               # 主程序入口
└── requirements.txt          # 项目依赖
```

## 技术栈

### 核心依赖
- **PyTorch**: 深度学习框架，提供张量计算和自动微分功能
- **jieba**: 中文分词库，用于文本预处理和词表构建
- **NumPy**: 数值计算库，支持多维数组操作
- **Python 3.x**: 项目开发语言

### 模型架构
- **Transformer Decoder**: 基于GPT架构的自回归语言模型
- **Multi-Head Attention**: 8头自注意力机制，嵌入维度512
- **Position Encoding**: 位置编码，支持最大序列长度256
- **Layer Normalization**: 层归一化，提升训练稳定性
- **Feed-Forward Network**: 前馈神经网络，隐藏层维度2048

## 安装与配置

### 环境要求
- Python 3.7+
- CUDA兼容的GPU（推荐）
- 8GB+ 显存（用于训练）

### 安装步骤

1. 克隆项目仓库
```bash
git clone https://github.com/yourusername/dive-into-transformer-pytorch.git
cd dive-into-transformer-pytorch
```

2. 安装依赖包
```bash
pip install torch jieba numpy pathlib
```

3. 准备训练数据
将中文文本文件放置到`data/raw/`目录下，默认使用`Hong_Lou_Meng.txt`作为训练语料

## 使用方法

### 模型训练
```bash
python src/main.py --train
```

训练过程将自动进行数据预处理、模型初始化和迭代优化。训练参数可在`src/config/config.py`中调整。

### 文本生成

**随机上下文生成**：
```bash
python src/main.py --generate
```

**指定上下文续写**：
```bash
python src/main.py --generate --context "贾宝玉看着林黛玉道："
```

**控制生成长度**：
```bash
python src/main.py --generate --max_tokens 500
```

## 技术实现

### 模型架构详解

**词嵌入层（Token Embedding）**
- 将离散的词汇映射到512维连续向量空间
- 使用可学习的嵌入矩阵，词表大小根据训练语料动态确定

**位置编码（Position Encoding）**
- 可学习的位置嵌入，最大支持256个token的序列长度
- 与词嵌入相加，为模型提供序列位置信息

**多头自注意力（Multi-Head Self-Attention）**
- 8个并行的注意力头，每个头维度64
- 使用因果掩码确保自回归特性
- Dropout正则化防止过拟合

**前馈神经网络（Feed-Forward Network）**
- 两层线性变换，中间层维度2048
- ReLU激活函数
- 残差连接和层归一化

### 关键参数配置
- **嵌入维度**: 512
- **注意力头数**: 8
- **Transformer层数**: 12
- **上下文窗口**: 256 tokens
- **学习率**: 3e-4
- **批次大小**: 128
- **Dropout率**: 0.2

## 项目特色

### 中文文本处理优化
- 基于jieba分词的中文文本预处理pipeline
- 针对古典文学语料的特殊处理
- 动态词表构建，适应不同规模的训练数据

### 训练效率优化
- 支持多GPU并行训练，提升训练速度
- 灵活的检查点管理系统
- 实时训练监控和损失可视化

### 生成质量控制
- 基于softmax采样的文本生成策略
- 可配置的生成长度和上下文窗口
- 支持批量生成和交互式生成

## 参考视频

### 教学视频
- [PyTorch手搓Transformer](https://www.bilibili.com/video/BV1BbFaeVE4W)

## 开发说明

本项目采用模块化设计，主要包含以下组件：
- `config/`: 模型和训练参数配置
- `models/`: Transformer模型实现
- `data_loader/`: 数据预处理和加载
- `training/`: 训练流程和评估
- `generate/`: 文本生成功能
- `utils/`: 工具函数和辅助模块

代码结构清晰，便于理解和扩展。适合作为Transformer架构学习和中文文本生成的参考实现。
