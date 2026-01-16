# 残差剥离塔 (RST) 训练指南

## 📋 训练前准备

### 1. 环境依赖

```bash
# 基础依赖（应该已安装）
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt

# F0提取依赖（必须安装）
pip install librosa  # 用于在线F0提取
```

### 2. 数据准备检查

确认以下文件/目录存在：

```
SafeEar/
├── datas/
│   ├── ASVSpoof2019/
│   │   ├── train.tsv
│   │   ├── dev.tsv
│   │   ├── eval.tsv
│   │   ├── ASVspoof2019.LA.cm.train.trn.txt
│   │   ├── ASVspoof2019.LA.cm.dev.trl.txt
│   │   └── ASVspoof2019.LA.cm.eval.trl.txt
│   │
│   └── datasets/
│       └── ASVSpoof2019_Hubert_L9/
│           ├── ASVspoof2019_LA_train/flac/  # HuBERT特征
│           ├── ASVspoof2019_LA_dev/flac/
│           └── ASVspoof2019_LA_eval/flac/
```

### 3. （可选但推荐）预计算F0特征

```bash
# 预计算训练集F0
python datas/extract_f0.py \
    --input_dir datas/datasets/ASVSpoof2019/ASVspoof2019_LA_train/flac \
    --output_dir datas/datasets/ASVSpoof2019_F0/train \
    --method librosa \
    --sample_rate 16000 \
    --hop_length 320 \
    --ext flac

# 预计算验证集F0
python datas/extract_f0.py \
    --input_dir datas/datasets/ASVSpoof2019/ASVspoof2019_LA_dev/flac \
    --output_dir datas/datasets/ASVSpoof2019_F0/dev \
    --method librosa \
    --sample_rate 16000 \
    --hop_length 320 \
    --ext flac

# 预计算测试集F0
python datas/extract_f0.py \
    --input_dir datas/datasets/ASVSpoof2019/ASVspoof2019_LA_eval/flac \
    --output_dir datas/datasets/ASVSpoof2019_F0/eval \
    --method librosa \
    --sample_rate 16000 \
    --hop_length 320 \
    --ext flac
```

如果预计算了F0，修改配置文件 `config/train_rst.yaml`：
```yaml
f0_dir: "datas/datasets/ASVSpoof2019_F0/train"  # 训练时会自动推断dev/eval路径
extract_f0_online: false
```

## 🚀 开始训练

### 方式1：直接训练（使用在线F0提取）

```bash
python train_rst.py --conf_dir config/train_rst.yaml --mode train
```

### 方式2：使用预计算F0训练（更快）

1. 先修改 `config/train_rst.yaml`：
```yaml
datamodule:
  DataClass_dict:
    f0_dir: "datas/datasets/ASVSpoof2019_F0"
    extract_f0_online: false
```

2. 训练：
```bash
python train_rst.py --conf_dir config/train_rst.yaml --mode train
```

## 📊 测试模型

```bash
python train_rst.py \
    --conf_dir Exps/RST_ASVspoof19/config.yaml \
    --mode test \
    --ckpt Exps/RST_ASVspoof19/checkpoints/best_model.ckpt
```

## ⚙️ 配置说明

### 关键配置参数

#### 数据配置
```yaml
datamodule:
  batch_size: 4           # 根据GPU内存调整
  num_workers: 8          # 数据加载线程数
  DataClass_dict:
    max_len: 64600        # 最大音频长度（采样点）
    f0_dir: null          # F0目录，null表示在线提取
    extract_f0_online: true  # 是否在线提取F0
```

#### 模型配置
```yaml
rst_model:
  # VQ层数配置
  n_q_semantic: 2         # 语义VQ层数
  n_q_speaker: 2          # 说话人VQ层数
  n_q_prosody: 2          # 韵律VQ层数
  n_q_residual: 0         # 残差VQ层数（0表示不量化）
  
  # 说话人模式
  num_speakers: 0         # 0=嵌入模式, >0=分类模式（设为训练集说话人数）
  
  # 特征融合
  feature_fusion: 'residual_only'  # 'residual_only', 'all_layers', 'weighted'
```

#### 训练配置
```yaml
system:
  lr: 3.0e-4              # 总体学习率
  lr_rst: 1.0e-4          # RST部分学习率
  lr_detector: 3.0e-4     # 检测器学习率
  
  # 损失权重
  semantic_weight: 1.0    # 语义蒸馏损失
  speaker_weight: 1.0     # 说话人损失
  prosody_weight: 1.0     # 韵律损失
  detection_weight: 1.0   # 检测损失
  gr_weight: 0.5          # 梯度反转对抗损失
  
  # 训练策略
  freeze_rst_epochs: 0    # 前N个epoch冻结RST（从0开始联合训练）
```

## 💡 训练技巧

### 1. 渐进式训练策略

**阶段1：先训练监督任务（可选）**
```yaml
system:
  detection_weight: 0.1   # 降低检测权重
  semantic_weight: 1.0
  speaker_weight: 1.0
  prosody_weight: 1.0
```

**阶段2：联合训练**
```yaml
system:
  detection_weight: 1.0   # 恢复检测权重
```

### 2. 说话人分类 vs 嵌入模式

如果训练集说话人数量固定且已知（ASVSpoof2019有20个说话人）：
```yaml
rst_model:
  num_speakers: 20        # 分类模式
system:
  use_speaker_classification: true
```

如果希望泛化到未见说话人（推荐）：
```yaml
rst_model:
  num_speakers: 0         # 嵌入模式
system:
  use_speaker_classification: false
```

### 3. 特征融合策略

- `residual_only`: 只用判伪残差（推荐，泛化性最强）
- `all_layers`: 使用所有层特征（性能更好但泛化性稍弱）
- `weighted`: 加权融合（平衡性能和泛化性）

## 📈 监控指标

训练过程中会记录：
- `train_loss`: 总训练损失
- `train_semantic`: 语义损失
- `train_speaker`: 说话人损失
- `train_prosody`: 韵律损失
- `train_detection`: 检测损失
- `val_eer`: 验证集EER（越低越好）
- `val_loss`: 验证集损失

## 🐛 常见问题

### 1. 内存不足
```yaml
datamodule:
  batch_size: 2  # 减小batch size
  num_workers: 4  # 减少数据加载线程
```

### 2. F0提取太慢
预计算F0特征（见上文）

### 3. 训练不稳定
```yaml
system:
  gr_weight: 0.1  # 降低梯度反转权重
  gradient_clip_val: 0.5  # 调整梯度裁剪
```

### 4. 说话人数量不确定
数据加载器会自动统计说话人数量，查看日志：
```
[ASVSpoof2019RST] Found 20 speakers
```

## 📁 输出目录结构

```
Exps/RST_ASVspoof19/
├── checkpoints/
│   ├── epoch=10-val_eer=0.0234.ckpt  # 最佳模型
│   └── last.ckpt                      # 最后一个epoch
├── logs/
│   └── wandb/                         # wandb日志
├── config.yaml                        # 训练配置备份
└── best_k_models.json                 # 最佳模型记录
```

## 🔬 实验建议

### 基线实验
1. **Baseline**: 只用残差特征，不做监督
```yaml
semantic_weight: 0.0
speaker_weight: 0.0
prosody_weight: 0.0
```

### 消融实验
2. **w/o Semantic**: 不剥离语义
```yaml
semantic_weight: 0.0
n_q_semantic: 0
```

3. **w/o Speaker**: 不剥离说话人
```yaml
speaker_weight: 0.0
n_q_speaker: 0
```

4. **w/o Prosody**: 不剥离韵律
```yaml
prosody_weight: 0.0
n_q_prosody: 0
```

5. **Full RST**: 完整模型（所有监督任务）
```yaml
semantic_weight: 1.0
speaker_weight: 1.0
prosody_weight: 1.0
```
