#!/bin/bash
# 残差剥离塔(RST)训练脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "  RST (Residual-Stripping Tower) Training"
echo "=========================================="
echo ""

# 检查环境
echo "🔍 检查环境依赖..."
if ! python -c "import librosa" 2>/dev/null; then
    echo "⚠️  警告: librosa未安装，F0提取将失败"
    echo "   请运行: pip install librosa"
    read -p "是否继续？(y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查数据
echo "📁 检查数据路径..."
DATA_DIRS=(
    "datas/ASVSpoof2019/train.tsv"
    "datas/ASVSpoof2019/dev.tsv"
    "datas/ASVSpoof2019/ASVspoof2019.LA.cm.train.trn.txt"
    "datas/datasets/ASVSpoof2019_Hubert_L9"
)

for dir in "${DATA_DIRS[@]}"; do
    if [ ! -e "$dir" ]; then
        echo "❌ 错误: 未找到 $dir"
        echo "   请检查README_RST.md中的数据准备步骤"
        exit 1
    fi
done

echo "✅ 数据检查通过"
echo ""

# 询问是否预计算F0
echo "💡 提示: 预计算F0特征可以显著加快训练速度"
echo ""
echo "选择F0提取方式:"
echo "  1) 在线提取F0（慢但无需预处理）"
echo "  2) 使用预计算F0（快，需要先运行extract_f0.py）"
echo "  3) 现在预计算F0然后训练"
echo ""
read -p "请选择 (1/2/3) [默认: 1]: " f0_choice
f0_choice=${f0_choice:-1}

if [ "$f0_choice" == "3" ]; then
    echo ""
    echo "📊 开始预计算F0特征..."
    
    # 创建F0目录
    mkdir -p datas/datasets/ASVSpoof2019_F0/{train,dev,eval}
    
    # 提取训练集F0
    echo "  处理训练集..."
    python datas/extract_f0.py \
        --input_dir datas/datasets/ASVSpoof2019/ASVspoof2019_LA_train/flac \
        --output_dir datas/datasets/ASVSpoof2019_F0/train \
        --method librosa \
        --sample_rate 16000 \
        --hop_length 320 \
        --ext flac
    
    # 提取验证集F0
    echo "  处理验证集..."
    python datas/extract_f0.py \
        --input_dir datas/datasets/ASVSpoof2019/ASVspoof2019_LA_dev/flac \
        --output_dir datas/datasets/ASVSpoof2019_F0/dev \
        --method librosa \
        --sample_rate 16000 \
        --hop_length 320 \
        --ext flac
    
    # 提取测试集F0
    echo "  处理测试集..."
    python datas/extract_f0.py \
        --input_dir datas/datasets/ASVSpoof2019/ASVspoof2019_LA_eval/flac \
        --output_dir datas/datasets/ASVSpoof2019_F0/eval \
        --method librosa \
        --sample_rate 16000 \
        --hop_length 320 \
        --ext flac
    
    echo "✅ F0特征预计算完成"
    f0_choice="2"
fi

# 创建临时配置
CONFIG_FILE="config/train_rst.yaml"
TEMP_CONFIG="config/train_rst_temp.yaml"

cp "$CONFIG_FILE" "$TEMP_CONFIG"

if [ "$f0_choice" == "2" ]; then
    echo "📝 配置使用预计算F0..."
    # 修改配置使用预计算F0（简化版，实际可能需要更精确的yaml编辑）
    # 这里假设用户已经手动修改了配置，或者使用默认配置
    echo "   请确保config/train_rst.yaml中设置了正确的f0_dir"
fi

echo ""
echo "🚀 开始训练..."
echo ""
echo "配置文件: $TEMP_CONFIG"
echo "实验名称: RST_ASVspoof19"
echo "输出目录: Exps/RST_ASVspoof19/"
echo ""
echo "按Ctrl+C可随时中断训练"
echo ""
sleep 2

# 开始训练
python train_rst.py --conf_dir "$TEMP_CONFIG" --mode train

echo ""
echo "=========================================="
echo "✅ 训练完成!"
echo "=========================================="
echo ""
echo "检查点保存在: Exps/RST_ASVspoof19/checkpoints/"
echo "日志保存在: Exps/RST_ASVspoof19/logs/"
echo ""
echo "运行测试:"
echo "  python train_rst.py \\"
echo "    --conf_dir Exps/RST_ASVspoof19/config.yaml \\"
echo "    --mode test \\"
echo "    --ckpt Exps/RST_ASVspoof19/checkpoints/best_model.ckpt"
echo ""
