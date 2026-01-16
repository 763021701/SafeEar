#!/usr/bin/env python3
"""
RST训练环境检查脚本
检查所有依赖和数据是否就绪
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

# 颜色输出
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")

def print_success(text: str):
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_error(text: str):
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

def print_info(text: str):
    print(f"{Colors.BLUE}ℹ {text}{Colors.RESET}")


def check_dependencies() -> Tuple[bool, List[str]]:
    """检查Python依赖"""
    print_header("检查Python依赖")
    
    missing = []
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchaudio', 'TorchAudio'),
        ('pytorch_lightning', 'PyTorch Lightning'),
        ('hydra', 'Hydra'),
        ('omegaconf', 'OmegaConf'),
        ('numpy', 'NumPy'),
        ('librosa', 'Librosa (用于F0提取)'),
    ]
    
    for package, name in required_packages:
        try:
            __import__(package)
            print_success(f"{name}")
        except ImportError:
            print_error(f"{name} - 未安装")
            missing.append(package)
    
    # 检查可选依赖
    print("\n可选依赖:")
    try:
        import parselmouth
        print_success("Parselmouth (可选的F0提取方法)")
    except ImportError:
        print_warning("Parselmouth - 未安装 (可选)")
    
    return len(missing) == 0, missing


def check_data_files() -> Tuple[bool, List[str]]:
    """检查数据文件"""
    print_header("检查数据文件")
    
    base_dir = Path("datas")
    missing = []
    
    required_files = [
        "ASVSpoof2019/train.tsv",
        "ASVSpoof2019/dev.tsv",
        "ASVSpoof2019/eval.tsv",
        "ASVSpoof2019/ASVspoof2019.LA.cm.train.trn.txt",
        "ASVSpoof2019/ASVspoof2019.LA.cm.dev.trl.txt",
        "ASVSpoof2019/ASVspoof2019.LA.cm.eval.trl.txt",
    ]
    
    for file_path in required_files:
        full_path = base_dir / file_path
        if full_path.exists():
            print_success(f"{file_path}")
        else:
            print_error(f"{file_path} - 未找到")
            missing.append(str(full_path))
    
    return len(missing) == 0, missing


def check_audio_data() -> Tuple[bool, List[str]]:
    """检查音频数据目录"""
    print_header("检查音频数据")
    
    base_dir = Path("datas/datasets")
    missing = []
    
    # 检查音频文件
    audio_dirs = [
        "ASVSpoof2019/ASVspoof2019_LA_train/flac",
        "ASVSpoof2019/ASVspoof2019_LA_dev/flac",
        "ASVSpoof2019/ASVspoof2019_LA_eval/flac",
    ]
    
    for dir_path in audio_dirs:
        full_path = base_dir / dir_path
        if full_path.exists():
            # 统计音频文件数量
            audio_files = list(full_path.glob("*.flac"))
            if audio_files:
                print_success(f"{dir_path} ({len(audio_files)} 文件)")
            else:
                print_warning(f"{dir_path} - 目录存在但无音频文件")
                missing.append(str(full_path))
        else:
            print_error(f"{dir_path} - 未找到")
            missing.append(str(full_path))
    
    return len(missing) == 0, missing


def check_hubert_features() -> Tuple[bool, List[str]]:
    """检查HuBERT特征"""
    print_header("检查HuBERT特征")
    
    base_dir = Path("datas/datasets/ASVSpoof2019_Hubert_L9")
    missing = []
    
    feature_dirs = [
        "ASVspoof2019_LA_train/flac",
        "ASVspoof2019_LA_dev/flac",
        "ASVspoof2019_LA_eval/flac",
    ]
    
    for dir_path in feature_dirs:
        full_path = base_dir / dir_path
        if full_path.exists():
            # 统计特征文件数量
            feat_files = list(full_path.glob("*.npy"))
            if feat_files:
                print_success(f"{dir_path} ({len(feat_files)} 文件)")
            else:
                print_warning(f"{dir_path} - 目录存在但无特征文件")
                missing.append(str(full_path))
        else:
            print_error(f"{dir_path} - 未找到")
            missing.append(str(full_path))
    
    return len(missing) == 0, missing


def check_f0_features() -> Tuple[bool, str]:
    """检查F0特征（可选）"""
    print_header("检查F0特征（可选）")
    
    base_dir = Path("datas/datasets/ASVSpoof2019_F0")
    
    if not base_dir.exists():
        print_warning("F0特征目录不存在")
        print_info("将使用在线F0提取（需要librosa）")
        return True, "online"
    
    f0_dirs = ["train", "dev", "eval"]
    all_exist = True
    
    for dir_name in f0_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            f0_files = list(dir_path.glob("*.npy"))
            if f0_files:
                print_success(f"{dir_name}/ ({len(f0_files)} 文件)")
            else:
                print_warning(f"{dir_name}/ - 目录存在但无F0文件")
                all_exist = False
        else:
            print_warning(f"{dir_name}/ - 未找到")
            all_exist = False
    
    if all_exist:
        print_info("建议在config中设置: f0_dir: 'datas/datasets/ASVSpoof2019_F0'")
        return True, "precomputed"
    else:
        print_info("F0特征不完整，将使用在线提取")
        return True, "online"


def check_config_file() -> bool:
    """检查配置文件"""
    print_header("检查配置文件")
    
    config_path = Path("config/train_rst.yaml")
    if config_path.exists():
        print_success(f"配置文件存在: {config_path}")
        
        # 读取并显示关键配置
        with open(config_path, 'r') as f:
            content = f.read()
            
        # 简单解析（不需要完整YAML解析）
        if 'batch_size:' in content:
            for line in content.split('\n'):
                if 'batch_size:' in line:
                    print_info(f"  {line.strip()}")
                elif 'extract_f0_online:' in line:
                    print_info(f"  {line.strip()}")
                elif 'num_speakers:' in line:
                    print_info(f"  {line.strip()}")
        
        return True
    else:
        print_error(f"配置文件未找到: {config_path}")
        return False


def check_gpu() -> bool:
    """检查GPU"""
    print_header("检查GPU")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print_success(f"CUDA可用")
            print_info(f"GPU数量: {gpu_count}")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                print_info(f"  GPU {i}: {gpu_name}")
            return True
        else:
            print_warning("CUDA不可用，将使用CPU训练（非常慢）")
            return False
    except Exception as e:
        print_error(f"检查GPU时出错: {e}")
        return False


def main():
    print(f"\n{Colors.BOLD}RST训练环境检查{Colors.RESET}")
    print(f"{Colors.BOLD}Residual-Stripping Tower Setup Checker{Colors.RESET}\n")
    
    checks = []
    
    # 依赖检查
    deps_ok, missing_deps = check_dependencies()
    checks.append(("依赖", deps_ok))
    
    # 数据检查
    data_ok, missing_data = check_data_files()
    checks.append(("数据文件", data_ok))
    
    audio_ok, missing_audio = check_audio_data()
    checks.append(("音频数据", audio_ok))
    
    hubert_ok, missing_hubert = check_hubert_features()
    checks.append(("HuBERT特征", hubert_ok))
    
    # F0检查（不影响总体状态）
    f0_ok, f0_mode = check_f0_features()
    
    # 配置检查
    config_ok = check_config_file()
    checks.append(("配置文件", config_ok))
    
    # GPU检查（不影响总体状态）
    gpu_ok = check_gpu()
    
    # 总结
    print_header("检查总结")
    
    all_ok = all(ok for _, ok in checks)
    
    for name, ok in checks:
        if ok:
            print_success(f"{name}: 通过")
        else:
            print_error(f"{name}: 失败")
    
    print()
    
    if all_ok:
        print_success(f"{Colors.BOLD}✓ 所有必要检查通过，可以开始训练！{Colors.RESET}\n")
        
        print(f"{Colors.BOLD}快速开始:{Colors.RESET}")
        print(f"  {Colors.GREEN}python train_rst.py --conf_dir config/train_rst.yaml --mode train{Colors.RESET}")
        print()
        print(f"{Colors.BOLD}或使用便捷脚本:{Colors.RESET}")
        print(f"  {Colors.GREEN}bash scripts/train_rst.sh{Colors.RESET}")
        print()
        
        if f0_mode == "online":
            print_warning("提示: 使用在线F0提取可能较慢")
            print_info("建议先预计算F0特征:")
            print(f"  {Colors.GREEN}python datas/extract_f0.py --input_dir ... --output_dir ...{Colors.RESET}")
        
        return 0
    else:
        print_error(f"{Colors.BOLD}✗ 存在问题，请修复后再训练{Colors.RESET}\n")
        
        # 提供修复建议
        if missing_deps:
            print(f"{Colors.YELLOW}安装缺失依赖:{Colors.RESET}")
            print(f"  pip install {' '.join(missing_deps)}")
            print()
        
        if missing_data or missing_audio or missing_hubert:
            print(f"{Colors.YELLOW}数据准备:{Colors.RESET}")
            print("  请参考 README_RST.md 中的数据准备步骤")
            print()
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
