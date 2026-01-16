"""
F0特征提取脚本

为ASVSpoof数据集预计算F0特征
支持两种提取方法：
1. librosa.pyin (默认)
2. parselmouth (praat)

用法:
    python extract_f0.py --input_dir datasets/ASVSpoof2019_LA_train/flac \
                         --output_dir datasets/ASVSpoof2019_F0/train \
                         --sample_rate 16000 \
                         --hop_length 320

输出格式:
    每个音频文件对应一个.npy文件，包含：
    {
        'f0': np.array,      # 原始F0序列
        'uv': np.array,      # voiced/unvoiced mask
        'voiced_mean': float, # voiced帧的F0均值 (log scale)
        'voiced_std': float   # voiced帧的F0标准差 (log scale)
    }
"""

import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 尝试导入不同的F0提取库
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not available")

try:
    import parselmouth
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    print("Warning: parselmouth not available")


def extract_f0_librosa(
    audio_path: str,
    sr: int = 16000,
    hop_length: int = 320,
    f0_min: float = 50.0,
    f0_max: float = 600.0
) -> dict:
    """
    使用librosa.pyin提取F0
    """
    # 加载音频
    y, _ = librosa.load(audio_path, sr=sr)
    
    # 提取F0
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=f0_min,
        fmax=f0_max,
        sr=sr,
        hop_length=hop_length,
        fill_na=0.0
    )
    
    # 处理NaN
    f0 = np.nan_to_num(f0, nan=0.0)
    uv = voiced_flag.astype(np.float32)
    
    # 计算voiced帧的统计信息
    voiced_f0 = f0[voiced_flag]
    if len(voiced_f0) > 0:
        log_f0 = np.log2(voiced_f0.clip(min=1.0))
        voiced_mean = float(log_f0.mean())
        voiced_std = float(log_f0.std()) if log_f0.std() > 0 else 1.0
    else:
        voiced_mean = 0.0
        voiced_std = 1.0
    
    return {
        'f0': f0.astype(np.float32),
        'uv': uv,
        'voiced_mean': voiced_mean,
        'voiced_std': voiced_std,
    }


def extract_f0_parselmouth(
    audio_path: str,
    sr: int = 16000,
    hop_length: int = 320,
    f0_min: float = 50.0,
    f0_max: float = 600.0
) -> dict:
    """
    使用parselmouth (Praat) 提取F0
    """
    import parselmouth
    from parselmouth.praat import call
    
    # 加载音频
    sound = parselmouth.Sound(audio_path)
    
    # 重采样
    if sound.sampling_frequency != sr:
        sound = sound.resample_to(sr)
    
    # 计算时间步
    time_step = hop_length / sr
    
    # 提取F0
    pitch = call(sound, "To Pitch", time_step, f0_min, f0_max)
    
    # 获取F0序列
    num_frames = call(pitch, "Get number of frames")
    f0 = np.zeros(num_frames, dtype=np.float32)
    uv = np.zeros(num_frames, dtype=np.float32)
    
    for i in range(num_frames):
        f0_value = call(pitch, "Get value in frame", i + 1, "Hertz")
        if not np.isnan(f0_value):
            f0[i] = f0_value
            uv[i] = 1.0
    
    # 计算voiced帧的统计信息
    voiced_mask = uv > 0.5
    voiced_f0 = f0[voiced_mask]
    if len(voiced_f0) > 0:
        log_f0 = np.log2(voiced_f0.clip(min=1.0))
        voiced_mean = float(log_f0.mean())
        voiced_std = float(log_f0.std()) if log_f0.std() > 0 else 1.0
    else:
        voiced_mean = 0.0
        voiced_std = 1.0
    
    return {
        'f0': f0,
        'uv': uv,
        'voiced_mean': voiced_mean,
        'voiced_std': voiced_std,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract F0 features")
    parser.add_argument('--input_dir', type=str, required=True, help='Input audio directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output F0 directory')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate')
    parser.add_argument('--hop_length', type=int, default=320, help='Hop length')
    parser.add_argument('--f0_min', type=float, default=50.0, help='Minimum F0')
    parser.add_argument('--f0_max', type=float, default=600.0, help='Maximum F0')
    parser.add_argument('--method', type=str, default='librosa', choices=['librosa', 'parselmouth'],
                       help='F0 extraction method')
    parser.add_argument('--ext', type=str, default='flac', help='Audio file extension')
    args = parser.parse_args()
    
    # 检查可用性
    if args.method == 'librosa' and not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required. Install with: pip install librosa")
    if args.method == 'parselmouth' and not PARSELMOUTH_AVAILABLE:
        raise ImportError("parselmouth is required. Install with: pip install praat-parselmouth")
    
    # 选择提取函数
    extract_fn = extract_f0_librosa if args.method == 'librosa' else extract_f0_parselmouth
    
    # 创建输出目录
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有音频文件
    audio_files = list(input_dir.glob(f'*.{args.ext}'))
    if not audio_files:
        audio_files = list(input_dir.rglob(f'*.{args.ext}'))
    
    print(f"Found {len(audio_files)} audio files")
    print(f"Using {args.method} for F0 extraction")
    
    # 提取F0
    for audio_path in tqdm(audio_files, desc="Extracting F0"):
        try:
            # 提取F0
            result = extract_fn(
                str(audio_path),
                sr=args.sample_rate,
                hop_length=args.hop_length,
                f0_min=args.f0_min,
                f0_max=args.f0_max
            )
            
            # 保存
            output_path = output_dir / audio_path.with_suffix('.npy').name
            np.save(output_path, result, allow_pickle=True)
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue
    
    print(f"Done! F0 features saved to {output_dir}")


if __name__ == '__main__':
    main()
