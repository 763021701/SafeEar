
"""
ASVSpoof2019 数据集 - RST版本
扩展支持：
1. F0提取和归一化
2. 说话人ID提取
3. UV (voiced/unvoiced) mask
4. 语义监督支持：HuBERT（预计算特征）或 Wav2Vec CTC（可在线提取）
"""

import glob
import random
import os
import torch
import torchaudio
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Union


def get_path_iterator(tsv):
    """
    Get the root path and list of file lines from the TSV file.
    """
    with open(tsv, "r") as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
    return root, lines


def load_feature(feat_path):
    """
    Load feature from the specified path.
    """
    feat = np.load(feat_path, mmap_mode="r")
    return feat


class F0Extractor:
    """
    F0提取器
    使用简化的方法从音频中提取F0（如果没有预计算的F0）
    
    支持两种模式：
    1. 使用预计算的F0文件
    2. 在线提取F0（使用pyin或其他方法）
    """
    def __init__(
        self, 
        sample_rate: int = 16000,
        hop_length: int = 320,
        f0_min: float = 50.0,
        f0_max: float = 600.0,
        use_precomputed: bool = True,
        precomputed_dir: Optional[str] = None
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.use_precomputed = use_precomputed
        self.precomputed_dir = Path(precomputed_dir) if precomputed_dir else None
        
        # 尝试导入可选的F0提取库
        self._pyin_available = False
        try:
            import librosa
            self._pyin_available = True
        except ImportError:
            pass
            
    def extract(self, audio: torch.Tensor, audio_path: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取F0和UV mask
        
        Args:
            audio: (1, T) 音频波形
            audio_path: 可选的音频路径（用于加载预计算F0）
            
        Returns:
            f0: (T',) 归一化的F0序列
            uv: (T',) voiced/unvoiced mask (1=voiced, 0=unvoiced)
        """
        # 尝试加载预计算的F0
        if self.use_precomputed and self.precomputed_dir is not None and audio_path is not None:
            f0_path = self.precomputed_dir / Path(audio_path).with_suffix('.npy').name
            if f0_path.exists():
                data = np.load(f0_path, allow_pickle=True).item()
                f0 = torch.tensor(data['f0'], dtype=torch.float32)
                uv = torch.tensor(data['uv'], dtype=torch.float32)
                return f0, uv
        
        # 在线提取F0
        return self._extract_f0_online(audio)
    
    def _extract_f0_online(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        在线提取F0（使用简化方法或pyin）
        """
        audio_np = audio.squeeze().numpy()
        
        if self._pyin_available:
            import librosa
            # 使用pyin提取F0
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_np,
                fmin=self.f0_min,
                fmax=self.f0_max,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            f0 = np.nan_to_num(f0, nan=0.0)
            uv = voiced_flag.astype(np.float32)
        else:
            # 如果没有librosa，返回零序列（需要预计算）
            num_frames = audio.shape[-1] // self.hop_length
            f0 = np.zeros(num_frames, dtype=np.float32)
            uv = np.zeros(num_frames, dtype=np.float32)
            
        return torch.tensor(f0, dtype=torch.float32), torch.tensor(uv, dtype=torch.float32)


def compute_encoder_output_length(audio_length: int, strides: List[int] = [8, 5, 4, 2]) -> int:
    """
    计算encoder输出的特征序列长度
    
    Args:
        audio_length: 输入音频长度（采样点数）
        strides: encoder各层的stride
        
    Returns:
        output_length: encoder输出特征序列长度
    """
    length = audio_length
    for stride in strides:
        # 模拟Conv1d with padding='same'
        # PyTorch Conv1d计算: out_length = floor((in_length + 2*padding - kernel_size) / stride) + 1
        # 对于stride=s, 近似为: out_length ≈ ceil(in_length / s)
        length = (length + stride - 1) // stride  # ceil division
    return length


def normalize_f0(f0: torch.Tensor, uv: Optional[torch.Tensor] = None, unvoiced_value: float = -10.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    归一化F0序列
    
    参考FACodec的实现：
    1. 找到voiced帧 (F0 > 5.0)
    2. 转换为log2尺度
    3. 计算mean和std进行z-score归一化
    4. unvoiced帧设为unvoiced_value (-10)
    
    Args:
        f0: (T,) 原始F0序列
        uv: (T,) 可选的voiced/unvoiced mask
        unvoiced_value: unvoiced帧的填充值
        
    Returns:
        normalized_f0: (T,) 归一化的F0
        uv_mask: (T,) voiced/unvoiced mask
    """
    # 确定voiced帧
    if uv is not None:
        voiced_mask = uv > 0.5
    else:
        voiced_mask = f0 > 5.0
        
    # 创建输出tensor
    normalized_f0 = torch.zeros_like(f0)
    uv_mask = voiced_mask.float()
    
    # 获取voiced帧的F0
    f0_voiced = f0[voiced_mask]
    
    if len(f0_voiced) > 0:
        # 转换为log尺度
        log_f0 = torch.log2(f0_voiced.clamp(min=1.0))
        
        # Z-score归一化
        mean_f0 = log_f0.mean()
        std_f0 = log_f0.std()
        
        if std_f0 > 0:
            normalized_voiced = (log_f0 - mean_f0) / std_f0
        else:
            normalized_voiced = log_f0 - mean_f0
            
        # 填充normalized_f0
        normalized_f0[voiced_mask] = normalized_voiced
        normalized_f0[~voiced_mask] = unvoiced_value
    else:
        # 如果没有voiced帧，全部设为unvoiced_value
        normalized_f0.fill_(unvoiced_value)
        
    # 处理nan和inf
    normalized_f0 = torch.nan_to_num(normalized_f0, nan=unvoiced_value, posinf=unvoiced_value, neginf=unvoiced_value)
    
    return normalized_f0, uv_mask


class ASVSpoof2019RST(Dataset):
    """
    ASVSpoof2019数据集 - RST版本
    
    扩展功能：
    1. 提取说话人ID
    2. 提取/加载F0特征
    3. 生成UV mask
    4. 语义监督：支持HuBERT（预计算）和Wav2Vec CTC（可在线提取）
    """
    def __init__(
        self, 
        tsv_path: str, 
        protocol_path: str, 
        feat_dir: Optional[str] = None,  # HuBERT特征目录（hubert模式必须）
        max_len: int = 64600, 
        is_train: bool = True,
        f0_dir: Optional[str] = None,
        extract_f0_online: bool = False,
        sample_rate: int = 16000,
        hop_length: int = 320,
        # 语义监督配置
        semantic_mode: str = "wav2vec_ctc",  # "hubert" 或 "wav2vec_ctc"
        # Encoder下采样配置（用于特征对齐）
        encoder_strides: List[int] = [8, 5, 4, 2],
        align_features: bool = True,  # 是否对齐所有特征到encoder输出长度
    ):
        """
        初始化数据集
        
        Args:
            tsv_path: TSV文件路径
            protocol_path: 协议文件路径
            feat_dir: HuBERT特征目录（semantic_mode='hubert'时必须）
            max_len: 最大音频长度（采样点）
            is_train: 是否为训练集
            f0_dir: F0预计算文件目录（可选）
            extract_f0_online: 是否在线提取F0
            sample_rate: 采样率
            hop_length: 帧移（用于F0提取）
            semantic_mode: 语义监督模式 ("hubert" 或 "wav2vec_ctc")
            encoder_strides: Encoder各层stride（用于计算特征长度）
            align_features: 是否对齐所有特征到encoder输出长度
        """
        super().__init__()
        root, self.lines = get_path_iterator(tsv_path)
        self.feat_dir = Path(feat_dir) if feat_dir else None
        self.f0_dir = Path(f0_dir) if f0_dir else None
        _, self.sr = torchaudio.load(root + "/" + self.lines[0].split('\t')[0])
        self.max_len = max_len 
        self.is_train = is_train
        self.root = Path(root)
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.semantic_mode = semantic_mode
        self.encoder_strides = encoder_strides
        self.align_features = align_features
        
        # 验证semantic_mode配置
        if semantic_mode == "hubert" and feat_dir is None:
            raise ValueError("feat_dir must be specified when semantic_mode='hubert'")
        
        print(f"[ASVSpoof2019RST] Semantic mode: {semantic_mode}")
        
        # 解析协议文件
        with open(protocol_path) as file:
            meta_infos = file.readlines()
        self.meta_infos = meta_infos
        
        # 构建映射表
        # 格式: "LA_0079 LA_T_1138215 - - bonafide"
        # 第1列: 说话人ID, 第2列: 文件名, 第5列: 标签
        self.file_to_label = {}  # 文件名 -> 标签
        self.file_to_speaker = {}  # 文件名 -> 说话人ID
        self.speaker_to_id = {}  # 说话人名 -> 数字ID
        
        speaker_set = set()
        for meta_info in meta_infos:
            parts = meta_info.strip().split(' ')
            if len(parts) >= 5:
                speaker_name = parts[0]  # LA_0079
                file_name = parts[1]      # LA_T_1138215
                label = parts[-1]         # bonafide/spoof
                
                self.file_to_label[file_name] = label
                self.file_to_speaker[file_name] = speaker_name
                speaker_set.add(speaker_name)
        
        # 创建说话人ID映射
        for idx, speaker in enumerate(sorted(speaker_set)):
            self.speaker_to_id[speaker] = idx
            
        self.num_speakers = len(self.speaker_to_id)
        print(f"[ASVSpoof2019RST] Found {self.num_speakers} speakers")
        
        # F0提取器
        self.extract_f0_online = extract_f0_online
        if extract_f0_online:
            self.f0_extractor = F0Extractor(
                sample_rate=sample_rate,
                hop_length=hop_length,
                use_precomputed=f0_dir is not None,
                precomputed_dir=f0_dir
            )

    def __len__(self):
        return len(self.lines)
    
    def _get_speaker_id(self, file_name: str) -> int:
        """获取说话人数字ID"""
        # file_name格式可能是 "LA_T_1138215.flac" 或 "LA_T_1138215"
        base_name = file_name.split('.')[0]
        speaker_name = self.file_to_speaker.get(base_name, None)
        if speaker_name is None:
            return 0  # 默认说话人
        return self.speaker_to_id.get(speaker_name, 0)
    
    def _get_label(self, file_name: str) -> int:
        """获取标签 (0=bonafide/real, 1=spoof/fake)"""
        base_name = file_name.split('.')[0]
        label_str = self.file_to_label.get(base_name, 'bonafide')
        return 1 if label_str == 'spoof' else 0
    
    def _load_f0(self, audio_path: Path, audio: torch.Tensor, num_frames: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        加载或提取F0
        
        Returns:
            f0: (T,) 归一化的F0
            uv: (T,) voiced/unvoiced mask
        """
        # 尝试从预计算文件加载
        if self.f0_dir is not None:
            f0_path = self.f0_dir / audio_path.with_suffix('.npy').name
            if f0_path.exists():
                try:
                    data = np.load(f0_path, allow_pickle=True).item()
                    f0 = torch.tensor(data['f0'], dtype=torch.float32)
                    uv = torch.tensor(data['uv'], dtype=torch.float32)
                    # 归一化
                    f0, uv = normalize_f0(f0, uv)
                    return f0, uv
                except Exception:
                    pass
        
        # 在线提取
        if self.extract_f0_online:
            raw_f0, raw_uv = self.f0_extractor.extract(audio, str(audio_path))
            f0, uv = normalize_f0(raw_f0, raw_uv)
            return f0, uv
        
        # 如果没有F0，返回默认值
        f0 = torch.zeros(num_frames) - 10.0  # unvoiced value
        uv = torch.zeros(num_frames)
        return f0, uv

    def __getitem__(self, index):
        """
        获取数据项
        
        Returns:
            训练时: (audio, semantic_feat, f0, uv, speaker_id, target)
            测试时: (audio, semantic_feat, f0, uv, speaker_id, target, audio_path)
            
        其中 semantic_feat:
            - HuBERT模式: (T, 768) 预计算的HuBERT特征
            - Wav2Vec CTC模式: None (phone IDs将在训练时在线提取)
        """
        feat_duration = self.max_len // self.hop_length
    
        relative_path = Path(self.lines[index].split('\t')[0])
        audio_path = self.root / relative_path
        
        # 加载音频
        audio = torchaudio.load(str(audio_path))[0]
        
        # 根据语义模式加载特征
        if self.semantic_mode == "hubert":
            # HuBERT模式：加载预计算特征
            feat_path = self.feat_dir / relative_path
            avg_hubert_feat = torch.tensor(load_feature(feat_path.with_suffix(".npy")))
        else:
            # Wav2Vec CTC模式：不需要预计算特征，返回空占位符
            # phone IDs将在训练时在线提取
            avg_hubert_feat = torch.zeros(feat_duration, 1)  # 占位符
        
        # 获取标签和说话人ID
        file_name = self.lines[index].split('.')[0]
        target = self._get_label(file_name)
        speaker_id = self._get_speaker_id(file_name)
        
        # 处理HuBERT特征维度 (仅在hubert模式下)
        if self.semantic_mode == "hubert" and avg_hubert_feat.ndim == 3:
            avg_hubert_feat = avg_hubert_feat.permute(2, 1, 0).squeeze(1)
        elif self.semantic_mode == "hubert" and avg_hubert_feat.ndim == 2:
            avg_hubert_feat = avg_hubert_feat.permute(1, 0)
        
        # 加载/提取F0
        f0, uv = self._load_f0(audio_path, audio, feat_duration)
        
        # 数据裁剪/填充
        if self.is_train and audio.shape[1] > self.max_len:
            st = random.randint(0, audio.shape[1] - self.max_len - 1)
            feat_st = st // self.hop_length
            ed = st + self.max_len
            
            audio = audio[:, st:ed]
            
            # 裁剪语义特征 (仅HuBERT模式)
            if self.semantic_mode == "hubert":
                if avg_hubert_feat[:, feat_st:feat_st + feat_duration].shape[1] < feat_duration:
                    avg_hubert_feat = avg_hubert_feat[:, feat_st:feat_st + feat_duration]
                    avg_hubert_feat = F.pad(avg_hubert_feat, (0, feat_duration - avg_hubert_feat.shape[1]), "constant", 0)
                else:
                    avg_hubert_feat = avg_hubert_feat[:, feat_st:feat_st + feat_duration]
            
            # 裁剪F0和UV
            if f0.shape[0] > feat_st + feat_duration:
                f0 = f0[feat_st:feat_st + feat_duration]
                uv = uv[feat_st:feat_st + feat_duration]
            else:
                f0 = f0[feat_st:]
                uv = uv[feat_st:]
                
        elif not self.is_train and audio.shape[1] > self.max_len:
            st = 0
            feat_st = 0
            ed = st + self.max_len
            
            audio = audio[:, st:ed]
            
            # 裁剪语义特征 (仅HuBERT模式)
            if self.semantic_mode == "hubert":
                if avg_hubert_feat[:, feat_st:feat_st + feat_duration].shape[1] < feat_duration:
                    avg_hubert_feat = avg_hubert_feat[:, feat_st:feat_st + feat_duration]
                    avg_hubert_feat = F.pad(avg_hubert_feat, (0, feat_duration - avg_hubert_feat.shape[1]), "constant", 0)
                else:
                    avg_hubert_feat = avg_hubert_feat[:, feat_st:feat_st + feat_duration]
            
            if f0.shape[0] > feat_duration:
                f0 = f0[:feat_duration]
                uv = uv[:feat_duration]

        # 填充短音频
        if audio.shape[1] < self.max_len:
            audio_pad_length = self.max_len - audio.shape[1]
            audio = F.pad(audio, (0, audio_pad_length), "constant", 0)
        
        # 填充语义特征 (仅HuBERT模式)
        if self.semantic_mode == "hubert" and avg_hubert_feat.shape[1] < feat_duration:
            avg_hubert_feat = F.pad(avg_hubert_feat, (0, feat_duration - avg_hubert_feat.shape[1]), "constant", 0)
        
        # 填充F0和UV
        if f0.shape[0] < feat_duration:
            pad_len = feat_duration - f0.shape[0]
            f0 = F.pad(f0, (0, pad_len), "constant", -10.0)
            uv = F.pad(uv, (0, pad_len), "constant", 0.0)

        # ========== 统一特征对齐（可选） ==========
        if self.align_features:
            # 计算encoder输出长度
            target_len = compute_encoder_output_length(
                audio.shape[1], 
                self.encoder_strides
            )
            
            # 对齐F0和UV到encoder输出长度
            if f0.shape[0] != target_len:
                # F0: 使用线性插值（连续值）
                f0 = F.interpolate(
                    f0.unsqueeze(0).unsqueeze(0),  # (1, 1, T)
                    size=target_len,
                    mode='linear',
                    align_corners=False
                ).squeeze()  # (target_len,)
                
                # UV: 使用最近邻插值（离散值）
                uv = F.interpolate(
                    uv.unsqueeze(0).unsqueeze(0),  # (1, 1, T)
                    size=target_len,
                    mode='nearest'
                ).squeeze()  # (target_len,)
            
            # 对齐语义特征（仅HuBERT模式）
            if self.semantic_mode == "hubert" and avg_hubert_feat.shape[1] != target_len:
                # avg_hubert_feat: (D, T) -> 插值到 (D, target_len)
                avg_hubert_feat = F.interpolate(
                    avg_hubert_feat.unsqueeze(0),  # (1, D, T)
                    size=target_len,
                    mode='linear',
                    align_corners=False
                ).squeeze(0)  # (D, target_len)
            elif self.semantic_mode == "wav2vec_ctc":
                # Wav2Vec CTC模式：更新占位符长度
                avg_hubert_feat = torch.zeros(target_len, 1)

        if self.is_train:
            return audio, avg_hubert_feat, f0, uv, speaker_id, target
        else:
            return audio, avg_hubert_feat, f0, uv, speaker_id, target, str(audio_path)


def pad_sequence(batch):
    """Pad a sequence of tensors to have the same length."""
    batch = [item.permute(1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    return batch.permute(0, 2, 1)


def collate_fn_rst(batch):
    """
    RST专用的collate函数
    
    Args:
        batch: list of tuples (audio, hubert_feat, f0, uv, speaker_id, target)
        
    Returns:
        tuple: (wavs, feats, f0s, uvs, speaker_ids, targets)
    """
    wavs = []
    feats = []
    f0s = []
    uvs = []
    speaker_ids = []
    targets = []
    
    for item in batch:
        if len(item) == 6:
            wav, feat, f0, uv, spk_id, target = item
        else:
            wav, feat, f0, uv, spk_id, target, _ = item
            
        wavs.append(wav)
        feats.append(feat)
        f0s.append(f0)
        uvs.append(uv)
        speaker_ids.append(spk_id)
        targets.append(target)

    wavs = pad_sequence(wavs)
    feats = pad_sequence(feats).permute(0, 2, 1)  # (B, T, D)
    
    # F0和UV可能长度不一致，需要padding
    max_f0_len = max(f0.shape[0] for f0 in f0s)
    f0s_padded = torch.stack([
        F.pad(f0, (0, max_f0_len - f0.shape[0]), value=-10.0) for f0 in f0s
    ])
    uvs_padded = torch.stack([
        F.pad(uv, (0, max_f0_len - uv.shape[0]), value=0.0) for uv in uvs
    ])
    
    return (
        wavs,
        feats, 
        f0s_padded,
        uvs_padded,
        torch.tensor(speaker_ids, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long)
    )


def collate_fn_rst_test(batch):
    """
    RST测试用的collate函数（包含audio_path）
    """
    wavs = []
    feats = []
    f0s = []
    uvs = []
    speaker_ids = []
    targets = []
    audio_paths = []
    
    for item in batch:
        wav, feat, f0, uv, spk_id, target, audio_path = item
        wavs.append(wav)
        feats.append(feat)
        f0s.append(f0)
        uvs.append(uv)
        speaker_ids.append(spk_id)
        targets.append(target)
        audio_paths.append(audio_path)

    wavs = pad_sequence(wavs)
    feats = pad_sequence(feats).permute(0, 2, 1)
    
    max_f0_len = max(f0.shape[0] for f0 in f0s)
    f0s_padded = torch.stack([
        F.pad(f0, (0, max_f0_len - f0.shape[0]), value=-10.0) for f0 in f0s
    ])
    uvs_padded = torch.stack([
        F.pad(uv, (0, max_f0_len - uv.shape[0]), value=0.0) for uv in uvs
    ])
    
    return (
        wavs,
        feats,
        f0s_padded,
        uvs_padded,
        torch.tensor(speaker_ids, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        audio_paths
    )


class RSTDataClass:
    """RST数据集工厂类"""
    def __init__(
        self,
        train_path: List[str], 
        val_path: List[str], 
        test_path: List[str], 
        max_len: int = 64600,
        f0_dir: Optional[str] = None,
        extract_f0_online: bool = False,
        semantic_mode: str = "wav2vec_ctc",  # 语义监督模式
        encoder_strides: List[int] = [8, 5, 4, 2],  # Encoder strides
        align_features: bool = True,  # 是否对齐特征
    ):
        super().__init__()

        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.max_len = max_len
        self.f0_dir = f0_dir
        self.extract_f0_online = extract_f0_online
        self.semantic_mode = semantic_mode
        self.encoder_strides = encoder_strides
        self.align_features = align_features

        # 创建数据集
        self.train = ASVSpoof2019RST(
            self.train_path[0], 
            self.train_path[1], 
            self.train_path[2], 
            self.max_len, 
            is_train=True,
            f0_dir=f0_dir,
            extract_f0_online=extract_f0_online,
            semantic_mode=semantic_mode,
            encoder_strides=encoder_strides,
            align_features=align_features,
        )
        self.val = ASVSpoof2019RST(
            self.val_path[0], 
            self.val_path[1], 
            self.val_path[2], 
            self.max_len, 
            is_train=True,
            f0_dir=f0_dir,
            extract_f0_online=extract_f0_online,
            semantic_mode=semantic_mode,
            encoder_strides=encoder_strides,
            align_features=align_features,
        )
        self.test = ASVSpoof2019RST(
            self.test_path[0], 
            self.test_path[1], 
            self.test_path[2],
            self.max_len,
            is_train=False,
            f0_dir=f0_dir,
            extract_f0_online=extract_f0_online,
            semantic_mode=semantic_mode,
            encoder_strides=encoder_strides,
            align_features=align_features,
        )
        
        # 获取说话人数量
        self.num_speakers = self.train.num_speakers
        
    def __call__(self, mode: str) -> ASVSpoof2019RST:
        if mode == "train":
            return self.train
        elif mode == "val":
            return self.val
        elif mode == "test":
            return self.test
        else:
            raise ValueError(f"Unknown mode: {mode}.")


class RSTDataModule(LightningDataModule):
    """RST数据模块"""
    def __init__(
        self, 
        DataClass_dict, # Hydra会自动实例化为RSTDataClass对象
        batch_size: int, 
        num_workers: int, 
        pin_memory: bool
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        # DataClass_dict已经被Hydra实例化为RSTDataClass对象
        if isinstance(DataClass_dict, RSTDataClass):
            self.dataset_select = DataClass_dict
        elif isinstance(DataClass_dict, dict):
            # 如果还是字典（向后兼容）
            DataClass_dict.pop("_target_", None)
            self.dataset_select = RSTDataClass(**DataClass_dict)
        else:
            raise TypeError(f"DataClass_dict should be RSTDataClass or dict, got {type(DataClass_dict)}")

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None
        
        # 暴露说话人数量供模型使用
        self.num_speakers = self.dataset_select.num_speakers

    def setup(self, stage=None):
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = self.dataset_select("train")
            self.data_val = self.dataset_select("val")
            self.data_test = self.dataset_select("test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=collate_fn_rst
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn_rst
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn_rst_test
        )
