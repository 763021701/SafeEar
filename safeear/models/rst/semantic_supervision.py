# Copyright (c) 2024 SafeEar Authors
# 语义监督模块 - 支持HuBERT和Wav2Vec CTC两种方式

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any


class Wav2VecCTCExtractor(nn.Module):
    """
    使用Wav2Vec2-CTC模型提取语义特征（phone IDs）
    参考FACodec实现：使用 facebook/wav2vec2-xlsr-53-espeak-cv-ft
    """
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
        device: str = "cuda",
        freeze: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.device_str = device
        self.freeze = freeze
        self.num_classes = 5003  # wav2vec2-xlsr-53-espeak-cv-ft 的词汇表大小
        
        # 延迟加载模型
        self._processor = None
        self._model = None
        self._loaded = False
        
    def _load_model(self):
        """延迟加载模型，避免在初始化时占用显存"""
        if self._loaded:
            return
            
        try:
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2ForCTC
            
            # 新版本transformers需要分别加载feature_extractor和tokenizer
            try:
                # 方式1：直接加载Processor（适用于较新版本）
                self._processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            except:
                # 方式2：分别加载feature_extractor和tokenizer（兼容更多版本）
                feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
                tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(self.model_name)
                self._processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            
            self._model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
            
            if self.freeze:
                self._model.eval()
                for param in self._model.parameters():
                    param.requires_grad = False
                    
            self._loaded = True
            print(f"[Wav2VecCTCExtractor] Loaded {self.model_name}")
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Wav2Vec2 model: {e}\n"
                "Please install transformers: pip install transformers"
            )
    
    def to(self, device):
        """移动模型到指定设备"""
        super().to(device)
        if self._model is not None:
            self._model = self._model.to(device)
        return self
        
    @torch.no_grad()
    def extract_phone_ids(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        target_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        从音频中提取phone IDs
        
        Args:
            waveform: 输入波形 (B, T) 或 (B, 1, T)
            sample_rate: 采样率，默认16000
            target_length: 目标序列长度（用于对齐），如果为None则不调整
            
        Returns:
            phone_ids: (B, T') 预测的phone ID序列
        """
        self._load_model()
        device = next(self._model.parameters()).device
        
        # 确保输入格式正确
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)  # (B, 1, T) -> (B, T)
        
        # 如果不是16kHz，需要重采样
        if sample_rate != 16000:
            import torchaudio
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=16000
            )
        
        # 处理输入
        # wav2vec2 processor 期望 numpy array 或 list
        waveform_np = waveform.cpu().numpy()
        inputs = self._processor(
            waveform_np,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(device)
        
        # 获取logits
        outputs = self._model(input_values)
        logits = outputs.logits  # (B, T', vocab_size)
        
        # 获取预测的phone IDs
        predicted_ids = torch.argmax(logits, dim=-1)  # (B, T')
        
        # 如果需要调整到目标长度
        if target_length is not None:
            # 使用最近邻插值
            predicted_ids = F.interpolate(
                predicted_ids.unsqueeze(1).float(),
                size=target_length,
                mode="nearest"
            ).squeeze(1).long()
        
        # 确保返回的tensor在正确的设备上
        return predicted_ids.to(device)
    
    @torch.no_grad()
    def extract_features(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        target_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从音频中提取phone IDs和logits
        
        Args:
            waveform: 输入波形 (B, T) 或 (B, 1, T)
            sample_rate: 采样率
            target_length: 目标序列长度
            
        Returns:
            phone_ids: (B, T') 预测的phone ID序列
            logits: (B, T', vocab_size) 原始logits（用于软标签）
        """
        self._load_model()
        device = next(self._model.parameters()).device
        
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        
        if sample_rate != 16000:
            import torchaudio
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=16000
            )
        
        waveform_np = waveform.cpu().numpy()
        inputs = self._processor(
            waveform_np,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(device)
        
        outputs = self._model(input_values)
        logits = outputs.logits
        predicted_ids = torch.argmax(logits, dim=-1)
        
        if target_length is not None:
            predicted_ids = F.interpolate(
                predicted_ids.unsqueeze(1).float(),
                size=target_length,
                mode="nearest"
            ).squeeze(1).long()
            
            logits = F.interpolate(
                logits.transpose(1, 2),  # (B, vocab, T')
                size=target_length,
                mode="nearest"
            ).transpose(1, 2)  # (B, T', vocab)
        
        return predicted_ids, logits


class SemanticPredictor(nn.Module):
    """
    语义预测器 - 支持两种模式：
    1. HuBERT模式：预测连续特征向量（用MSE或Cosine Loss）
    2. Wav2Vec CTC模式：预测离散phone ID（用CrossEntropy或Focal Loss）
    """
    
    def __init__(
        self,
        input_dim: int,
        mode: str = "wav2vec_ctc",  # "hubert" 或 "wav2vec_ctc"
        hubert_dim: int = 768,  # HuBERT特征维度
        num_phone_classes: int = 5003,  # Wav2Vec CTC的词汇表大小
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.mode = mode
        self.input_dim = input_dim
        self.hubert_dim = hubert_dim
        self.num_phone_classes = num_phone_classes
        
        hidden_dim = hidden_dim or input_dim
        
        if mode == "hubert":
            # HuBERT模式：预测连续特征
            self.predictor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hubert_dim),
            )
            self.output_dim = hubert_dim
            
        elif mode == "wav2vec_ctc":
            # Wav2Vec CTC模式：预测离散phone ID
            # 参考FACodec的CNNLSTM设计
            self.predictor = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, num_phone_classes, kernel_size=1),
            )
            self.output_dim = num_phone_classes
            
        else:
            raise ValueError(f"Unknown semantic mode: {mode}. Use 'hubert' or 'wav2vec_ctc'")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 (B, C, T) 或 (B, T, C)
            
        Returns:
            HuBERT模式: (B, T, hubert_dim)
            Wav2Vec CTC模式: (B, T, num_phone_classes)
        """
        if self.mode == "hubert":
            # HuBERT模式期望 (B, T, C)
            if x.dim() == 3 and x.size(1) == self.input_dim:
                x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
            return self.predictor(x)  # (B, T, hubert_dim)
            
        else:  # wav2vec_ctc
            # CNN期望 (B, C, T)
            if x.dim() == 3 and x.size(-1) == self.input_dim:
                x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
            out = self.predictor(x)  # (B, num_classes, T)
            return out.transpose(1, 2)  # (B, T, num_classes)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    参考FACodec实现
    """
    
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: (B, T, C) or (B, C, T) - 预测logits
            target: (B, T) - 目标类别索引
        """
        # 确保input是 (B*T, C) 格式
        if input.dim() == 3:
            if input.size(-1) != target.size(-1):
                # input: (B, C, T), target: (B, T)
                input = input.transpose(1, 2)  # -> (B, T, C)
            B, T, C = input.shape
            input = input.reshape(-1, C)  # (B*T, C)
            target = target.reshape(-1)  # (B*T,)
        
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        return focal_loss.mean()


class SemanticLoss(nn.Module):
    """
    统一的语义损失计算模块
    支持HuBERT（连续特征）和Wav2Vec CTC（离散ID）两种模式
    """
    
    def __init__(
        self,
        mode: str = "wav2vec_ctc",
        hubert_loss_type: str = "cosine",  # "cosine" 或 "mse"
        ctc_loss_type: str = "focal",  # "focal" 或 "ce"
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.mode = mode
        
        if mode == "hubert":
            if hubert_loss_type == "cosine":
                self.loss_fn = lambda pred, target: (
                    1 - F.cosine_similarity(pred, target, dim=-1).mean()
                )
            else:  # mse
                self.loss_fn = nn.MSELoss()
                
        else:  # wav2vec_ctc
            if ctc_loss_type == "focal":
                self.loss_fn = FocalLoss(gamma=focal_gamma)
            else:  # ce
                self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测值
                - HuBERT模式: (B, T, hubert_dim)
                - Wav2Vec CTC模式: (B, T, num_classes)
            target: 目标值
                - HuBERT模式: (B, T, hubert_dim)
                - Wav2Vec CTC模式: (B, T) - phone IDs
        """
        if self.mode == "hubert":
            return self.loss_fn(pred, target)
        else:  # wav2vec_ctc
            # pred: (B, T, C) -> (B*T, C)
            # target: (B, T) -> (B*T,)
            B, T, C = pred.shape
            pred_flat = pred.reshape(-1, C)
            target_flat = target.reshape(-1)
            return self.loss_fn(pred_flat, target_flat)
