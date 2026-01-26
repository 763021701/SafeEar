"""
残差剥离塔(RST)损失函数

包括：
1. 语义蒸馏损失 - HuBERT特征对齐 或 Wav2Vec CTC phone预测
2. 说话人识别损失 - 说话人分类/嵌入对齐
3. 韵律预测损失 - F0和UV预测
4. 深度伪造检测损失 - 二分类
5. 重建损失 (可选) - 音频重建
6. 承诺损失 - VQ承诺损失

语义监督模式：
- hubert: 使用HuBERT特征蒸馏（余弦相似度损失）
- wav2vec_ctc: 使用Wav2Vec CTC phone预测（交叉熵/Focal损失）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from torchaudio.transforms import MelSpectrogram


class SemanticDistillationLoss(nn.Module):
    """
    HuBERT语义蒸馏损失
    使用余弦相似度损失将VQ输出与HuBERT特征对齐
    
    参考SafeEar论文公式(4):
    L_distill = (1/T) * Σ log(σ(cos(W·S_t, H_t)))
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self, 
        pred_semantic: torch.Tensor,  # (B, T, D)
        target_hubert: torch.Tensor   # (B, T, D)
    ) -> torch.Tensor:
        """
        Args:
            pred_semantic: 预测的语义特征
            target_hubert: HuBERT目标特征
        """
        # 归一化
        pred_norm = F.normalize(pred_semantic, p=2, dim=-1)
        target_norm = F.normalize(target_hubert, p=2, dim=-1)
        
        # 余弦相似度
        cosine_sim = (pred_norm * target_norm).sum(dim=-1)  # (B, T)
        
        # 对数sigmoid损失
        loss = -torch.log(torch.sigmoid(cosine_sim / self.temperature) + 1e-8)
        
        return loss.mean()


class Wav2VecCTCLoss(nn.Module):
    """
    Wav2Vec CTC语义损失
    使用交叉熵或Focal损失预测phone IDs
    
    参考FACodec实现
    """
    def __init__(self, use_focal: bool = True, focal_gamma: float = 2.0):
        super().__init__()
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        
    def forward(
        self, 
        pred_logits: torch.Tensor,  # (B, T_pred, num_classes)
        target_ids: torch.Tensor    # (B, T_target) phone IDs
    ) -> torch.Tensor:
        """
        Args:
            pred_logits: 预测的phone logits
            target_ids: 目标phone IDs
        """
        # 对齐序列长度：将target插值到pred长度
        if pred_logits.shape[1] != target_ids.shape[1]:
            pred_len = pred_logits.shape[1]
            # 使用最近邻插值对齐phone IDs（离散值）
            target_ids = F.interpolate(
                target_ids.unsqueeze(1).float(),  # (B, 1, T_target)
                size=pred_len,
                mode='nearest'
            ).squeeze(1).long()  # (B, T_pred)
        
        # Flatten
        B, T, C = pred_logits.shape
        pred_flat = pred_logits.reshape(-1, C)  # (B*T, C)
        target_flat = target_ids.reshape(-1)    # (B*T,)
        
        if self.use_focal:
            # Focal Loss
            ce_loss = F.cross_entropy(pred_flat, target_flat, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.focal_gamma) * ce_loss
            return focal_loss.mean()
        else:
            return F.cross_entropy(pred_flat, target_flat)


class SemanticLoss(nn.Module):
    """
    统一的语义损失
    根据模式自动选择HuBERT蒸馏或Wav2Vec CTC损失
    """
    def __init__(
        self, 
        mode: str = "wav2vec_ctc",  # "hubert" 或 "wav2vec_ctc"
        temperature: float = 1.0,
        use_focal: bool = True,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.mode = mode
        
        if mode == "hubert":
            self.loss_fn = SemanticDistillationLoss(temperature=temperature)
        else:  # wav2vec_ctc
            self.loss_fn = Wav2VecCTCLoss(use_focal=use_focal, focal_gamma=focal_gamma)
            
    def forward(
        self, 
        pred: torch.Tensor,   # HuBERT: (B, T, D), CTC: (B, T, num_classes)
        target: torch.Tensor  # HuBERT: (B, T, D), CTC: (B, T) phone IDs
    ) -> torch.Tensor:
        return self.loss_fn(pred, target)


class SpeakerLoss(nn.Module):
    """
    说话人识别损失
    支持分类模式和嵌入对齐模式
    """
    def __init__(self, use_classification: bool = True):
        super().__init__()
        self.use_classification = use_classification
        
    def forward(
        self,
        pred_speaker: torch.Tensor,  # 分类: (B, num_speakers), 嵌入: (B, D)
        target: torch.Tensor         # 分类: (B,) 标签, 嵌入: (B, D)
    ) -> torch.Tensor:
        # 自动检测target类型：如果是整数类型，使用分类模式
        is_classification = target.dtype in [torch.long, torch.int, torch.int32, torch.int64]
        
        if is_classification or self.use_classification:
            return F.cross_entropy(pred_speaker, target.long())
        else:
            # 嵌入对齐：余弦相似度损失
            pred_norm = F.normalize(pred_speaker, p=2, dim=-1)
            target_norm = F.normalize(target, p=2, dim=-1)
            cosine_sim = (pred_norm * target_norm).sum(dim=-1)
            return (1 - cosine_sim).mean()


class ProsodyLoss(nn.Module):
    """
    韵律预测损失
    包括F0预测和UV预测
    """
    def __init__(self, f0_weight: float = 1.0, uv_weight: float = 1.0):
        super().__init__()
        self.f0_weight = f0_weight
        self.uv_weight = uv_weight
        
    def forward(
        self,
        pred_prosody: Dict[str, torch.Tensor],  # {'f0': (B, T), 'uv': (B, T)}
        target_f0: torch.Tensor,                 # (B, T) normalized F0
        target_uv: torch.Tensor                  # (B, T) voiced/unvoiced mask
    ) -> torch.Tensor:
        pred_f0 = pred_prosody['f0']  # (B, T_pred)
        pred_uv = pred_prosody['uv']  # (B, T_pred)
        
        # 对齐时间维度：将预测插值到目标长度
        if pred_f0.shape[1] != target_f0.shape[1]:
            target_len = target_f0.shape[1]
            # 使用线性插值对齐F0
            pred_f0 = F.interpolate(
                pred_f0.unsqueeze(1),  # (B, 1, T_pred)
                size=target_len,
                mode='linear',
                align_corners=False
            ).squeeze(1)  # (B, T_target)
            
            # 使用最近邻插值对齐UV（离散值）
            pred_uv = F.interpolate(
                pred_uv.unsqueeze(1),  # (B, 1, T_pred)
                size=target_len,
                mode='nearest'
            ).squeeze(1)  # (B, T_target)
        
        # F0损失：只在voiced区域计算
        voiced_mask = target_uv > 0.5
        if voiced_mask.any():
            f0_loss = F.smooth_l1_loss(
                pred_f0[voiced_mask], 
                target_f0[voiced_mask]
            )
        else:
            f0_loss = torch.tensor(0.0, device=pred_f0.device)
            
        # UV损失：二分类
        uv_loss = F.binary_cross_entropy_with_logits(pred_uv, target_uv)
        
        return self.f0_weight * f0_loss + self.uv_weight * uv_loss


class DeepfakeDetectionLoss(nn.Module):
    """
    深度伪造检测损失
    支持标签平滑和焦点损失
    """
    def __init__(
        self, 
        label_smoothing: float = 0.0,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        
    def forward(
        self,
        logits: torch.Tensor,   # (B, 2)
        targets: torch.Tensor   # (B,) labels
    ) -> torch.Tensor:
        if targets.numel() > 0:
            tmin = int(targets.min().item())
            tmax = int(targets.max().item())
            if tmin < 0 or tmax >= logits.shape[-1]:
                raise ValueError(
                    f"[DeepfakeDetectionLoss] targets out of range: min={tmin}, max={tmax}, "
                    f"num_classes={logits.shape[-1]}. "
                    "这通常意味着batch解包错误（labels被speaker_id覆盖）或数据标签映射异常。"
                )
        if self.use_focal_loss:
            return self._focal_loss(logits, targets)
        else:
            return F.cross_entropy(
                logits, targets, 
                label_smoothing=self.label_smoothing
            )
            
    def _focal_loss(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # 获取正确类别的概率
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal权重
        focal_weight = (1 - p_t) ** self.focal_gamma
        
        return (focal_weight * ce_loss).mean()


class ReconstructionLoss(nn.Module):
    """
    音频重建损失
    包括时域L1和多尺度频谱损失
    
    参考SafeEar论文公式(5)
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        time_weight: float = 1.0,
        freq_weight: float = 1.0
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.time_weight = time_weight
        self.freq_weight = freq_weight
        
    def forward(
        self,
        pred_audio: torch.Tensor,   # (B, 1, T)
        target_audio: torch.Tensor  # (B, 1, T)
    ) -> torch.Tensor:
        # 时域L1损失
        time_loss = F.l1_loss(pred_audio, target_audio)
        
        # 多尺度频谱损失
        freq_loss = torch.tensor(0.0, device=pred_audio.device)
        
        for i in range(6, 12):  # 2^6 到 2^11
            n_fft = 2 ** i
            hop_length = n_fft // 4
            
            mel_transform = MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=64,
            ).to(pred_audio.device)
            
            pred_mel = mel_transform(pred_audio.squeeze(1))
            target_mel = mel_transform(target_audio.squeeze(1))
            
            # L1 + L2 损失
            freq_loss += F.l1_loss(pred_mel, target_mel)
            freq_loss += F.mse_loss(pred_mel, target_mel)
            
        freq_loss = freq_loss / 6  # 平均
        
        return self.time_weight * time_loss + self.freq_weight * freq_loss


class RSTLoss(nn.Module):
    """
    RST完整损失函数
    组合所有监督损失
    
    支持两种语义监督模式：
    - hubert: HuBERT特征蒸馏
    - wav2vec_ctc: Wav2Vec CTC phone预测
    """
    def __init__(
        self,
        # 损失权重
        semantic_weight: float = 1.0,
        speaker_weight: float = 1.0,
        prosody_weight: float = 1.0,
        detection_weight: float = 1.0,
        reconstruction_weight: float = 0.0,
        commit_weight: float = 0.25,
        
        # GR损失权重
        gr_semantic_weight: float = 0.5,
        gr_speaker_weight: float = 0.5,
        gr_prosody_weight: float = 0.5,
        
        # 语义监督配置
        semantic_mode: str = "wav2vec_ctc",  # "hubert" 或 "wav2vec_ctc"
        
        # 其他参数
        use_speaker_classification: bool = True,
        use_focal_loss: bool = False,
        label_smoothing: float = 0.0,
        sample_rate: int = 16000,
    ):
        super().__init__()
        
        self.semantic_mode = semantic_mode
        
        self.weights = {
            'semantic': semantic_weight,
            'speaker': speaker_weight,
            'prosody': prosody_weight,
            'detection': detection_weight,
            'reconstruction': reconstruction_weight,
            'commit': commit_weight,
            'gr_semantic': gr_semantic_weight,
            'gr_speaker': gr_speaker_weight,
            'gr_prosody': gr_prosody_weight,
        }
        
        # 各项损失
        # 语义损失：根据模式选择
        self.semantic_loss = SemanticLoss(
            mode=semantic_mode,
            use_focal=True,
            focal_gamma=2.0
        )
        self.speaker_loss = SpeakerLoss(use_classification=use_speaker_classification)
        self.prosody_loss = ProsodyLoss()
        self.detection_loss = DeepfakeDetectionLoss(
            label_smoothing=label_smoothing,
            use_focal_loss=use_focal_loss
        )
        self.reconstruction_loss = ReconstructionLoss(sample_rate=sample_rate)
        
    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        detection_only: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        计算所有损失
        
        Args:
            model_output: 模型输出dict
            targets: 目标dict，包含:
                - labels: (B,) 深度伪造标签
                - semantic_target: 
                    HuBERT模式: (B, T, D) HuBERT特征
                    Wav2Vec CTC模式: (B, T) phone IDs
                - speaker_ids/speaker_embeds: 说话人标签/嵌入
                - f0: (B, T) 归一化F0
                - uv: (B, T) voiced/unvoiced mask
                - audio: (B, 1, T) 原始音频 (可选，用于重建)
            detection_only: 是否只计算检测损失（验证/测试时使用）
                
        Returns:
            dict with all losses and total loss
        """
        losses = {}
        
        # 检测损失（始终计算）
        if 'logits' in model_output and 'labels' in targets:
            losses['detection'] = self.weights['detection'] * self.detection_loss(
                model_output['logits'],
                targets['labels']
            )
        
        # 如果只计算检测损失（验证/测试阶段），直接返回
        if detection_only:
            losses['total'] = losses.get('detection', torch.tensor(0.0))
            return losses
        
        # ========== 以下为训练阶段的辅助损失 ==========
        
        # 语义损失（HuBERT蒸馏或Wav2Vec CTC）
        rst_output = model_output.get('rst_output', {})
        pred_semantic = rst_output.get('pred_semantic', model_output.get('pred_semantic'))
        semantic_target = targets.get('semantic_target')
        
        if pred_semantic is not None and semantic_target is not None:
            losses['semantic'] = self.weights['semantic'] * self.semantic_loss(
                pred_semantic,
                semantic_target
            )
            
        # 说话人损失
        if 'pred_speaker' in model_output:
            speaker_target = targets.get('speaker_ids', targets.get('speaker_embeds'))
            if speaker_target is not None:
                losses['speaker'] = self.weights['speaker'] * self.speaker_loss(
                    model_output['pred_speaker'],
                    speaker_target
                )
                
        # 韵律损失
        if 'pred_prosody' in model_output and 'f0' in targets:
            losses['prosody'] = self.weights['prosody'] * self.prosody_loss(
                model_output['pred_prosody'],
                targets['f0'],
                targets['uv']
            )
            
        # 重建损失 (可选)
        if 'reconstructed' in model_output and 'audio' in targets:
            losses['reconstruction'] = self.weights['reconstruction'] * self.reconstruction_loss(
                model_output['reconstructed'],
                targets['audio']
            )
            
        # 承诺损失
        if 'rst_output' in model_output and 'commit_losses' in model_output['rst_output']:
            commit_loss = sum(model_output['rst_output']['commit_losses'])
            losses['commit'] = self.weights['commit'] * commit_loss
            
        # 梯度反转损失 (对抗损失，确保残差不含非判伪信息)
        if 'gr_pred_semantic' in rst_output:
            if semantic_target is not None:
                losses['gr_semantic'] = self.weights['gr_semantic'] * self.semantic_loss(
                    rst_output['gr_pred_semantic'],
                    semantic_target
                )
                
            speaker_target = targets.get('speaker_ids', targets.get('speaker_embeds'))
            if 'gr_pred_speaker' in rst_output and speaker_target is not None:
                losses['gr_speaker'] = self.weights['gr_speaker'] * self.speaker_loss(
                    rst_output['gr_pred_speaker'],
                    speaker_target
                )
                
            if 'gr_pred_prosody' in rst_output and 'f0' in targets:
                losses['gr_prosody'] = self.weights['gr_prosody'] * self.prosody_loss(
                    rst_output['gr_pred_prosody'],
                    targets['f0'],
                    targets['uv']
                )
                
        # 总损失
        losses['total'] = sum(losses.values())
        
        return losses


def compute_eer(target_scores, nontarget_scores):
    """计算EER (Equal Error Rate)"""
    import numpy as np
    
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))
    
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    
    return eer
