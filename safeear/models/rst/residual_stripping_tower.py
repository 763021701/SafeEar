"""
残差剥离塔 (Residual-Stripping Tower)

核心思想：
通过多级RVQ逐层剥离非判伪相关信息（语义、说话人、韵律），
最终残差只包含判伪相关特征，从而提升跨数据集泛化能力。

架构：
Layer 1: Semantic VQ - 剥离语义内容（HuBERT/Wav2Vec CTC监督）
Layer 2: Speaker VQ - 剥离说话人信息（说话人ID/ECAPA监督）
Layer 3: Prosody VQ - 剥离韵律信息（Normalized F0监督）
Layer 4: Deepfake Residual - 仅包含判伪相关特征

语义监督模式：
- hubert: 使用HuBERT特征蒸馏（需预计算特征）
- wav2vec_ctc: 使用Wav2Vec2-CTC预测phone ID（可在线提取）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
from einops import rearrange

# 复用现有的模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..modules.seanet import SEANetEncoder, SEANetDecoder
from ..modules.quantization import ResidualVectorQuantizer
from .gradient_reversal import GradientReversal
from .predictors import (
    SemanticPredictor, 
    SpeakerPredictor, 
    ProsodyPredictor,
    DeepfakeClassifier
)


class StrippingLayer(nn.Module):
    """
    单个剥离层
    包含一个VQ量化器和对应的预测器
    """
    def __init__(
        self,
        dim: int = 1024,
        codebook_size: int = 1024,
        num_quantizers: int = 1,
        codebook_dim: int = 8,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        self.quantizer = ResidualVectorQuantizer(
            dimension=dim,
            n_q=num_quantizers,
            bins=codebook_size,
        )
        
    def forward(self, x, n_q=None):
        """
        Args:
            x: (B, D, T) 输入特征
        Returns:
            quantized: (B, D, T) 量化后的特征
            residual: (B, D, T) 残差
            commit_loss: 承诺损失
            codes: 量化码
        """
        # 量化
        quantized, codes, commit_loss, quantized_list = self.quantizer(x, n_q=n_q)
        
        # 计算残差
        residual = x - quantized.detach()  # detach避免梯度回传到之前层
        
        return quantized, residual, commit_loss, codes


class ResidualStrippingTower(nn.Module):
    """
    残差剥离塔主模型
    
    通过三级VQ逐层剥离：
    1. 语义层 (Semantic) - HuBERT/Wav2Vec CTC监督
    2. 说话人层 (Speaker) - 说话人ID/嵌入监督
    3. 韵律层 (Prosody) - Normalized F0监督
    
    最终残差用于深度伪造检测
    """
    def __init__(
        self,
        # 编码器参数
        encoder_dim: int = 1024,
        encoder_n_filters: int = 64,
        encoder_strides: List[int] = [8, 5, 4, 2],
        encoder_lstm_layers: int = 2,
        
        # 量化器参数
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        
        # 各层量化器数量
        n_q_semantic: int = 2,
        n_q_speaker: int = 2,
        n_q_prosody: int = 2,
        n_q_residual: int = 2,  # 可选：对最终残差也做VQ
        
        # 语义监督配置
        semantic_mode: str = "wav2vec_ctc",  # "hubert" 或 "wav2vec_ctc"
        hubert_dim: int = 768,  # HuBERT特征维度
        num_phone_classes: int = 5003,  # Wav2Vec CTC词汇表大小
        
        # 说话人监督配置
        num_speakers: int = 0,  # 0表示使用嵌入模式
        speaker_embed_dim: int = 192,
        
        # 梯度反转
        use_gradient_reversal: bool = True,
        gr_alpha: float = 1.0,
        
        # 其他
        sample_rate: int = 16000,
    ):
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.sample_rate = sample_rate
        self.use_gradient_reversal = use_gradient_reversal
        self.semantic_mode = semantic_mode
        self.hubert_dim = hubert_dim
        self.num_phone_classes = num_phone_classes
        
        # ========== 编码器 ==========
        self.encoder = SEANetEncoder(
            n_filters=encoder_n_filters,
            dimension=encoder_dim,
            ratios=encoder_strides,
            lstm=encoder_lstm_layers,
            bidirectional=True,
            dilation_base=2,
            residual_kernel_size=3,
            n_residual_layers=1,
            activation='ELU'
        )
        self.hop_length = 1
        for s in encoder_strides:
            self.hop_length *= s
        
        # ========== 剥离层 ==========
        # Layer 1: 语义剥离
        self.semantic_vq = ResidualVectorQuantizer(
            dimension=encoder_dim,
            n_q=n_q_semantic,
            bins=codebook_size,
        )
        
        # Layer 2: 说话人剥离
        self.speaker_vq = ResidualVectorQuantizer(
            dimension=encoder_dim,
            n_q=n_q_speaker,
            bins=codebook_size,
        )
        
        # Layer 3: 韵律剥离
        self.prosody_vq = ResidualVectorQuantizer(
            dimension=encoder_dim,
            n_q=n_q_prosody,
            bins=codebook_size,
        )
        
        # Layer 4: 残差VQ (可选，对最终判伪特征做量化)
        self.use_residual_vq = n_q_residual > 0
        if self.use_residual_vq:
            self.residual_vq = ResidualVectorQuantizer(
                dimension=encoder_dim,
                n_q=n_q_residual,
                bins=codebook_size,
            )
        
        # ========== 预测器 (监督任务) ==========
        # 语义预测器 (支持HuBERT和Wav2Vec CTC两种模式)
        self.semantic_predictor = SemanticPredictor(
            in_dim=encoder_dim,
            mode=semantic_mode,
            hubert_dim=hubert_dim,
            num_phone_classes=num_phone_classes,
        )
        
        # 说话人预测器
        self.speaker_predictor = SpeakerPredictor(
            in_dim=encoder_dim,
            num_speakers=num_speakers,
            speaker_embed_dim=speaker_embed_dim
        )
        
        # 韵律预测器
        self.prosody_predictor = ProsodyPredictor(
            in_dim=encoder_dim
        )
        
        # ========== 梯度反转预测器 (确保残差不含这些信息) ==========
        if use_gradient_reversal:
            self.gr_layer = GradientReversal(alpha=gr_alpha)
            
            # 对最终残差的对抗预测器
            self.gr_semantic_predictor = SemanticPredictor(
                in_dim=encoder_dim,
                mode=semantic_mode,
                hubert_dim=hubert_dim,
                num_phone_classes=num_phone_classes,
            )
            self.gr_speaker_predictor = SpeakerPredictor(
                in_dim=encoder_dim,
                num_speakers=num_speakers,
                speaker_embed_dim=speaker_embed_dim
            )
            self.gr_prosody_predictor = ProsodyPredictor(
                in_dim=encoder_dim
            )
        
        # ========== 解码器 (可选，用于重建) ==========
        self.decoder = SEANetDecoder(
            n_filters=encoder_n_filters,
            dimension=encoder_dim,
            ratios=encoder_strides[::-1],  # 反转strides
            lstm=encoder_lstm_layers,
            bidirectional=False,
            dilation_base=2,
            residual_kernel_size=3,
            n_residual_layers=1,
            activation='ELU'
        )
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码音频
        Args:
            x: (B, 1, T) 音频波形
        Returns:
            (B, D, T') 编码特征
        """
        return self.encoder(x)
    
    def strip_layers(
        self, 
        encoded: torch.Tensor,
        return_all_residuals: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        逐层剥离非判伪信息
        
        Args:
            encoded: (B, D, T) 编码特征
            return_all_residuals: 是否返回每层的残差
            
        Returns:
            dict containing:
                - quantized_semantic: 语义量化特征
                - quantized_speaker: 说话人量化特征  
                - quantized_prosody: 韵律量化特征
                - deepfake_residual: 最终残差（判伪特征）
                - commit_losses: 各层承诺损失
                - (可选) intermediate_residuals: 中间残差
        """
        output = {}
        commit_losses = []
        
        # Layer 1: 语义剥离
        q_sem, codes_sem, loss_sem, _ = self.semantic_vq(encoded)
        residual_1 = encoded - q_sem.detach()
        output['quantized_semantic'] = q_sem
        commit_losses.append(loss_sem)
        
        # Layer 2: 说话人剥离 (从残差1中剥离)
        q_spk, codes_spk, loss_spk, _ = self.speaker_vq(residual_1)
        residual_2 = residual_1 - q_spk.detach()
        output['quantized_speaker'] = q_spk
        commit_losses.append(loss_spk)
        
        # Layer 3: 韵律剥离 (从残差2中剥离)
        q_pros, codes_pros, loss_pros, _ = self.prosody_vq(residual_2)
        residual_3 = residual_2 - q_pros.detach()
        output['quantized_prosody'] = q_pros
        commit_losses.append(loss_pros)
        
        # Layer 4: 最终残差 (判伪特征)
        if self.use_residual_vq:
            q_res, codes_res, loss_res, _ = self.residual_vq(residual_3)
            output['deepfake_residual'] = q_res
            commit_losses.append(loss_res)
        else:
            output['deepfake_residual'] = residual_3
            
        output['commit_losses'] = commit_losses
        
        if return_all_residuals:
            output['residual_after_semantic'] = residual_1
            output['residual_after_speaker'] = residual_2
            output['residual_after_prosody'] = residual_3
            
        return output
    
    def predict_supervision_tasks(
        self, 
        stripped_output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        执行监督任务预测
        
        Returns:
            dict with predictions:
                - pred_semantic: 预测的HuBERT特征
                - pred_speaker: 预测的说话人logits/嵌入
                - pred_prosody: 预测的F0和UV
                - (如果使用GR) gr_pred_*: 对残差的对抗预测
        """
        predictions = {}
        
        # 主预测任务
        predictions['pred_semantic'] = self.semantic_predictor(
            stripped_output['quantized_semantic']
        )
        predictions['pred_speaker'] = self.speaker_predictor(
            stripped_output['quantized_speaker']
        )
        predictions['pred_prosody'] = self.prosody_predictor(
            stripped_output['quantized_prosody']
        )
        
        # 梯度反转对抗预测 (确保残差不含这些信息)
        if self.use_gradient_reversal:
            residual = stripped_output['deepfake_residual']
            reversed_residual = self.gr_layer(residual)
            
            predictions['gr_pred_semantic'] = self.gr_semantic_predictor(reversed_residual)
            predictions['gr_pred_speaker'] = self.gr_speaker_predictor(reversed_residual)
            predictions['gr_pred_prosody'] = self.gr_prosody_predictor(reversed_residual)
            
        return predictions
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        解码重建音频
        Args:
            features: (B, D, T) 特征（可以是完整特征或部分特征）
        Returns:
            (B, 1, T') 重建音频
        """
        return self.decoder(features)
    
    def forward(
        self,
        x: torch.Tensor,
        return_predictions: bool = True,
        return_reconstruction: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        完整前向传播
        
        Args:
            x: (B, 1, T) 音频波形
            return_predictions: 是否返回监督任务预测
            return_reconstruction: 是否返回重建音频
            
        Returns:
            dict containing all outputs
        """
        output = {}
        
        # 编码
        encoded = self.encode(x)
        output['encoded'] = encoded
        
        # 剥离层
        stripped = self.strip_layers(encoded, return_all_residuals=True)
        output.update(stripped)
        
        # 监督任务预测
        if return_predictions:
            predictions = self.predict_supervision_tasks(stripped)
            output.update(predictions)
            
        # 重建 (使用所有量化特征之和)
        if return_reconstruction:
            reconstructed_features = (
                stripped['quantized_semantic'] +
                stripped['quantized_speaker'] +
                stripped['quantized_prosody'] +
                stripped['deepfake_residual']
            )
            output['reconstructed'] = self.decode(reconstructed_features)
            
        return output
    
    def get_deepfake_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取用于深度伪造检测的特征
        
        Args:
            x: (B, 1, T) 音频波形
        Returns:
            (B, D, T') 判伪特征
        """
        encoded = self.encode(x)
        stripped = self.strip_layers(encoded)
        return stripped['deepfake_residual']


class RSTWithDetector(nn.Module):
    """
    残差剥离塔 + 深度伪造检测器
    端到端的深度伪造检测模型
    """
    def __init__(
        self,
        rst_config: dict,
        detector_config: dict
    ):
        super().__init__()
        
        self.rst = ResidualStrippingTower(**rst_config)
        self.detector = DeepfakeClassifier(**detector_config)
        
    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 1, T) 音频波形
            return_all: 是否返回所有中间结果
        Returns:
            dict with:
                - logits: (B, num_classes) 分类logits
                - deepfake_features: (B, D, T') 判伪特征
                - (可选) 其他中间结果
        """
        # 获取RST输出
        rst_output = self.rst(x, return_predictions=return_all, return_reconstruction=False)
        
        # 深度伪造检测
        deepfake_features = rst_output['deepfake_residual']
        logits, features = self.detector(deepfake_features, return_features=True)
        
        output = {
            'logits': logits,
            'deepfake_features': deepfake_features,
            'pooled_features': features
        }
        
        if return_all:
            output.update(rst_output)
            
        return output
    
    def detect(self, x: torch.Tensor) -> torch.Tensor:
        """
        简化的检测接口
        Returns:
            (B,) 预测概率 (1=fake, 0=real)
        """
        logits = self.forward(x)['logits']
        probs = F.softmax(logits, dim=-1)
        return probs[:, 1]  # 返回fake的概率
