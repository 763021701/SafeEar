"""
RST深度伪造检测器
整合残差剥离塔和分类器的完整检测系统
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple


class RSTDetector(nn.Module):
    """
    基于残差剥离塔的深度伪造检测器
    
    支持两种模式：
    1. 端到端训练：同时训练RST和检测器
    2. 特征提取+分类：使用预训练RST提取特征，只训练分类器
    """
    def __init__(
        self,
        # RST参数
        encoder_dim: int = 1024,
        encoder_n_filters: int = 64,
        encoder_strides: List[int] = [8, 5, 4, 2],
        n_q_semantic: int = 2,
        n_q_speaker: int = 2,
        n_q_prosody: int = 2,
        n_q_residual: int = 0,
        codebook_size: int = 1024,
        
        # 语义监督配置
        semantic_mode: str = "wav2vec_ctc",  # "hubert" 或 "wav2vec_ctc"
        hubert_dim: int = 768,  # HuBERT特征维度
        num_phone_classes: int = 5003,  # Wav2Vec CTC词汇表大小
        
        # 说话人监督配置
        num_speakers: int = 0,
        speaker_embed_dim: int = 192,
        use_gradient_reversal: bool = True,
        gr_alpha: float = 1.0,
        
        # 检测器参数
        num_classes: int = 2,
        detector_num_layers: int = 2,
        detector_num_heads: int = 8,
        detector_mlp_ratio: float = 2.0,
        detector_dropout: float = 0.1,
        
        # 特征融合方式
        feature_fusion: str = 'residual_only',  # 'residual_only', 'all_layers', 'weighted'
    ):
        super().__init__()
        
        self.feature_fusion = feature_fusion
        self.encoder_dim = encoder_dim
        self.semantic_mode = semantic_mode
        
        # 导入RST
        from .residual_stripping_tower import ResidualStrippingTower
        from .predictors import DeepfakeClassifier
        
        # 残差剥离塔
        self.rst = ResidualStrippingTower(
            encoder_dim=encoder_dim,
            encoder_n_filters=encoder_n_filters,
            encoder_strides=encoder_strides,
            n_q_semantic=n_q_semantic,
            n_q_speaker=n_q_speaker,
            n_q_prosody=n_q_prosody,
            n_q_residual=n_q_residual,
            codebook_size=codebook_size,
            semantic_mode=semantic_mode,
            hubert_dim=hubert_dim,
            num_phone_classes=num_phone_classes,
            num_speakers=num_speakers,
            speaker_embed_dim=speaker_embed_dim,
            use_gradient_reversal=use_gradient_reversal,
            gr_alpha=gr_alpha,
        )
        
        # 计算检测器输入维度
        if feature_fusion == 'residual_only':
            detector_input_dim = encoder_dim
        elif feature_fusion == 'all_layers':
            detector_input_dim = encoder_dim * 4  # sem + spk + pros + res
        elif feature_fusion == 'weighted':
            detector_input_dim = encoder_dim
            self.layer_weights = nn.Parameter(torch.ones(4) / 4)
        else:
            raise ValueError(f"Unknown feature_fusion: {feature_fusion}")
            
        # 特征投影层（如果需要）
        if feature_fusion == 'all_layers':
            self.feature_proj = nn.Sequential(
                nn.Conv1d(detector_input_dim, encoder_dim, kernel_size=1),
                nn.SiLU(),
                nn.Conv1d(encoder_dim, encoder_dim, kernel_size=1),
            )
        else:
            self.feature_proj = nn.Identity()
        
        # 深度伪造检测器
        self.detector = DeepfakeClassifier(
            embedding_dim=encoder_dim,
            num_classes=num_classes,
            num_layers=detector_num_layers,
            num_heads=detector_num_heads,
            mlp_ratio=detector_mlp_ratio,
            dropout_rate=detector_dropout,
        )
        
    def _fuse_features(self, stripped_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        融合各层特征
        """
        if self.feature_fusion == 'residual_only':
            return stripped_output['deepfake_residual']
            
        elif self.feature_fusion == 'all_layers':
            features = torch.cat([
                stripped_output['quantized_semantic'],
                stripped_output['quantized_speaker'],
                stripped_output['quantized_prosody'],
                stripped_output['deepfake_residual'],
            ], dim=1)
            return self.feature_proj(features)
            
        elif self.feature_fusion == 'weighted':
            weights = F.softmax(self.layer_weights, dim=0)
            features = (
                weights[0] * stripped_output['quantized_semantic'] +
                weights[1] * stripped_output['quantized_speaker'] +
                weights[2] * stripped_output['quantized_prosody'] +
                weights[3] * stripped_output['deepfake_residual']
            )
            return features
            
    def forward(
        self,
        x: torch.Tensor,
        return_predictions: bool = True,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: (B, 1, T) 音频波形
            return_predictions: 是否返回监督任务预测
            return_features: 是否返回中间特征
            
        Returns:
            dict with:
                - logits: (B, num_classes) 分类logits
                - probs: (B, num_classes) 分类概率
                - (可选) rst_output: RST完整输出
                - (可选) pooled_features: 池化后的特征
        """
        # RST前向传播
        rst_output = self.rst(x, return_predictions=return_predictions, return_reconstruction=False)
        
        # 特征融合
        fused_features = self._fuse_features(rst_output)
        
        # 检测
        logits, pooled_features = self.detector(fused_features, return_features=True)
        probs = F.softmax(logits, dim=-1)
        
        output = {
            'logits': logits,
            'probs': probs,
        }
        
        if return_predictions:
            output['rst_output'] = rst_output
            
        if return_features:
            output['fused_features'] = fused_features
            output['pooled_features'] = pooled_features
            
        return output
    
    def detect(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        简化的检测接口
        
        Returns:
            predictions: (B,) 预测标签 (1=fake, 0=real)
            confidences: (B,) 预测置信度
        """
        output = self.forward(x, return_predictions=False, return_features=False)
        probs = output['probs']
        predictions = torch.argmax(probs, dim=-1)
        confidences = probs.max(dim=-1).values
        return predictions, confidences
    
    def get_deepfake_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取深度伪造分数
        
        Returns:
            scores: (B,) 伪造概率分数 (越高越可能是fake)
        """
        output = self.forward(x, return_predictions=False, return_features=False)
        return output['probs'][:, 1]  # fake概率
    
    def freeze_rst(self):
        """冻结RST参数，只训练检测器"""
        for param in self.rst.parameters():
            param.requires_grad = False
            
    def unfreeze_rst(self):
        """解冻RST参数"""
        for param in self.rst.parameters():
            param.requires_grad = True
            
    def get_rst_parameters(self):
        """获取RST参数"""
        return self.rst.parameters()
    
    def get_detector_parameters(self):
        """获取检测器参数"""
        return self.detector.parameters()
