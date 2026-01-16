"""
预测器模块 (Predictors)
用于监督各层VQ学习特定信息

包括：
1. SemanticPredictor - 语义预测器 (监督VQ学习语义信息)
2. SpeakerPredictor - 说话人预测器 (监督VQ学习说话人信息)
3. ProsodyPredictor - 韵律预测器 (监督VQ学习韵律/F0信息)
4. DeepfakeClassifier - 深度伪造分类器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ResidualConvBlock(nn.Module):
    """残差卷积块，用于特征提取"""
    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)


class CNNLSTMPredictor(nn.Module):
    """
    CNN-LSTM预测器基类
    参考FACodec的CNNLSTM结构
    """
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        num_heads: int = 1,
        hidden_dim: int = 256,
        global_pred: bool = False
    ):
        super().__init__()
        self.global_pred = global_pred
        
        self.conv_blocks = nn.Sequential(
            ResidualConvBlock(in_dim, dilation=1),
            ResidualConvBlock(in_dim, dilation=2),
            ResidualConvBlock(in_dim, dilation=3),
        )
        
        # 多头输出
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, out_dim)
            )
            for _ in range(num_heads)
        ])

    def forward(self, x):
        """
        Args:
            x: (B, C, T)
        Returns:
            list of (B, T, out_dim) or (B, out_dim) if global_pred
        """
        x = self.conv_blocks(x)  # (B, C, T)
        x = x.transpose(1, 2)  # (B, T, C)
        
        if self.global_pred:
            x = torch.mean(x, dim=1, keepdim=False)  # (B, C)
            
        outputs = [head(x) for head in self.heads]
        return outputs


class SemanticPredictor(nn.Module):
    """
    语义预测器
    监督语义VQ学习语义信息
    
    支持两种模式：
    1. HuBERT模式 (mode='hubert'): 预测连续HuBERT特征，使用余弦相似度/MSE损失
    2. Wav2Vec CTC模式 (mode='wav2vec_ctc'): 预测离散phone ID，使用交叉熵/Focal损失
    """
    def __init__(
        self,
        in_dim: int = 1024,
        mode: str = "hubert",  # "hubert" 或 "wav2vec_ctc"
        hubert_dim: int = 768,  # HuBERT特征维度
        num_phone_classes: int = 5003,  # Wav2Vec CTC的词汇表大小
        hidden_dim: int = 512
    ):
        super().__init__()
        self.mode = mode
        self.in_dim = in_dim
        self.hubert_dim = hubert_dim
        self.num_phone_classes = num_phone_classes
        
        # 共享的卷积块
        self.conv_blocks = nn.Sequential(
            ResidualConvBlock(in_dim, dilation=1),
            ResidualConvBlock(in_dim, dilation=2),
        )
        
        if mode == "hubert":
            # HuBERT模式：预测连续特征向量
            self.head = nn.Sequential(
                nn.Conv1d(in_dim, hidden_dim, kernel_size=1),
                nn.SiLU(),
                nn.Conv1d(hidden_dim, hubert_dim, kernel_size=1),
            )
            self.output_dim = hubert_dim
            
        elif mode == "wav2vec_ctc":
            # Wav2Vec CTC模式：预测离散phone ID
            # 参考FACodec的CNNLSTM设计，使用更多卷积层
            self.head = nn.Sequential(
                nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv1d(hidden_dim, num_phone_classes, kernel_size=1),
            )
            self.output_dim = num_phone_classes
            
        else:
            raise ValueError(f"Unknown semantic mode: {mode}. Use 'hubert' or 'wav2vec_ctc'")
        
    def forward(self, x):
        """
        Args:
            x: (B, C, T) 量化后的语义特征
        Returns:
            HuBERT模式: (B, T, hubert_dim) 预测的HuBERT特征
            Wav2Vec CTC模式: (B, T, num_phone_classes) phone logits
        """
        x = self.conv_blocks(x)  # (B, C, T)
        out = self.head(x)  # (B, out_dim, T)
        return out.transpose(1, 2)  # (B, T, out_dim)


class SpeakerPredictor(nn.Module):
    """
    说话人预测器
    监督说话人VQ学习说话人身份信息
    
    支持两种模式：
    1. 分类模式：预测说话人ID (num_speakers > 0)
    2. 嵌入模式：预测说话人嵌入向量 (用于与ECAPA-TDNN对齐)
    """
    def __init__(
        self,
        in_dim: int = 1024,
        num_speakers: int = 0,  # 0表示使用嵌入模式
        speaker_embed_dim: int = 192,  # ECAPA-TDNN输出维度
        hidden_dim: int = 512
    ):
        super().__init__()
        self.num_speakers = num_speakers
        self.use_classification = num_speakers > 0
        
        # 时序特征提取
        self.conv_blocks = nn.Sequential(
            ResidualConvBlock(in_dim, dilation=1),
            ResidualConvBlock(in_dim, dilation=2),
            ResidualConvBlock(in_dim, dilation=3),
        )
        
        # 全局池化后的预测头
        if self.use_classification:
            self.head = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_speakers)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, speaker_embed_dim)
            )
            
    def forward(self, x):
        """
        Args:
            x: (B, C, T) 量化后的说话人特征
        Returns:
            如果分类模式: (B, num_speakers) logits
            如果嵌入模式: (B, speaker_embed_dim) 说话人嵌入
        """
        x = self.conv_blocks(x)  # (B, C, T)
        x = torch.mean(x, dim=2)  # 全局平均池化 (B, C)
        return self.head(x)


class ProsodyPredictor(nn.Module):
    """
    韵律预测器
    监督韵律VQ学习F0和能量信息
    
    输出：
    1. Normalized F0 (log scale, z-score normalized)
    2. UV (voiced/unvoiced) flag
    3. (可选) Energy
    """
    def __init__(
        self,
        in_dim: int = 1024,
        hidden_dim: int = 256,
        predict_energy: bool = False
    ):
        super().__init__()
        self.predict_energy = predict_energy
        
        self.conv_blocks = nn.Sequential(
            ResidualConvBlock(in_dim, dilation=1),
            ResidualConvBlock(in_dim, dilation=2),
            ResidualConvBlock(in_dim, dilation=3),
        )
        
        # F0预测头
        self.f0_head = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1)
        )
        
        # UV预测头 (voiced/unvoiced)
        self.uv_head = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1)
        )
        
        # 能量预测头 (可选)
        if predict_energy:
            self.energy_head = nn.Sequential(
                nn.Conv1d(in_dim, hidden_dim, kernel_size=1),
                nn.SiLU(),
                nn.Conv1d(hidden_dim, 1, kernel_size=1)
            )
            
    def forward(self, x):
        """
        Args:
            x: (B, C, T) 量化后的韵律特征
        Returns:
            dict with:
                - f0: (B, T) normalized F0
                - uv: (B, T) voiced/unvoiced logits
                - energy: (B, T) if predict_energy
        """
        x = self.conv_blocks(x)
        
        f0 = self.f0_head(x).squeeze(1)  # (B, T)
        uv = self.uv_head(x).squeeze(1)  # (B, T)
        
        output = {'f0': f0, 'uv': uv}
        
        if self.predict_energy:
            energy = self.energy_head(x).squeeze(1)
            output['energy'] = energy
            
        return output


class DeepfakeClassifier(nn.Module):
    """
    深度伪造分类器
    基于Transformer架构，对残差特征进行分类
    
    使用注意力池化 + 分类头
    """
    def __init__(
        self,
        embedding_dim: int = 1024,
        num_classes: int = 2,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout_rate: float = 0.1,
        max_seq_len: int = 1000
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # 位置编码
        self.positional_emb = nn.Parameter(
            self._sinusoidal_embedding(max_seq_len, embedding_dim),
            requires_grad=False
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=int(embedding_dim * mlp_ratio),
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 层归一化
        self.norm = nn.LayerNorm(embedding_dim)
        
        # 注意力池化
        self.attention_pool = nn.Linear(embedding_dim, 1)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim // 2, num_classes)
        )
        
    def _sinusoidal_embedding(self, n_channels, dim):
        pe = torch.FloatTensor([
            [p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
            for p in range(n_channels)
        ])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)
        
    def forward(self, x, return_features: bool = False):
        """
        Args:
            x: (B, C, T) 残差特征
            return_features: 是否返回中间特征
        Returns:
            logits: (B, num_classes)
            features: (B, embedding_dim) if return_features
        """
        x = x.transpose(1, 2)  # (B, T, C)
        seq_len = x.size(1)
        
        # 添加位置编码
        x = x + self.positional_emb[:, :seq_len, :].to(x.device)
        
        # Transformer编码
        x = self.transformer(x)
        x = self.norm(x)
        
        # 注意力池化
        attn_weights = F.softmax(self.attention_pool(x), dim=1)  # (B, T, 1)
        features = torch.sum(x * attn_weights, dim=1)  # (B, C)
        
        # 分类
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits


class MultiTaskPredictor(nn.Module):
    """
    多任务预测器
    集成所有监督任务的预测器
    支持对残差使用梯度反转层进行对抗训练
    """
    def __init__(
        self,
        in_dim: int = 1024,
        hubert_dim: int = 768,
        num_speakers: int = 0,
        speaker_embed_dim: int = 192,
        use_gr_on_residual: bool = True,
        gr_alpha: float = 1.0
    ):
        super().__init__()
        from .gradient_reversal import GradientReversal
        
        self.use_gr = use_gr_on_residual
        
        # 主预测器 (正向监督)
        self.semantic_predictor = SemanticPredictor(in_dim, hubert_dim)
        self.speaker_predictor = SpeakerPredictor(in_dim, num_speakers, speaker_embed_dim)
        self.prosody_predictor = ProsodyPredictor(in_dim)
        
        # 残差对抗预测器 (梯度反转，确保残差不含这些信息)
        if use_gr_on_residual:
            self.gr_layer = GradientReversal(alpha=gr_alpha)
            self.residual_semantic_predictor = SemanticPredictor(in_dim, hubert_dim)
            self.residual_speaker_predictor = SpeakerPredictor(in_dim, num_speakers, speaker_embed_dim)
            self.residual_prosody_predictor = ProsodyPredictor(in_dim)
            
    def forward(self, quantized_layers, residual=None):
        """
        Args:
            quantized_layers: dict with 'semantic', 'speaker', 'prosody' quantized features
            residual: (B, C, T) 最终残差特征
        Returns:
            predictions: dict with all predictions
            gr_predictions: dict with gradient-reversed predictions on residual
        """
        predictions = {}
        gr_predictions = {}
        
        # 主预测
        if 'semantic' in quantized_layers:
            predictions['semantic'] = self.semantic_predictor(quantized_layers['semantic'])
        if 'speaker' in quantized_layers:
            predictions['speaker'] = self.speaker_predictor(quantized_layers['speaker'])
        if 'prosody' in quantized_layers:
            predictions['prosody'] = self.prosody_predictor(quantized_layers['prosody'])
            
        # 残差对抗预测
        if self.use_gr and residual is not None:
            reversed_residual = self.gr_layer(residual)
            gr_predictions['semantic'] = self.residual_semantic_predictor(reversed_residual)
            gr_predictions['speaker'] = self.residual_speaker_predictor(reversed_residual)
            gr_predictions['prosody'] = self.residual_prosody_predictor(reversed_residual)
            
        return predictions, gr_predictions
