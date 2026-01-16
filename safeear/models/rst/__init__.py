# Residual-Stripping Tower (RST) for Deepfake Detection
# 残差剥离塔：通过逐层剥离非判伪信息，保留判伪相关特征
# 
# 语义监督模式：
# - hubert: 使用HuBERT特征蒸馏（需预计算特征）
# - wav2vec_ctc: 使用Wav2Vec2-CTC预测phone ID（可在线提取，推荐）

from .residual_stripping_tower import ResidualStrippingTower
from .rst_detector import RSTDetector
from .gradient_reversal import GradientReversal
from .predictors import (
    SemanticPredictor,
    SpeakerPredictor, 
    ProsodyPredictor,
    DeepfakeClassifier
)
from .semantic_supervision import (
    Wav2VecCTCExtractor,
    SemanticLoss,
    FocalLoss,
)

__all__ = [
    'ResidualStrippingTower',
    'RSTDetector',
    'GradientReversal',
    'SemanticPredictor',
    'SpeakerPredictor',
    'ProsodyPredictor',
    'DeepfakeClassifier',
    # 语义监督相关
    'Wav2VecCTCExtractor',
    'SemanticLoss',
    'FocalLoss',
]
