"""
残差剥离塔(RST)训练器
基于PyTorch Lightning

支持两种语义监督模式：
- hubert: 使用预计算的HuBERT特征
- wav2vec_ctc: 使用Wav2Vec2-CTC在线提取phone IDs（推荐）
"""

import math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Optional, List, Any
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from ..losses.rst_loss import RSTLoss, compute_eer
from ..models.rst.semantic_supervision import Wav2VecCTCExtractor, FocalLoss


def get_input(x):
    x = x.to(memory_format=torch.contiguous_format)
    return x.float()


class RSTTrainer(pl.LightningModule):
    """
    RST深度伪造检测器训练器
    
    支持：
    1. 端到端训练（RST + 检测器）
    2. 两阶段训练：先训练RST，再训练检测器
    3. 多任务监督：语义、说话人、韵律
    """
    def __init__(
        self,
        model,
        lr: float = 3e-4,
        lr_rst: float = 1e-4,  # RST部分的学习率
        lr_detector: float = 3e-4,  # 检测器部分的学习率
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        
        # 损失权重
        semantic_weight: float = 1.0,
        speaker_weight: float = 1.0,
        prosody_weight: float = 1.0,
        detection_weight: float = 1.0,
        commit_weight: float = 0.25,
        gr_weight: float = 0.5,
        
        # 训练策略
        freeze_rst_epochs: int = 0,  # 前N个epoch冻结RST
        use_speaker_classification: bool = True,
        use_focal_loss: bool = False,
        label_smoothing: float = 0.0,
        
        # 语义监督配置
        semantic_mode: str = "wav2vec_ctc",  # "hubert" 或 "wav2vec_ctc"
        w2v_model_name: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
        
        # 其他
        save_score_path: str = './scores',
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        self.lr = lr
        self.lr_rst = lr_rst
        self.lr_detector = lr_detector
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.freeze_rst_epochs = freeze_rst_epochs
        self.save_score_path = save_score_path
        self.semantic_mode = semantic_mode
        
        # Wav2Vec CTC特征提取器（用于在线提取phone IDs）
        self.w2v_extractor = None
        if semantic_mode == "wav2vec_ctc":
            self.w2v_extractor = Wav2VecCTCExtractor(
                model_name=w2v_model_name,
                freeze=True
            )
            print(f"[RSTTrainer] Using Wav2Vec CTC semantic supervision ({w2v_model_name})")
        else:
            print(f"[RSTTrainer] Using HuBERT semantic supervision")
        
        # 损失函数
        self.criterion = RSTLoss(
            semantic_weight=semantic_weight,
            speaker_weight=speaker_weight,
            prosody_weight=prosody_weight,
            detection_weight=detection_weight,
            commit_weight=commit_weight,
            gr_semantic_weight=gr_weight,
            gr_speaker_weight=gr_weight,
            gr_prosody_weight=gr_weight,
            use_speaker_classification=use_speaker_classification,
            use_focal_loss=use_focal_loss,
            label_smoothing=label_smoothing,
            semantic_mode=semantic_mode,  # 语义监督模式
        )
        
        self.automatic_optimization = False
        
        # 验证/测试结果收集
        self.val_index_loader = []
        self.val_score_loader = []
        self.test_index_loader = []
        self.test_score_loader = []
        self.test_filename_loader = []
        
        self.default_monitor = "val_eer"
        
    def forward(self, batch, is_train: bool = True):
        """
        前向传播
        
        batch应包含：
        - audio: (B, 1, T) 音频波形
        - labels: (B,) 深度伪造标签
        - semantic_feat: HuBERT模式下为(B, T', D) HuBERT特征；Wav2Vec CTC模式下为占位符
        - speaker_ids/speaker_embeds: 说话人信息
        - f0: (B, T') 归一化F0
        - uv: (B, T') voiced/unvoiced mask
        """
        if is_train:
            audio, semantic_feat, f0, uv, speaker_id, labels = self._unpack_batch(batch, is_train)
        else:
            audio, semantic_feat, f0, uv, speaker_id, labels, audio_path = self._unpack_batch(batch, is_train)
            
        audio = get_input(audio)
        
        # 模型前向传播
        model_output = self.model(
            audio, 
            return_predictions=True,
            return_features=True
        )
        
        # 获取编码后的特征长度（用于对齐语义目标）
        target_length = model_output['rst_output']['encoded'].shape[-1] if 'rst_output' in model_output else None
        
        # 根据语义模式处理语义目标
        if self.semantic_mode == "wav2vec_ctc":
            # 在线提取phone IDs
            if self.w2v_extractor is not None:
                # 确保extractor在正确的设备上
                if hasattr(self.w2v_extractor, '_model') and self.w2v_extractor._model is not None:
                    self.w2v_extractor = self.w2v_extractor.to(audio.device)
                    
                phone_ids = self.w2v_extractor.extract_phone_ids(
                    audio.squeeze(1),  # (B, 1, T) -> (B, T)
                    sample_rate=16000,
                    target_length=target_length
                )
                semantic_target = phone_ids.to(audio.device)  # 确保在正确设备上
            else:
                semantic_target = None
        else:
            # HuBERT模式：使用预计算特征
            semantic_target = semantic_feat  # (B, T', D)
        
        # 构建目标dict
        targets = {
            'labels': labels,
            'semantic_target': semantic_target,  # phone_ids或hubert特征
            'f0': f0,
            'uv': uv,
        }
        
        if speaker_id is not None:
            if speaker_id.dtype == torch.long or speaker_id.dtype == torch.int:
                targets['speaker_ids'] = speaker_id
            else:
                targets['speaker_embeds'] = speaker_id
                
        if is_train:
            return model_output, targets
        else:
            return model_output, targets, audio_path
            
    def _unpack_batch(self, batch, is_train: bool):
        """
        解包batch数据
        
        RST数据集的batch格式:
        训练: (wavs, feats, f0s, uvs, speaker_ids, targets)
        测试: (wavs, feats, f0s, uvs, speaker_ids, targets, audio_paths)
        
        兼容原始SafeEar数据集格式:
        训练: (wavs, feats, targets)
        测试: (wavs, feats, targets, audio_paths)
        """
        if is_train:
            # RST格式: (audio, hubert_feat, f0, uv, speaker_id, labels)
            if len(batch) == 6:
                audio, hubert_feat, f0, uv, speaker_id, labels = batch
            # 原始格式兼容
            elif len(batch) == 5:
                audio, hubert_feat, f0, uv, labels = batch
                speaker_id = None
            elif len(batch) == 4:
                audio, hubert_feat, labels, speaker_id = batch
                f0, uv = None, None
            elif len(batch) == 3:
                audio, hubert_feat, labels = batch
                f0, uv, speaker_id = None, None, None
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")
            return audio, hubert_feat, f0, uv, speaker_id, labels
        else:
            # RST格式: (audio, hubert_feat, f0, uv, speaker_id, labels, audio_path)
            if len(batch) == 7:
                audio, hubert_feat, f0, uv, speaker_id, labels, audio_path = batch
            # 原始格式兼容
            elif len(batch) == 6:
                audio, hubert_feat, f0, uv, labels, audio_path = batch
                speaker_id = None
            elif len(batch) == 5:
                audio, hubert_feat, labels, speaker_id, audio_path = batch
                f0, uv = None, None
            elif len(batch) == 4:
                audio, hubert_feat, labels, audio_path = batch
                f0, uv, speaker_id = None, None, None
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")
            return audio, hubert_feat, f0, uv, speaker_id, labels, audio_path
            
    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        
        # 检查是否需要冻结RST
        if self.current_epoch < self.freeze_rst_epochs:
            if hasattr(self.model, 'freeze_rst'):
                self.model.freeze_rst()
        else:
            if hasattr(self.model, 'unfreeze_rst'):
                self.model.unfreeze_rst()
        
        # 前向传播
        model_output, targets = self(batch, is_train=True)
        
        # 计算损失
        losses = self.criterion(model_output, targets)
        
        # 反向传播
        optimizer.zero_grad()
        self.manual_backward(losses['total'])
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 日志
        log_dict = {f'train_{k}': v for k, v in losses.items() if k != 'total'}
        log_dict['train_loss'] = losses['total']
        
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True
        )
        
    def validation_step(self, batch, batch_idx):
        model_output, targets, _ = self(batch, is_train=False)
        
        # 获取预测分数
        probs = model_output['probs'][:, 0]  # real的概率
        labels = targets['labels']
        
        self.val_index_loader.append(labels)
        self.val_score_loader.append(probs)
        
        # 计算检测损失（验证阶段只计算检测损失，跳过其他辅助任务）
        losses = self.criterion(model_output, targets, detection_only=True)
        
        self.log_dict(
            {'val_loss': losses['total']},
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True
        )
        
    def on_validation_epoch_end(self):
        # 收集所有验证结果
        all_index = self.all_gather(torch.cat(self.val_index_loader, dim=0)).view(-1).cpu().numpy()
        all_score = self.all_gather(torch.cat(self.val_score_loader, dim=0)).view(-1).cpu().numpy()
        
        # 计算EER
        val_eer = compute_eer(all_score[all_index == 0], all_score[all_index == 1])
        other_val_eer = compute_eer(-all_score[all_index == 0], -all_score[all_index == 1])
        val_eer = min(val_eer, other_val_eer)
        
        self.log_dict(
            {"val_eer": val_eer},
            sync_dist=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        
        # 清理
        self.val_index_loader.clear()
        self.val_score_loader.clear()
        
        # 更新学习率
        self.log_dict(
            {"lr": self.optimizers().param_groups[0]['lr']},
            sync_dist=True,
            on_epoch=True,
            prog_bar=False,
            logger=True
        )
        
        # 调整学习率
        adjust_learning_rate(
            self.optimizers(), 
            self.current_epoch, 
            self.lr, 
            self.warmup_epochs, 
            self.max_epochs
        )
        
    def test_step(self, batch, batch_idx):
        model_output, targets, audio_path = self(batch, is_train=False)
        
        # 获取预测分数
        probs = model_output['probs'][:, 0]  # real的概率
        labels = targets['labels']
        
        self.test_index_loader.append(labels)
        self.test_score_loader.append(probs)
        self.test_filename_loader.append(audio_path)
        
    def on_test_epoch_end(self):
        # 收集所有测试结果
        all_index = self.all_gather(torch.cat(self.test_index_loader, dim=0)).view(-1).cpu().numpy()
        all_score = self.all_gather(torch.cat(self.test_score_loader, dim=0)).view(-1).cpu().numpy()
        
        # 计算EER
        test_eer = compute_eer(all_score[all_index == 0], all_score[all_index == 1])
        other_test_eer = compute_eer(-all_score[all_index == 0], -all_score[all_index == 1])
        test_eer = min(test_eer, other_test_eer)
        
        self.log_dict(
            {"test_eer": test_eer},
            sync_dist=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        
        # 清理
        self.test_index_loader.clear()
        self.test_score_loader.clear()
        self.test_filename_loader.clear()
        
    def configure_optimizers(self):
        # 分离RST和检测器的参数
        if hasattr(self.model, 'get_rst_parameters') and hasattr(self.model, 'get_detector_parameters'):
            params = [
                {'params': self.model.get_rst_parameters(), 'lr': self.lr_rst},
                {'params': self.model.get_detector_parameters(), 'lr': self.lr_detector},
            ]
        else:
            params = self.model.parameters()
            
        optimizer = torch.optim.AdamW(
            params if isinstance(params, list) else [{'params': params}],
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        return optimizer


def adjust_learning_rate(optimizer, epoch, lr, warmup, epochs=100):
    """余弦退火学习率调度"""
    if epoch < warmup:
        lr = lr / (warmup - epoch)
    else:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup) / (epochs - warmup)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class RSTTrainerSimple(pl.LightningModule):
    """
    简化版RST训练器
    只训练深度伪造检测，不训练监督任务
    适用于使用预训练RST特征提取器的场景
    """
    def __init__(
        self,
        model,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        use_focal_loss: bool = False,
        label_smoothing: float = 0.0,
        save_score_path: str = './scores',
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.save_score_path = save_score_path
        
        # 简单的BCE损失
        if use_focal_loss:
            from ..losses.rst_loss import DeepfakeDetectionLoss
            self.criterion = DeepfakeDetectionLoss(
                use_focal_loss=True,
                label_smoothing=label_smoothing
            )
        else:
            self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            
        self.automatic_optimization = False
        
        self.val_index_loader = []
        self.val_score_loader = []
        self.default_monitor = "val_eer"
        
    def forward(self, batch):
        audio, _, labels = batch[:3]
        audio = get_input(audio)
        
        # 冻结RST，只使用特征
        with torch.no_grad():
            self.model.freeze_rst()
            
        output = self.model(audio, return_predictions=False, return_features=False)
        return output['logits'], labels
        
    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        
        logits, labels = self(batch)
        loss = self.criterion(logits, labels)
        
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
    def validation_step(self, batch, batch_idx):
        logits, labels = self(batch)
        probs = F.softmax(logits, dim=-1)[:, 0]
        
        self.val_index_loader.append(labels)
        self.val_score_loader.append(probs)
        
    def on_validation_epoch_end(self):
        all_index = self.all_gather(torch.cat(self.val_index_loader, dim=0)).view(-1).cpu().numpy()
        all_score = self.all_gather(torch.cat(self.val_score_loader, dim=0)).view(-1).cpu().numpy()
        
        val_eer = compute_eer(all_score[all_index == 0], all_score[all_index == 1])
        other_val_eer = compute_eer(-all_score[all_index == 0], -all_score[all_index == 1])
        val_eer = min(val_eer, other_val_eer)
        
        self.log("val_eer", val_eer, prog_bar=True)
        
        self.val_index_loader.clear()
        self.val_score_loader.clear()
        
        adjust_learning_rate(self.optimizers(), self.current_epoch, self.lr, self.warmup_epochs, self.max_epochs)
        
    def configure_optimizers(self):
        # 只优化检测器参数
        if hasattr(self.model, 'get_detector_parameters'):
            params = self.model.get_detector_parameters()
        else:
            params = self.model.parameters()
            
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
