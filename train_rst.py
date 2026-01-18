"""
残差剥离塔 (Residual-Stripping Tower) 训练脚本
用于深度伪造检测的泛化能力增强
"""

import importlib
import json
import os
import warnings
warnings.filterwarnings("ignore")
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import argparse
import pytorch_lightning as pl
import torch
import hydra

torch.set_float32_matmul_precision("high")

from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.strategies.ddp import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only


@rank_zero_only
def print_only(message: str):
    """Prints a message only on rank 0."""
    print(message)


def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    训练RST深度伪造检测模型
    """
    # 实例化数据模块
    print_only(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()
    
    # 实例化RST模型
    print_only(f"Instantiating RST model <{cfg.rst_model._target_}>")
    model: torch.nn.Module = hydra.utils.instantiate(cfg.rst_model)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_only(f"Total parameters: {total_params / 1e6:.2f}M")
    print_only(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # 实例化训练系统
    print_only(f"Instantiating system <{cfg.system._target_}>")
    system: LightningModule = hydra.utils.instantiate(
        cfg.system,
        model=model,
    )
    
    # 实例化回调
    callbacks: List[Callback] = []
    if cfg.get("early_stopping"):
        print_only(f"Instantiating early_stopping <{cfg.early_stopping._target_}>")
        callbacks.append(hydra.utils.instantiate(cfg.early_stopping))
    if cfg.get("checkpoint"):
        print_only(f"Instantiating checkpoint <{cfg.checkpoint._target_}>")
        checkpoint: pl.callbacks.ModelCheckpoint = hydra.utils.instantiate(cfg.checkpoint)
        callbacks.append(checkpoint)
        
    # 实例化日志器
    print_only(f"Instantiating logger <{cfg.logger._target_}>")
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name, "logs"), exist_ok=True)
    logger = hydra.utils.instantiate(cfg.logger)

    # 保存配置：每次运行都保存到独立目录，避免覆盖历史 config.yaml
    # - 写入 TensorBoardLogger 的 log_dir（包含 version_x）
    # - 同时在 exp 根目录保存一份时间戳文件，并维护 config_latest.yaml
    exp_dir = os.path.join(cfg.exp.dir, cfg.exp.name)
    os.makedirs(exp_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    OmegaConf.save(cfg, os.path.join(exp_dir, f"config_{run_id}.yaml"))
    try:
        log_dir = getattr(logger, "log_dir", None)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            OmegaConf.save(cfg, os.path.join(log_dir, "config.yaml"))
    except Exception as e:
        print_only(f"[WARN] Failed to save config into logger.log_dir: {e}")
    
    # 实例化训练器
    print_only(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=True),
    )
    
    # 开始训练
    print_only("=" * 60)
    print_only("Starting RST training...")
    print_only("=" * 60)
    
    trainer.fit(system, datamodule=datamodule)
    
    print_only("Training finished!")
    
    # 保存最佳模型信息
    if cfg.get("checkpoint"):
        best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
        with open(os.path.join(cfg.exp.dir, cfg.exp.name, "best_k_models.json"), "w") as f:
            json.dump(best_k, f, indent=0)

    # 关闭wandb
    import wandb
    if wandb.run:
        print_only("Closing wandb!")
        wandb.finish()


def test(cfg: DictConfig, ckpt_path: str = None):
    """
    测试RST深度伪造检测模型
    """
    # 实例化数据模块
    print_only(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()
    
    # 实例化RST模型
    print_only(f"Instantiating RST model <{cfg.rst_model._target_}>")
    model: torch.nn.Module = hydra.utils.instantiate(cfg.rst_model)
    
    # 实例化训练系统
    print_only(f"Instantiating system <{cfg.system._target_}>")
    system: LightningModule = hydra.utils.instantiate(
        cfg.system,
        model=model,
    )
    
    # 加载检查点
    if ckpt_path:
        print_only(f"Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        system.load_state_dict(checkpoint['state_dict'])
    
    # 实例化训练器
    print_only(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=[],
        logger=None,
    )
    
    # 开始测试
    print_only("=" * 60)
    print_only("Starting RST testing...")
    print_only("=" * 60)
    
    trainer.test(system, datamodule=datamodule)
    
    print_only("Testing finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Test RST Deepfake Detector")
    parser.add_argument(
        "--conf_dir",
        default="config/train_rst.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "test"],
        help="Mode: train or test",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Checkpoint path for testing",
    )
    
    args = parser.parse_args()
    cfg = OmegaConf.load(args.conf_dir)
    
    # 创建实验目录
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name), exist_ok=True)
    
    if args.mode == "train":
        train(cfg)
    else:
        test(cfg, args.ckpt)
