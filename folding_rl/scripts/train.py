"""Main training script for protein folding RL."""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from folding_rl.config import Config
from folding_rl.env.protein_env import ProteinFoldingEnv
from folding_rl.rl.ppo import PPOLightning


def make_env(config: Config):
    """Factory function for a single environment (used with SyncVectorEnv)."""
    def _make():
        return ProteinFoldingEnv(config=config)
    return _make


class DummyDataModule(pl.LightningDataModule):
    """Provides a dummy dataloader — PPO collects its own data internally.

    Each 'batch' is a single integer index; the actual data comes from the
    environment via the PPO rollout buffer.
    """

    def __init__(self, total_updates: int):
        super().__init__()
        self.total_updates = total_updates

    def train_dataloader(self):
        # Dataset of dummy indices, one per PPO update
        dataset = TensorDataset(torch.arange(self.total_updates))
        return DataLoader(dataset, batch_size=1, shuffle=False)


def main():
    parser = argparse.ArgumentParser(description="Train protein folding RL agent")
    _defaults = Config()
    parser.add_argument("--total-timesteps", type=int, default=_defaults.total_timesteps)
    parser.add_argument("--num-envs", type=int, default=_defaults.num_envs)
    parser.add_argument("--num-steps", type=int, default=_defaults.num_steps)
    parser.add_argument("--learning-rate", type=float, default=_defaults.learning_rate)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true",
                        help="Run only 1 update for debugging")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    # Config
    cfg = Config(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        use_wandb=not args.no_wandb,
    )

    pl.seed_everything(cfg.seed)

    total_updates = cfg.total_timesteps // (cfg.num_steps * cfg.num_envs)

    # Data
    dm = DummyDataModule(total_updates)

    # Model
    model = PPOLightning(
        config=cfg,
        env_factory=make_env(cfg),
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename="ppo-{step}-{episode/mean_lddt:.3f}",
            save_top_k=3,
            monitor="episode/mean_lddt",
            mode="max",
            save_last=True,
            every_n_train_steps=cfg.save_interval,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Logger
    logger = None
    if cfg.use_wandb and not args.fast_dev_run:
        logger = WandbLogger(
            project=cfg.wandb_project,
            name=f"ppo-trpcage-seed{cfg.seed}",
        )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=total_updates,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.log_interval,
        enable_progress_bar=True,
        fast_dev_run=args.fast_dev_run,
        accelerator="auto",
        devices=1,
    )

    trainer.fit(model, datamodule=dm)
    print("Training complete.")


if __name__ == "__main__":
    main()
