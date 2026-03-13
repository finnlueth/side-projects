"""PPO training loop as a PyTorch Lightning module."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv

from folding_rl.config import Config
from folding_rl.model.transformer import ProteinTransformer
from folding_rl.model.policy_head import PolicyHead
from folding_rl.model.value_head import ValueHead
from folding_rl.rl.rollout_buffer import RolloutBuffer
from folding_rl.rl.utils import compute_gae, normalize_advantages, explained_variance


def _obs_to_torch(
    obs: dict[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    """Convert numpy observation dict to torch tensors."""
    return {
        "residue_positions": torch.tensor(
            obs["residue_positions"], dtype=torch.int32, device=device
        ),
        "residue_types": torch.tensor(
            obs["residue_types"], dtype=torch.int32, device=device
        ),
        "grid_resolution": torch.tensor(
            obs["grid_resolution"], dtype=torch.int32, device=device
        ),
        "step_count": torch.tensor(
            obs["step_count"], dtype=torch.int32, device=device
        ),
    }


class PPOLightning(pl.LightningModule):
    """PyTorch Lightning module implementing PPO for protein folding.

    Training works by collecting rollouts via environment interaction in
    `training_step`, then performing minibatch PPO updates within the same step.

    Lightning's `manual_optimization` is used to allow the custom rollout-then-
    update loop that PPO requires.
    """

    def __init__(self, config: Config, env_factory):
        super().__init__()
        self.cfg = config
        self.env_factory = env_factory
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["env_factory"])

        # Model
        self.transformer = ProteinTransformer(
            n_residues=config.n_residues,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
        )
        self.policy_head = PolicyHead(d_model=config.d_model)
        self.value_head = ValueHead(d_model=config.d_model)

        # Environment (created lazily so it can be on CPU)
        self._envs: SyncVectorEnv | None = None
        self._obs: dict[str, np.ndarray] | None = None

        # Buffer (created lazily after we know device)
        self._buffer: RolloutBuffer | None = None

        # Episode tracking
        self._episode_rewards: list[float] = []
        self._episode_lddt: list[float] = []
        self._current_ep_rewards: np.ndarray | None = None

        # Step counter
        self._global_step_count = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _setup_envs(self) -> None:
        if self._envs is not None:
            return
        self._envs = SyncVectorEnv([self.env_factory for _ in range(self.cfg.num_envs)])
        obs, _ = self._envs.reset(seed=self.cfg.seed)
        self._obs = obs
        self._last_done = np.zeros(self.cfg.num_envs, dtype=np.float32)
        self._current_ep_rewards = np.zeros(self.cfg.num_envs, dtype=np.float32)

    def _setup_buffer(self) -> None:
        if self._buffer is not None:
            return
        self._buffer = RolloutBuffer(
            num_steps=self.cfg.num_steps,
            num_envs=self.cfg.num_envs,
            n_residues=self.cfg.n_residues,
            device=self.device,
        )

    def on_train_start(self) -> None:
        self._setup_envs()
        self._setup_buffer()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, obs: dict[str, torch.Tensor]):
        features = self.transformer(obs)
        return self.policy_head(features), self.value_head(features)

    def _get_value(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.transformer(obs)
        return self.value_head(features).squeeze(-1)  # (B,)

    def _get_action_and_value(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.transformer(obs)
        actions, log_probs, entropy = self.policy_head.sample_actions(features)
        values = self.value_head(features).squeeze(-1)
        return actions, log_probs, entropy, values

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _collect_rollout(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fill the buffer with num_steps of experience.

        Returns:
            next_value: (num_envs,) bootstrap value for GAE
            next_done:  (num_envs,) whether last step ended an episode
        """
        buf = self._buffer
        buf.ptr = 0  # reset write pointer

        for step in range(self.cfg.num_steps):
            obs_torch = _obs_to_torch(self._obs, self.device)

            actions, log_probs, _, values = self._get_action_and_value(obs_torch)

            # Environment step (numpy)
            actions_np = actions.cpu().numpy()
            next_obs, rewards, terminated, truncated, infos = self._envs.step(actions_np)
            dones = (terminated | truncated).astype(np.float32)

            # Track episode rewards
            self._current_ep_rewards += rewards
            for i, done in enumerate(dones):
                if done:
                    self._episode_rewards.append(float(self._current_ep_rewards[i]))
                    self._current_ep_rewards[i] = 0.0
                    # SyncVectorEnv stores info values directly with _key masks
                    if "_lddt" in infos and infos["_lddt"][i]:
                        self._episode_lddt.append(float(infos["lddt"][i]))

            buf.add(
                obs=obs_torch,
                actions=actions,
                log_probs=log_probs,
                values=values,
                rewards=torch.tensor(rewards, dtype=torch.float32, device=self.device),
                dones=torch.tensor(dones, dtype=torch.float32, device=self.device),
            )
            self._obs = next_obs
            self._last_done = dones
            self._global_step_count += self.cfg.num_envs

        # Bootstrap value for GAE
        next_obs_torch = _obs_to_torch(self._obs, self.device)
        next_value = self._get_value(next_obs_torch)
        next_done = torch.tensor(self._last_done, dtype=torch.float32, device=self.device)
        return next_value, next_done

    # ------------------------------------------------------------------
    # PPO Update
    # ------------------------------------------------------------------

    def _ppo_update(
        self,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> dict[str, float]:
        """Run PPO minibatch updates and return metrics."""
        cfg = self.cfg
        opt = self.optimizers()

        flat = self._buffer.get_flat()
        obs_flat = flat["obs"]
        actions_flat = flat["actions"]
        old_log_probs_flat = flat["log_probs"]
        old_values_flat = flat["values"]

        T, E = cfg.num_steps, cfg.num_envs
        batch_size = T * E
        minibatch_size = batch_size // cfg.num_minibatches

        pg_losses, v_losses, ent_losses, kl_divs, clip_fracs = [], [], [], [], []

        for _ in range(cfg.update_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start:start + minibatch_size]

                mb_obs = {k: v[mb_idx] for k, v in obs_flat.items()}
                mb_actions = actions_flat[mb_idx]
                mb_old_log_probs = old_log_probs_flat[mb_idx]
                mb_old_values = old_values_flat[mb_idx]
                mb_advantages = advantages.view(T * E)[mb_idx]
                mb_returns = returns.view(T * E)[mb_idx]

                # Recompute log_probs, values, entropy
                features = self.transformer(mb_obs)
                new_log_probs, entropy = self.policy_head.evaluate_actions(
                    features, mb_actions
                )
                new_values = self.value_head(features).squeeze(-1)

                # Policy loss (clipped)
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean()

                mb_adv = normalize_advantages(mb_advantages)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                v_loss_unclipped = (new_values - mb_returns) ** 2
                v_clipped = mb_old_values + (new_values - mb_old_values).clamp(
                    -cfg.clip_coef, cfg.clip_coef
                )
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                ent_loss = -entropy.mean()

                loss = pg_loss + cfg.vf_coef * v_loss + cfg.entropy_coef * ent_loss

                opt.zero_grad()
                self.manual_backward(loss)
                nn.utils.clip_grad_norm_(self.parameters(), cfg.max_grad_norm)
                opt.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(-ent_loss.item())
                kl_divs.append(approx_kl.item())
                clip_fracs.append(clip_frac.item())

        ev = explained_variance(returns.flatten(), old_values_flat)
        return {
            "loss/policy": float(np.mean(pg_losses)),
            "loss/value": float(np.mean(v_losses)),
            "loss/entropy": float(np.mean(ent_losses)),
            "diagnostics/approx_kl": float(np.mean(kl_divs)),
            "diagnostics/clip_fraction": float(np.mean(clip_fracs)),
            "diagnostics/explained_variance": ev,
        }

    # ------------------------------------------------------------------
    # Lightning training_step
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        """One PPO update = collect rollout + run PPO epochs."""
        # Collect rollout
        next_value, next_done = self._collect_rollout()

        # Compute GAE
        advantages, returns = compute_gae(
            rewards=self._buffer.rewards,
            values=self._buffer.values,
            dones=self._buffer.dones,
            next_value=next_value,
            next_done=next_done,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )

        # PPO update
        metrics = self._ppo_update(advantages, returns)

        # Step LR scheduler (manual_optimization requires explicit stepping)
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

        # Log
        self.log_dict(metrics, on_step=True, prog_bar=False)
        self.log(
            "global_step_count", float(self._global_step_count),
            on_step=True, prog_bar=True,
        )

        if self._episode_rewards:
            self.log("episode/mean_reward", float(np.mean(self._episode_rewards[-20:])),
                     on_step=True, prog_bar=True)
            self._episode_rewards.clear()
        if self._episode_lddt:
            self.log("episode/mean_lddt", float(np.mean(self._episode_lddt[-20:])),
                     on_step=True, prog_bar=True)
            self._episode_lddt.clear()

        # Return dummy loss (required by Lightning even with manual optimization)
        return torch.tensor(metrics["loss/policy"])

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate, eps=1e-5)
        # Cosine annealing with floor at 10% of initial LR
        total_updates = self.cfg.total_timesteps // (self.cfg.num_steps * self.cfg.num_envs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_updates),
            eta_min=self.cfg.learning_rate * 0.1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
