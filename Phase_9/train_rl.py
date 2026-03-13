
import logging
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from config import (
    PPO_LEARNING_RATE, PPO_N_STEPS, PPO_BATCH_SIZE,
    PPO_N_EPOCHS, PPO_GAMMA, PPO_GAE_LAMBDA,
    PPO_CLIP_RANGE, PPO_ENT_COEF, PPO_VF_COEF,
    PPO_MAX_GRAD_NORM, TOTAL_TIMESTEPS,
    CKPT_DIR,
)
from trading_env import TradingEnv

log = logging.getLogger(__name__)


class LoggingCallback(BaseCallback):

    def __init__(self, log_freq=10_000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_rew = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                mean_len = np.mean([ep["l"] for ep in self.model.ep_info_buffer])
                log.info(
                    f"  Step {self.n_calls:>7d}/{TOTAL_TIMESTEPS} | "
                    f"Mean Reward: {mean_rew:>8.2f} | "
                    f"Mean Ep Len: {mean_len:>6.0f}"
                )
        return True


def train_ppo(train_data, eval_data=None):
    log.info(f"Creating training environment...")
    log.info(f"  Tickers: {len(train_data)}")
    total_days = sum(len(df) for df in train_data.values())
    log.info(f"  Total training steps available: {total_days}")

    def make_env():
        return TradingEnv(train_data, mode="train")

    env = DummyVecEnv([make_env])

    log.info(f"\n  PPO Configuration:")
    log.info(f"    Learning Rate:  {PPO_LEARNING_RATE}")
    log.info(f"    N Steps:        {PPO_N_STEPS}")
    log.info(f"    Batch Size:     {PPO_BATCH_SIZE}")
    log.info(f"    Epochs:         {PPO_N_EPOCHS}")
    log.info(f"    Gamma:          {PPO_GAMMA}")
    log.info(f"    Clip Range:     {PPO_CLIP_RANGE}")
    log.info(f"    Entropy Coef:   {PPO_ENT_COEF}")
    log.info(f"    Total Steps:    {TOTAL_TIMESTEPS}")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=PPO_LEARNING_RATE,
        n_steps=min(PPO_N_STEPS, total_days),
        batch_size=PPO_BATCH_SIZE,
        n_epochs=PPO_N_EPOCHS,
        gamma=PPO_GAMMA,
        gae_lambda=PPO_GAE_LAMBDA,
        clip_range=PPO_CLIP_RANGE,
        ent_coef=PPO_ENT_COEF,
        vf_coef=PPO_VF_COEF,
        max_grad_norm=PPO_MAX_GRAD_NORM,
        verbose=0,
        device="auto",
        policy_kwargs={
            "net_arch": dict(pi=[128, 64], vf=[128, 64]),
        },
    )

    log.info(f"\n  Training PPO agent...")
    callback = LoggingCallback(log_freq=10_000)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    save_path = CKPT_DIR / "ppo_trading_agent"
    model.save(str(save_path))
    log.info(f"  Model saved: {save_path}")

    env.close()
    return model


def evaluate_agent(model, eval_data):
    env = TradingEnv(eval_data, mode="eval")
    obs, _ = env.reset()

    records = []
    total_reward = 0
    step = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if step < len(env.index):
            t_idx, d_idx = env.index[step]
            ticker = env.tickers[t_idx]
            row = env.data[ticker].iloc[d_idx]

            records.append({
                "date": row["date"],
                "ticker": ticker,
                "position": info.get("position", 0),
                "actual_return": info.get("actual_return", 0),
                "reward": float(reward),
                "portfolio_value": info.get("portfolio_value", 1.0),
                "prob_bull": row["prob_bull"],
                "confidence": row["confidence"],
                "uncertainty": row["uncertainty"],
            })

        total_reward += reward
        step += 1

        if terminated or truncated:
            break

    log.info(f"  Evaluation: {step} steps, total reward: {total_reward:.2f}")
    return records

