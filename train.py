"""
train.py — Fixed training with reward normalization and stable critic
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.envs.registration import register
from collections import deque

register(
    id="FieldDispatch-v0",
    entry_point="env:FieldDispatchEnv",
)

# ── Hyperparameters ────────────────────────────────────────────────────────
NUM_TECHS  = 5
NUM_JOBS   = 3
EPISODES   = 5000
GAMMA      = 0.99
LR_ACTOR   = 0.0003
LR_CRITIC  = 0.001     # critic learns faster than actor
EVAL_EVERY = 500
EVAL_RUNS  = 30

env = gym.make("FieldDispatch-v0", num_techs=NUM_TECHS, num_jobs=NUM_JOBS)

INPUT_SIZE  = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n

print(f"Observation size : {INPUT_SIZE}")
print(f"Action size      : {OUTPUT_SIZE}")
print(f"Training for up to {EPISODES} episodes...")


# ═══════════════════════════════════════════════════════════════════════════
# Reward normalizer — keeps rewards in a stable range
# ═══════════════════════════════════════════════════════════════════════════
class RunningNormalizer:
    """Normalizes rewards using running mean and std."""
    def __init__(self, clip=5.0):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = 0
        self.clip  = clip

    def update(self, x):
        self.count += 1
        old_mean    = self.mean
        self.mean  += (x - self.mean) / self.count
        self.var   += (x - old_mean) * (x - self.mean)

    def normalize(self, x):
        std = np.sqrt(self.var / max(self.count, 1)) + 1e-8
        return float(np.clip((x - self.mean) / std, -self.clip, self.clip))


reward_normalizer = RunningNormalizer(clip=5.0)


# ═══════════════════════════════════════════════════════════════════════════
# Networks
# ═══════════════════════════════════════════════════════════════════════════
class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 128),        nn.ReLU(),
            nn.Linear(128, 64),         nn.ReLU(),
            nn.Linear(64, output_size),
        )
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


class Critic(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 128),        nn.ReLU(),
            nn.Linear(128, 64),         nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


actor  = Actor(INPUT_SIZE, OUTPUT_SIZE)
critic = Critic(INPUT_SIZE)

actor_optimizer  = optim.Adam(actor.parameters(),  lr=LR_ACTOR)
critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)

# Reduce LR every 1000 episodes
actor_scheduler  = optim.lr_scheduler.StepLR(actor_optimizer,  step_size=1000, gamma=0.7)
critic_scheduler = optim.lr_scheduler.StepLR(critic_optimizer, step_size=1000, gamma=0.7)


# ── Action selection ───────────────────────────────────────────────────────
def select_action(state):
    state_t      = torch.tensor(state, dtype=torch.float32)
    action_probs = actor(state_t)
    dist         = torch.distributions.Categorical(action_probs)
    action       = dist.sample()
    return action.item(), dist.log_prob(action)


# ── Loss with normalized reward ────────────────────────────────────────────
def calculate_losses(log_prob, reward, state, next_state, done):
    # Normalize reward before computing losses
    reward_normalizer.update(reward)
    norm_reward = reward_normalizer.normalize(reward)

    s  = torch.tensor(state,      dtype=torch.float32)
    s_ = torch.tensor(next_state, dtype=torch.float32)
    r  = torch.tensor(norm_reward, dtype=torch.float32)
    d  = torch.tensor(done,        dtype=torch.float32)

    value = critic(s)
    with torch.no_grad():
        next_value = critic(s_)

    td_target   = r + (1 - d) * GAMMA * next_value
    td_error    = td_target - value
    actor_loss  = -log_prob * td_error.detach()
    critic_loss = td_error.pow(2)
    return actor_loss, critic_loss


# ── Evaluation helpers ─────────────────────────────────────────────────────
def run_episode(use_actor=True):
    state, _ = env.reset()
    done, total = False, 0.0
    while not done:
        if use_actor:
            action, _ = select_action(state)
        else:
            action = env.action_space.sample()
        state, r, term, trunc, _ = env.step(action)
        total += r
        done = term or trunc
    return total

def run_baseline(n=EVAL_RUNS):
    return float(np.mean([run_episode(use_actor=False) for _ in range(n)]))

def bellman_consistency(n=EVAL_RUNS):
    td_errors = []
    actor.eval(); critic.eval()
    with torch.no_grad():
        for _ in range(n):
            state, _ = env.reset()
            done = False
            while not done:
                action, _ = select_action(state)
                next_state, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                s  = torch.tensor(state,       dtype=torch.float32)
                s_ = torch.tensor(next_state,  dtype=torch.float32)
                # Use normalized reward for Bellman check
                reward_normalizer.update(reward)
                nr = reward_normalizer.normalize(reward)
                r  = torch.tensor(nr,          dtype=torch.float32)
                d  = torch.tensor(float(done), dtype=torch.float32)
                delta = r + (1 - d) * GAMMA * critic(s_) - critic(s)
                td_errors.append(abs(delta.item()))
                state = next_state
    actor.train(); critic.train()
    return float(np.mean(td_errors)) if td_errors else 0.0

def regret_analysis(n=EVAL_RUNS):
    J_ref = run_baseline(n)
    actor.eval()
    rewards = [run_episode(use_actor=True) for _ in range(n)]
    actor.train()
    J_pi   = float(np.mean(rewards))
    return J_ref, J_pi, J_ref - J_pi


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════
episode_rewards = []
eval_log        = []
best_reward     = -np.inf
best_regret     = np.inf

print("\n── Training ─────────────────────────────────────────────────")

for episode in range(1, EPISODES + 1):
    state, _  = env.reset()
    done      = False
    ep_reward = 0.0

    while not done:
        action, log_prob = select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        actor_loss, critic_loss = calculate_losses(
            log_prob, reward, state, next_state, done
        )

        actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_optimizer.step()

        state      = next_state
        ep_reward += reward

    episode_rewards.append(ep_reward)
    actor_scheduler.step()
    critic_scheduler.step()

    # Save best model based on rolling average
    if episode >= 50:
        avg_50 = float(np.mean(episode_rewards[-50:]))
        if avg_50 > best_reward:
            best_reward = avg_50
            torch.save(actor.state_dict(),  "actor_best.pth")
            torch.save(critic.state_dict(), "critic_best.pth")

    if episode % 100 == 0:
        avg_50 = float(np.mean(episode_rewards[-50:]))
        print(f"Episode {episode:4d} | Avg(50): {avg_50:8.2f} | Best: {best_reward:8.2f}")

    # ── Evaluation ────────────────────────────────────────────────────────
    if episode % EVAL_EVERY == 0:
        bc             = bellman_consistency()
        J_ref, J_pi, regret = regret_analysis()

        eval_log.append({
            "episode":          episode,
            "J_ref":            round(J_ref, 3),
            "J_learned":        round(J_pi, 3),
            "regret":           round(regret, 3),
            "bellman_residual": round(bc, 4),
            "avg_reward_50":    round(float(np.mean(episode_rewards[-50:])), 3),
        })

        status = "✅ outperforms baseline" if regret < 0 else "⚠️  below baseline"
        print(f"\n  📊 Evaluation @ Episode {episode}")
        print(f"     Bellman Residual  : {bc:.4f}  {'✅ stable' if bc < 2.0 else '⚠️ unstable'}")
        print(f"     Baseline reward   : {J_ref:.2f}")
        print(f"     Learned reward    : {J_pi:.2f}")
        print(f"     Regret R(π)       : {regret:.2f}  {status}\n")

        # Save best regret model separately
        if regret < best_regret:
            best_regret = regret
            torch.save(actor.state_dict(), "actor_best_regret.pth")

        # Early stop: stable AND outperforming
        if regret < -5.0 and bc < 2.0:
            print(f"🎯 Converged at episode {episode}!")
            break

# ── Save models ────────────────────────────────────────────────────────────
import os, shutil
torch.save(actor.state_dict(),  "actor.pth")
torch.save(critic.state_dict(), "critic.pth")

# Use best regret model if available, else best reward model
if os.path.exists("actor_best_regret.pth"):
    shutil.copy("actor_best_regret.pth", "actor.pth")
    print("✅ Best regret model saved as actor.pth")
elif os.path.exists("actor_best.pth"):
    shutil.copy("actor_best.pth", "actor.pth")
    print("✅ Best reward model saved as actor.pth")

# Save eval log
if eval_log:
    pd.DataFrame(eval_log).to_csv("eval_log.csv", index=False)
    print("✅ eval_log.csv saved")

# Final summary
bc_final            = bellman_consistency(50)
J_ref, J_pi, regret = regret_analysis(50)

print("\n── Final Evaluation ─────────────────────────────────────────")
print(f"  Best avg reward        : {best_reward:.2f}")
print(f"  Best regret achieved   : {best_regret:.2f}")
print(f"  Final Bellman Residual : {bc_final:.4f}")
print(f"  Final Baseline reward  : {J_ref:.2f}")
print(f"  Final Learned reward   : {J_pi:.2f}")
print(f"  Final Regret           : {regret:.2f}")
print(f"  Result: {'✅ Outperforms baseline' if regret < 0 else '⚠️ Below baseline'}")
