"""Train the DQN agent with 4 hyperparameter configurations.

    python -m src.train                  # all configurations
    python -m src.train --config base    # only one

Episodes start from random TRAIN contexts. Every 50 episodes the agent is
evaluated without exploration (epsilon = 0) on 30 fixed VALIDATION contexts.
The checkpoint with the highest validation return over all configurations is
saved to checkpoints/agent_best.pt. Logs go to TensorBoard (runs/workshop3/).
"""

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from src.agent import DQNAgent
from src.baselines import BASELINES, run_episodes
from src.env import FIGURES_DIR, ROOT, SEED, TextGenerationEnv, get_device, load_contexts, set_seeds

CHECKPOINT = ROOT / "checkpoints" / "agent_best.pt"
REPORTS_DIR = ROOT / "reports"
RUNS_DIR = ROOT / "runs" / "workshop3"

N_TRAIN_CONTEXTS = 20_000
N_VAL_CONTEXTS = 30
EVAL_EVERY = 50  # episodes
WARMUP_STEPS = 500  # random steps in the buffer before learning starts
TARGET_UPDATE = 500  # steps between target network updates
EPS_MIN = 0.05

# Hyperparameter search: change one thing at a time with respect to "base".
CONFIGS = {
    "base":        {"lr": 1e-3, "gamma": 0.95, "eps_decay": 0.995},
    "low_lr":      {"lr": 1e-4, "gamma": 0.95, "eps_decay": 0.995},
    "high_gamma":  {"lr": 1e-3, "gamma": 0.99, "eps_decay": 0.995},
    "fast_decay":  {"lr": 1e-3, "gamma": 0.95, "eps_decay": 0.990},
}


def evaluate_agent(agent, env, n):
    """Mean return of the greedy policy (no exploration) on n fixed contexts."""
    return run_episodes(env, lambda obs, rng: agent.act(obs, 0.0), n)["mean_return"]


def train(name, config, episodes, train_env, val_env, device):
    set_seeds()
    agent = DQNAgent(train_env.observation_space.shape[0], train_env.action_space.n,
                     lr=config["lr"], gamma=config["gamma"], device=device)
    writer = SummaryWriter(RUNS_DIR / name)
    epsilon, steps, history = 1.0, 0, []
    best = {"val_return": -np.inf}
    start = time.time()

    for episode in range(1, episodes + 1):
        obs, _ = train_env.reset(seed=SEED if episode == 1 else None)
        done, total, losses = False, 0.0, []
        while not done:
            action = agent.act(obs, epsilon)
            next_obs, reward, done, _, _ = train_env.step(action)
            agent.buffer.add(obs, action, reward, next_obs, done)
            obs, total, steps = next_obs, total + reward, steps + 1
            if steps >= WARMUP_STEPS:
                losses.append(agent.update())
            if steps % TARGET_UPDATE == 0:
                agent.update_target()
        epsilon = max(EPS_MIN, epsilon * config["eps_decay"])  # exploration decay

        row = {"config": name, "episode": episode, "steps": steps, "return": total,
               "epsilon": epsilon, "loss": np.mean(losses) if losses else np.nan,
               "val_return": np.nan}
        writer.add_scalar("train/return", total, episode)
        writer.add_scalar("train/epsilon", epsilon, episode)
        writer.add_scalar("train/episode_length", train_env.steps, episode)
        if losses:
            writer.add_scalar("train/loss", row["loss"], episode)

        if episode % EVAL_EVERY == 0:
            row["val_return"] = evaluate_agent(agent, val_env, N_VAL_CONTEXTS)
            writer.add_scalar("val/return", row["val_return"], episode)
            writer.add_scalar("val/return_vs_steps", row["val_return"], steps)
            print(f"[{name}] episode {episode}  steps {steps}  epsilon {epsilon:.3f}  "
                  f"val return {row['val_return']:.2f}")
            if row["val_return"] > best["val_return"]:
                best = {"val_return": row["val_return"], "episode": episode, "steps": steps,
                        "state_dict": {k: v.cpu().clone() for k, v in agent.q.state_dict().items()}}
        history.append(row)

    writer.close()
    best["minutes"] = (time.time() - start) / 60
    return best, history


def save_if_best(name, config, best):
    """Keep in checkpoints/agent_best.pt the agent with the highest validation return."""
    if CHECKPOINT.exists():
        current = torch.load(CHECKPOINT, weights_only=False)["val_return"]
        if best["val_return"] <= current:
            return
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best["state_dict"], "config_name": name, "config": config,
                "val_return": best["val_return"], "episode": best["episode"]}, CHECKPOINT)
    print(f"saved {name} to agent_best.pt (val return {best['val_return']:.2f})")


def plot_curves(history, baselines_val):
    """Learning curves of all configurations and exploration (epsilon)."""
    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for name, g in h.groupby("config", sort=False):
        axes[0].plot(g["episode"], g["return"].rolling(50, min_periods=1).mean(), label=name)
        v = g.dropna(subset=["val_return"])
        axes[1].plot(v["steps"], v["val_return"], "o-", label=name)
        axes[2].plot(g["episode"], g["epsilon"], label=name)
    for (label, value), style in zip(baselines_val.items(), [":", "--", "-."]):
        axes[1].axhline(value, color="gray", linestyle=style, label=label)
    axes[0].set(title="Training return (moving average of 50 episodes)", xlabel="episode")
    axes[1].set(title="Validation return vs environment steps", xlabel="environment steps")
    axes[2].set(title="Exploration rate (epsilon)", xlabel="episode")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "learning_curves.png", dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIGS), default=None, help="default: all")
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()

    set_seeds()
    device = get_device()
    train_env = TextGenerationEnv(load_contexts("train", N_TRAIN_CONTEXTS), device)
    val_env = TextGenerationEnv(load_contexts("val", N_VAL_CONTEXTS), device)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # baselines on the same validation contexts (horizontal lines in the plot)
    baselines_val = {name: run_episodes(val_env, policy, N_VAL_CONTEXTS)["mean_return"]
                     for name, policy in BASELINES.items()}
    print("validation baselines:", {k: round(v, 2) for k, v in baselines_val.items()})

    names = [args.config] if args.config else list(CONFIGS)
    summary, history = [], []
    for name in names:
        best, hist = train(name, CONFIGS[name], args.episodes, train_env, val_env, device)
        save_if_best(name, CONFIGS[name], best)
        history += hist
        # first number of steps where validation return beats the greedy LSTM
        beat = [r["steps"] for r in hist
                if r["val_return"] > baselines_val["greedy LSTM (W2)"]]
        summary.append({"config": name, **CONFIGS[name],
                        "best_val_return": best["val_return"], "best_episode": best["episode"],
                        "steps_to_beat_greedy": beat[0] if beat else None,
                        "minutes": round(best["minutes"], 1)})

    table = pd.DataFrame(summary)
    print("\n" + table.to_string(index=False))
    if args.config is None:  # full search: save the table and the curves
        table.to_csv(REPORTS_DIR / "hparam_search.csv", index=False)
        pd.DataFrame(history).to_csv(REPORTS_DIR / "training_history.csv", index=False)
        plot_curves(history, baselines_val)


if __name__ == "__main__":
    main()
