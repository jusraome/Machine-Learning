"""Evaluate the best agent and the baselines on 100 fixed TEST contexts.

    python -m src.evaluate

Test contexts come from the Workshop 1 test split (speakers never seen in
training). Outputs: reports/test_comparison.csv, reports/trajectories.txt and
reports/figures/trajectories.png.
"""

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.agent import DQNAgent
from src.baselines import BASELINES, run_episodes
from src.env import FIGURES_DIR, TextGenerationEnv, get_device, load_contexts, set_seeds
from src.train import CHECKPOINT, REPORTS_DIR

N_TEST_CONTEXTS = 100


def load_agent(env, device):
    ckpt = torch.load(CHECKPOINT, weights_only=False)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, device=device)
    agent.q.load_state_dict(ckpt["state_dict"])
    return agent, ckpt


def trajectory(env, agent, index):
    """Play one test episode with the agent and record every step."""
    obs, _ = env.reset(options={"index": index})
    steps, done, n_words = [], False, 0
    while not done:
        action = agent.act(obs, 0.0)
        obs, reward, done, _, info = env.step(action)
        word_ended = sum(info["stats"].values()) > n_words  # a word was scored in this step
        n_words = sum(info["stats"].values())
        steps.append({"char": info["char"], "action": action, "reward": reward,
                      "word_ended": word_ended})
    return env.context, steps


def plot_trajectories(trajectories):
    """Action chosen at every step and reward received when a word ends."""
    fig, axes = plt.subplots(len(trajectories), 1, figsize=(14, 3.2 * len(trajectories)))
    colors = {1.0: "green", 0.0: "orange", -1.0: "red"}
    for ax, (context, steps) in zip(axes, trajectories):
        ax.step(range(len(steps)), [s["action"] for s in steps], where="mid", color="steelblue",
                label="action (rank of chosen character)")
        for t, s in enumerate(steps):
            if s["word_ended"]:  # reward marker: green +1, orange 0, red -1
                ax.scatter(t, 4.6, color=colors[s["reward"]], s=40, zorder=3)
        text = "".join(s["char"] for s in steps)
        ax.set_title(textwrap.fill(f"...{context[-40:]} || {text}", 140), fontsize=9, loc="left")
        ax.set_yticks(range(5))
        ax.set_ylim(-0.5, 5)
        ax.set_xlabel("step")
    axes[0].scatter([], [], color="green", label="+1 new real word")
    axes[0].scatter([], [], color="orange", label="0 repeated word")
    axes[0].scatter([], [], color="red", label="-1 invented word")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(FIGURES_DIR / "trajectories.png", dpi=110)
    plt.close(fig)


def main():
    set_seeds()
    device = get_device()
    env = TextGenerationEnv(load_contexts("test", N_TEST_CONTEXTS), device)
    agent, ckpt = load_agent(env, device)
    print(f"agent: config {ckpt['config_name']}, episode {ckpt['episode']}, "
          f"val return {ckpt['val_return']:.2f}")

    policies = dict(BASELINES)
    policies["DQN agent (W3)"] = lambda obs, rng: agent.act(obs, 0.0)
    results = {name: run_episodes(env, policy, N_TEST_CONTEXTS) for name, policy in policies.items()}

    table = pd.DataFrame([{"policy": name, **{k: v for k, v in r.items() if k != "texts"}}
                          for name, r in results.items()])
    table.to_csv(REPORTS_DIR / "test_comparison.csv", index=False)
    print("\n" + table.round(3).to_string(index=False))

    # sample trajectories: same 3 test contexts, greedy LSTM vs agent
    lines = []
    for i in range(3):
        lines.append(f"Context: ...{env.contexts[i][-60:]!r}")
        lines.append(f"  greedy LSTM (W2): {results['greedy LSTM (W2)']['texts'][i]!r}")
        lines.append(f"  DQN agent (W3):   {results['DQN agent (W3)']['texts'][i]!r}\n")
    (REPORTS_DIR / "trajectories.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    plot_trajectories([trajectory(env, agent, i) for i in range(3)])


if __name__ == "__main__":
    main()
