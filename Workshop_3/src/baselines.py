"""Baseline strategies, evaluated on the same contexts as the agent.

- random: chooses one of the 5 candidates uniformly at random.
- greedy LSTM: always chooses the most likely character. This is the Workshop 2
  model used on its own, without RL.
- sampling LSTM: samples a candidate with the LSTM probabilities (renormalised
  over the top 5).
"""

import numpy as np

from src.env import K


def random_policy(obs, rng):
    return int(rng.integers(K))


def greedy_policy(obs, rng):
    return 0  # candidates are sorted by probability, so action 0 = most likely


def sampling_policy(obs, rng):
    probs = obs[0:4 * K:4]  # the probability is the first feature of every candidate
    return int(rng.choice(K, p=probs / probs.sum()))


BASELINES = {"random": random_policy, "greedy LSTM (W2)": greedy_policy,
             "sampling LSTM (W2)": sampling_policy}


def run_episodes(env, policy, n_episodes, seed=42):
    """Run one episode per context (index 0..n-1) and return the metrics."""
    rng = np.random.default_rng(seed)
    returns, log_probs, stats = [], [], {"new": 0, "repeated": 0, "invalid": 0}
    texts = []
    for i in range(n_episodes):
        obs, _ = env.reset(options={"index": i})
        done, total = False, 0.0
        while not done:
            obs, reward, done, _, info = env.step(policy(obs, rng))
            total += reward
            log_probs.append(info["log_prob"])
        returns.append(total)
        texts.append(env.generated)
        for key in stats:
            stats[key] += info["stats"][key]
    words = sum(stats.values())
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "new_word_rate": stats["new"] / words,       # real and not repeated
        "repeated_word_rate": stats["repeated"] / words,
        "invalid_word_rate": stats["invalid"] / words,
        "words_per_episode": words / n_episodes,
        "mean_log_prob": float(np.mean(log_probs)),  # fluency according to the LSTM
        "texts": texts,
    }
