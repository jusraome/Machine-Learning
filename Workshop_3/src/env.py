"""Gymnasium environment: controllable text generation with the Workshop 2 LSTM.

At every step the Workshop 2 LSTM proposes its K = 5 most likely next
characters and the agent chooses one of them. The goal is to write real
Shakespeare words without repeating them (the LSTM alone, choosing always the
most likely character, falls into loops like "I will be but the prize. I will
be but the prize").

Reward (given when a word ends, i.e. when a space or punctuation is chosen):
    +1  the word exists in the Shakespeare dictionary and is new in the episode
     0  the word exists but was already written in this episode
    -1  the word does not exist (invented word)

Run `python -m src.env` to validate the environment with 100 random episodes.
"""

import random
import re
import string
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from torch import nn

SEED = 42
ROOT = Path(__file__).resolve().parents[1]  # Workshop_3/
WORKSHOP1 = ROOT.parent / "Workshop_1"
LSTM_CHECKPOINT = ROOT.parent / "Workshop_2" / "checkpoints" / "model_best.pt"
WORDS_FILE = ROOT / "data" / "words.txt"
FIGURES_DIR = ROOT / "reports" / "figures"

WORD_CHARS = set(string.ascii_letters + "'")  # characters that belong to a word
K = 5  # number of candidate characters (= number of actions)
EPISODE_LEN = 100  # characters generated per episode


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- data
def load_contexts(split, n):
    """n fixed random 80-character windows from a Workshop 1 split (train/val/test)."""
    df = pd.read_parquet(WORKSHOP1 / "data" / "processed" / f"{split}.parquet", columns=["x"])
    idx = np.random.default_rng(SEED).choice(len(df), size=n, replace=False)
    return df["x"].iloc[idx].tolist()


def load_dictionary():
    """Set of Shakespeare words (lowercase) taken from the TRAIN split, and all their prefixes."""
    if not WORDS_FILE.exists():
        df = pd.read_parquet(WORKSHOP1 / "data" / "processed" / "train.parquet", columns=["x"])
        words = set()
        for window in df["x"].iloc[::40]:  # windows slide 1 char, so every 40th covers the text
            tokens = re.findall(r"[A-Za-z']+", window)[1:-1]  # first/last may be cut
            words.update(t.lower() for t in tokens)
        WORDS_FILE.parent.mkdir(parents=True, exist_ok=True)
        WORDS_FILE.write_text("\n".join(sorted(words)), encoding="utf-8")
    words = set(WORDS_FILE.read_text(encoding="utf-8").split())
    prefixes = {w[:i] for w in words for i in range(1, len(w) + 1)}
    return words, prefixes


# ---------------------------------------------------------------- Workshop 2 language model
class CharLSTM(nn.Module):
    """Same architecture as Workshop 2 (only used for inference here)."""

    def __init__(self, vocab_size, embedding_dim=8, hidden_size=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.ModuleList(
            [nn.LSTM(embedding_dim if i == 0 else hidden_size, hidden_size, batch_first=True)
             for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        out = self.embedding(x)
        for layer in self.lstm:
            out, (h_n, _) = layer(out)
        return self.fc(h_n[-1])


def load_language_model(device):
    ckpt = torch.load(LSTM_CHECKPOINT, map_location="cpu", weights_only=False)
    model = CharLSTM(**ckpt["model_hparams"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["vocab"]


# ---------------------------------------------------------------- environment
class TextGenerationEnv(gym.Env):
    """State (22 numbers): for each of the 5 candidates [probability, ends a word,
    keeps the word valid, repeats a word] + [length of current word / 10, step / 100].
    Actions: 0..4 = choose the candidate with that rank (0 = most likely)."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, contexts, device=None, render_mode=None):
        super().__init__()
        self.contexts = contexts
        self.device = device or get_device()
        self.render_mode = render_mode
        self.lm, self.vocab = load_language_model(self.device)
        self.words, self.prefixes = load_dictionary()
        self.action_space = spaces.Discrete(K)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(4 * K + 2,), dtype=np.float32)

    # --- helpers
    @torch.no_grad()
    def _update_candidates(self):
        """Ask the LSTM for its top-5 next characters (never <UNK>)."""
        text = (self.context + self.generated)[-80:]
        ids = torch.tensor([[self.vocab["stoi"].get(c, 0) for c in text]], device=self.device)
        probs = self.lm(ids).softmax(-1)[0]
        probs[self.vocab["unk_index"]] = 0
        values, indices = probs.topk(K)
        self.cand_chars = [self.vocab["itos"][i] for i in indices.tolist()]
        self.cand_probs = values.cpu().numpy()

    def _obs(self):
        features = []
        word = self.word.lower()
        for char, p in zip(self.cand_chars, self.cand_probs):
            ends_word = char not in WORD_CHARS
            if ends_word:
                valid = word == "" or word in self.words
                repeated = word != "" and word in self.used
            else:
                valid = (word + char.lower()) in self.prefixes
                repeated = False
            features += [p, float(ends_word), float(valid), float(repeated)]
        features += [min(len(word) / 10, 1.0), self.steps / EPISODE_LEN]
        return np.array(features, dtype=np.float32)

    def _score_word(self):
        word = self.word.lower()
        if word not in self.words:
            self.stats["invalid"] += 1
            return -1.0
        if word in self.used:
            self.stats["repeated"] += 1
            return 0.0
        self.used.add(word)
        self.stats["new"] += 1
        return 1.0

    # --- gymnasium API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and "index" in options:  # fixed context (validation / test)
            index = options["index"]
        else:
            index = int(self.np_random.integers(len(self.contexts)))
        self.context = self.contexts[index]
        self.generated, self.word, self.used, self.steps = "", "", set(), 0
        self.stats = {"new": 0, "repeated": 0, "invalid": 0}
        self.last_reward = 0.0
        self._update_candidates()
        return self._obs(), {}

    def step(self, action):
        char = self.cand_chars[action]
        log_prob = float(np.log(self.cand_probs[action] + 1e-12))
        reward = 0.0
        if char in WORD_CHARS:
            self.word += char
        elif self.word:
            reward = self._score_word()  # a word just ended
            self.word = ""
        self.generated += char
        self.steps += 1
        self.last_reward = reward
        terminated = self.steps >= EPISODE_LEN
        if not terminated:
            self._update_candidates()
        if self.render_mode == "human":
            self.render()
        info = {"char": char, "log_prob": log_prob, "stats": dict(self.stats)}
        return self._obs(), reward, terminated, False, info

    def render(self):
        print(f"...{self.context[-30:]!r} + {self.generated!r}  (last reward {self.last_reward:+.0f})")


# ---------------------------------------------------------------- validation with a random policy
def main():
    set_seeds()
    env = TextGenerationEnv(load_contexts("val", 100))
    env.action_space.seed(SEED)
    returns, lengths, actions = [], [], []
    for episode in range(100):
        obs, _ = env.reset(seed=SEED + episode)
        done, total, length = False, 0.0, 0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            total += reward
            length += 1
            actions.append(action)
        returns.append(total)
        lengths.append(length)
        if episode < 3:
            env.render()

    counts = np.bincount(actions, minlength=K)
    print(f"\n100 random episodes")
    print(f"  mean return:   {np.mean(returns):.2f} (std {np.std(returns):.2f}, "
          f"min {min(returns):.0f}, max {max(returns):.0f})")
    print(f"  episode length: min {min(lengths)}, max {max(lengths)}, mean {np.mean(lengths):.1f}")
    print(f"  action frequency: {dict(enumerate((counts / counts.sum()).round(3)))}")
    print(f"  observation shape: {obs.shape}, all inside [0, 1]: {bool(((obs >= 0) & (obs <= 1)).all())}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(returns, bins=15)
    axes[0].set_title("Random policy: episode return")
    axes[0].set_xlabel("return")
    axes[1].bar(range(K), counts / counts.sum())
    axes[1].set_title("Random policy: action frequency")
    axes[1].set_xlabel("action (rank of the chosen candidate)")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "random_policy.png", dpi=110)


if __name__ == "__main__":
    main()
