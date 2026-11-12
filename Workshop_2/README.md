# Workshop 2 — LSTM for next-character prediction

Dataset: `flwrlabs/shakespeare` (LEAF). Each sample is a window of 80 characters (`x`) and the next
character (`y`), 76 classes (75 characters + `<UNK>`), with strong class imbalance (the space is
~19% of the targets).

We reuse the data from Workshop 1 without changing it: the split by speaker
(`Workshop_1/data/splits.json`, 70/15/15) and the vocabulary built on train
(`Workshop_1/data/vocab.json`). The test split is only used in `src/evaluate.py`.

## Model (`src/model.py`)

Character-level LSTM, the reference model of LEAF for this dataset:

| Layer | Output shape | Parameters |
|---|---|---:|
| Embedding (76 → 8) | (B, 80, 8) | 608 |
| LSTM 1 (8 → 256) | (B, 80, 256) | 272,384 |
| Dropout | | 0 |
| LSTM 2 (256 → 256), last hidden state | (B, 256) | 526,336 |
| Dropout | | 0 |
| Linear (256 → 76) | (B, 76) logits | 19,532 |
| **Total** | | **818,860** |

Activations: sigmoid/tanh inside the LSTM gates, softmax inside the cross-entropy loss.

**Why an LSTM?** Text is sequential and the LSTM keeps a memory (hidden state) of the previous
characters. Alternatives: a simple RNN (vanishing gradient over 80 steps), a GRU (similar, but the
LSTM is the benchmark reference), a Transformer (too heavy for a 4 GB GPU) and a 1D CNN (limited
context). There are no pre-trained models for this character vocabulary, so we train from scratch
(Xavier / orthogonal initialisation, forget-gate bias = 1).

## Training (`src/train.py`)

- Loss: cross-entropy. Optimizer: AdamW. Scheduler: cosine annealing (lr → lr/100).
- Regularisation: dropout, weight decay, gradient clipping (norm 5), early stopping (patience 3 on
  validation macro-F1).
- The best epoch (highest validation macro-F1) is saved to `checkpoints/model_best.pt`.
- Logs: loss, accuracy, macro-F1, AUC-ROC and learning rate per epoch in TensorBoard (`runs/`);
  learning curves in `reports/figures/`.

A full epoch on the 3 M training rows takes ~13 min on our GPU, so to fit in ~3.5 hours we train on
a fixed random subsample (seed 42): 300k rows for the search and 1.5M rows for the final models.

**Hyperparameter search** (4 epochs, 300k train rows, 100k val rows) — `reports/hparam_search.csv`:

| lr | dropout | weight decay | val macro-F1 | val accuracy |
|---:|---:|---:|---:|---:|
| **3e-3** | **0.1** | **1e-4** | **0.189** | **0.520** |
| 3e-3 | 0.3 | 1e-2 | 0.178 | 0.505 |
| 1e-3 | 0.3 | 1e-2 | 0.141 | 0.459 |
| 1e-3 | 0.1 | 1e-4 | 0.138 | 0.450 |
| 3e-4 | 0.1 | 1e-4 | 0.083 | 0.374 |
| 3e-4 | 0.3 | 1e-2 | 0.079 | 0.367 |

**Final hyperparameters:** lr 3e-3, dropout 0.1, weight decay 1e-4, batch size 512, 10 epochs,
1.5M train rows, seed 42.

**Class imbalance.** We trained two variants with these hyperparameters: (a) without class weights
and (b) with class weights `1/sqrt(frequency)` computed on train only. We keep the one with the best
validation macro-F1:

| Variant | Best epoch | Val macro-F1 | Val accuracy |
|---|---:|---:|---:|
| (a) no class weights | 9 | 0.2606 | 0.5879 |
| (b) class weights (**selected**) | 9 | 0.2617 | 0.5495 |

## Results on test (`src/evaluate.py`)

Workshop 1 baselines were evaluated on a 10k-row sample of the test split, so all models are
compared on the same 10k rows (`reports/results_comparison.csv`):

| Model | Accuracy | F1 macro | F1 weighted | AUC-ROC |
|---|---:|---:|---:|---:|
| Majority class (space) | 0.183 | 0.005 | 0.057 | 0.500 |
| Logistic regression (W1) | 0.369 | 0.108 | 0.324 | 0.871 |
| Naive Bayes (W1) | 0.334 | 0.107 | 0.304 | 0.849 |
| Random forest (W1) | 0.308 | 0.140 | 0.350 | 0.820 |
| LSTM without class weights | 0.590 | 0.305 | 0.575 | 0.965 |
| **LSTM with class weights** | 0.557 | 0.301 | 0.571 | 0.962 |

On the full test split (644k rows) the selected model gets accuracy 0.553, macro-F1 0.290 and
AUC-ROC 0.954. The LSTM is much better than the baselines
because it uses all 80 characters of context, while the baselines only used the last 5.

## How to run

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# Workshop 1 data must exist first (Workshop_1: python -m src.dataset --prepare)

python -m src.model        # architecture + forward-pass check
python -m src.train --search --epochs 4 --max-train-samples 300000 --max-val-samples 100000
python -m src.train --run-name lstm_noweights --epochs 10 --max-train-samples 1500000
python -m src.train --run-name lstm_weighted  --epochs 10 --max-train-samples 1500000 --class-weights
python -m src.evaluate     # test metrics, comparison table, confusion matrix
python -m src.test_model   # loads model_best.pt and predicts the next character
tensorboard --logdir runs
```

The results notebook is `notebooks/workshop2_results.ipynb` (it only loads saved results).

## Hardware and time

NVIDIA GTX 1650 (4 GB), 7.4 GB RAM, Windows 11, torch 2.14.1 (CUDA 12.6).
Search: 36 min. Each final variant: ~84 min. Evaluation: ~3 min.

## Notes about Workshop 1

- `Workshop_1/src/baselines.py` crashes on Windows with the default matplotlib backend; we ran it
  with `MPLBACKEND=Agg` and copied its test metrics to `reports/baselines_workshop1.csv`.
- `splits.json` and `vocab.json` were checked (no speaker in more than one split) and were not
  changed.

## Structure

```
src/          model.py, train.py, evaluate.py, test_model.py
notebooks/    workshop2_results.ipynb
checkpoints/  model_best.pt (+ lstm_noweights.pt, lstm_weighted.pt used in the comparison table)
runs/         TensorBoard logs of every run (search + final variants)
reports/      hparam_search.csv, results_comparison.csv, baselines_workshop1.csv, figures/
```
