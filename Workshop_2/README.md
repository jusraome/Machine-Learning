# Workshop 2 — Character Transformer

This workshop trains a character-level Transformer to predict the next character
from an 80-character context in the Shakespeare dataset prepared in Workshop 1.

## Structure

```text
Workshop_2/
├── data/processed/       # Tokenized train, validation, and test splits (.npz)
├── src/                  # Tokenizer, data preparation, model, and scripts
├── checkpoints/          # Best model checkpoint
├── results/              # Final evaluation metrics
└── runs/                 # Training logs and plots
```

## Run from the repository root

```bash
python -m Workshop_2.src.data_preparation
python -m Workshop_2.src.train
python -m Workshop_2.src.evaluate
python -m Workshop_2.src.test_model
```

`evaluate` uses the held-out test split. Run it only when the model configuration
is final. `test_model` loads the checkpoint and interactively predicts one next
character from a supplied prefix.

The model and data scripts require PyTorch. NumPy and scikit-learn are also used;
the repository's root `requirements.txt` includes the latter dependencies.
