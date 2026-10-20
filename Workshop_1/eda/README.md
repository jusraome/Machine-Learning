# Workshop 1 — Shakespeare Next-Character Prediction

Downloading, exploring and preprocessing the `flwrlabs/shakespeare` dataset, with a split that
avoids data leakage.

**Course:** Machine Learning — Universidad Distrital
**Status:** Steps 1–3 done (download, EDA, cleaning + split). Models are out of scope; see
[Out of scope / Next steps](#out-of-scope--next-steps).

---

## 1. The dataset and the task

[`flwrlabs/shakespeare`](https://huggingface.co/datasets/flwrlabs/shakespeare) is the Shakespeare
part of the **LEAF** benchmark, built from *The Complete Works of William Shakespeare*
(Project Gutenberg).

| | |
|---|---|
| Rows | 4,226,158 |
| Splits provided | one (`train`) |
| Columns | `character_id` (text), `x` (80 characters), `y` (1 character) |
| Target classes | 79 |
| Distinct `character_id` | 1,129 (across 36 plays) |
| Size | 561 MiB in memory, 49 MiB as Parquet |
| Pinned revision | `2cfe3a5ba0d9b34634ade086803a479b2c2b8e11` |
| License | BSD-2-Clause |

**Task.** Given 80 characters of dialogue `x`, predict the next character `y`. This is
**multiclass classification** with 79 classes.

**The key fact about this data.** The rows are a window that slides **one character at a time**
over each character's lines. Checked on the whole dataset: all **4,225,029** consecutive row pairs
that belong to the same `character_id` satisfy `x[i+1] == x[i][1:] + y[i]` — 100.000000%, with no
exceptions. So two neighbouring rows share **79 of their 80 characters (98.75%)**. This is why the
split has to be done carefully (§6.2).

### Citation

```bibtex
@article{caldas2018leaf,
  title   = {LEAF: A Benchmark for Federated Settings},
  author  = {Caldas, Sebastian and Duddu, Sai Meher Karthik and Wu, Peter and Li, Tian
             and Kone{\v{c}}n{\'y}, Jakub and McMahan, H. Brendan and Smith, Virginia
             and Talwalkar, Ameet},
  journal = {arXiv preprint arXiv:1812.01097},
  year    = {2018}
}
```

Source text: *The Complete Works of William Shakespeare*, Project Gutenberg (public domain).
Dataset packaging: BSD-2-Clause, © the `flwrlabs` maintainers.

---

## 2. Setting up the environment

**Requires Python 3.12 or newer** (developed and tested on 3.13.2). You also need about
**700 MB of free disk space** for the dataset.

Create the environment:

```
python -m venv .venv
```

Activate it:

| System | Command |
|---|---|
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (cmd) | `.venv\Scripts\activate.bat` |
| Linux / macOS | `source .venv/bin/activate` |

Then install the dependencies:

```
pip install -r requirements.txt
```

All versions in `requirements.txt` are pinned.

> If PowerShell refuses to run the activation script ("running scripts is disabled"), you can skip
> activation and call the interpreter directly instead: use `.venv\Scripts\python.exe` in place of
> `python` in every command below.

> **In VS Code**, if the editor reports the packages as missing, it is using a different
> interpreter. Fix it once with `Ctrl+Shift+P` → *Python: Select Interpreter* → pick the one
> inside `.venv`. For `notebooks/eda.ipynb` the kernel is chosen separately, with *Select Kernel*
> at the top right.

---

## 3. Running everything

Run these from the project folder, with the environment activated. **The order matters** — each
step needs the previous one, and running one too early prints a message telling you which command
to run first.

```
# Step 1 - download the data                       (~1 min, downloads ~500 MB)
python -m src.download

# Step 2 - print a summary of the data             (~20 s)
python dataset_info.py

# Step 3 - validate, clean and split               (~2 min)
python -m src.dataset --prepare

# Step 4 - run the EDA notebook from start to end  (~3 min)
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/eda.ipynb
```

Steps 1-3 write into `data/`, which is not in the repository, so a fresh clone has to run them
before the notebook will work. You can also just open `notebooks/eda.ipynb` and read it: it is
saved with all of its outputs and figures already in place.

Extra options: `python dataset_info.py --top 20 --all-classes` shows longer tables;
`python -m src.download --force` downloads again; `python -m src.dataset --prepare
--keep-duplicates` skips the deduplication rule.

### How the download works

`src/download.py` uses `load_dataset()` from the `datasets` library, with the **revision pinned to
a specific commit** so a later change on the Hub cannot silently change our data. It then:

1. converts the dataset to a pandas DataFrame and saves it to `data/raw/shakespeare.parquet`;
2. computes a **SHA256 checksum** and writes it to `data/raw/CHECKSUMS.txt`. The dataset provider
   does not publish checksums, so this one is generated locally — it lets us detect a corrupted or
   accidentally modified copy. Verify later with `sha256sum -c CHECKSUMS.txt`;
3. writes `data/raw/dataset_meta.json` with the revision, license, row count, column types and
   file size.

---

## 4. Project structure

```
.
├── README.md
├── requirements.txt              # pinned versions
├── .gitignore
├── dataset_info.py               # prints a summary of the data
├── src/
│   ├── __init__.py
│   ├── config.py                 # SEED, folder paths, split settings
│   ├── download.py               # step 1: download + checksums
│   └── dataset.py                # steps 2-3: load / validate / clean / split / vocabulary
├── notebooks/
│   └── eda.ipynb                 # the EDA, 8 sections, each ending with findings
├── data/
│   ├── raw/                      # not in git
│   │   ├── shakespeare.parquet
│   │   ├── CHECKSUMS.txt
│   │   └── dataset_meta.json
│   ├── processed/                # not in git
│   │   ├── train.parquet
│   │   ├── val.parquet
│   │   ├── test.parquet
│   │   └── prepare_report.json   # rebuilt by --prepare
│   ├── splits.json               # in git: which characters go in each split
│   └── vocab.json                # in git: the character vocabulary
└── reports/
    └── figures/                  # figures from the EDA (PNG)
```

`data/raw/` and `data/processed/` are not tracked in git (large and easy to rebuild), but
`data/splits.json` and `data/vocab.json` **are**, because they are what makes the split
reproducible without sharing the data itself.

---

## 5. EDA summary

Full analysis with figures: [`notebooks/eda.ipynb`](notebooks/eda.ipynb). Every number covers all
4,226,158 rows — **nothing in the notebook is sampled**.

| Finding | Number |
|---|---|
| Overlap between consecutive rows | 79/80 = **98.75%**, on 100.000000% of same-character pairs |
| Most frequent class (space) | **18.74%** of rows — the accuracy to beat |
| Imbalance ratio | **791,902:1** (space vs. `0` / `9`) |
| Entropy H(y) | **4.5002 bits** vs. 6.3038 if equal → about 22.6 effective classes |
| Concentration | **113 of 1,129 characters (10%) hold half the rows** |
| Rows per character | 3 → 66,903 (median 1,163) |
| What `x` contains | 72.43% lowercase, 18.73% space, 4.64% punctuation, 4.19% uppercase, 0.0011% digits |
| Non-ASCII / control characters | **0** out of 338,092,640 characters |
| Missing values, empty strings, wrong lengths | **0** |
| Duplicate rows | 1,445 (**0.0342%**) |
| Windows with more than one target | 12 windows / 29 rows (**0.000686%**) |
| Usable rows | **99.9658%** |
| Information from context | 1 character → 0.97 bits (21.6%); 5 characters → 2.81 bits (62.4%) |

**Why accuracy alone is not enough.** A model that always predicts a space scores 18.74% without
learning anything, and it can *raise* its accuracy by ignoring rare classes completely. The
modelling stage should also report **macro F1**, **top-k accuracy**, **per-class recall** for rare
classes, and **cross-entropy**, which compares directly with the 4.5002 bits measured above.

---

## 6. Preprocessing decisions

### 6.1 Cleaning rules

Each rule reports how many rows it removed, so a rule that removes nothing is still visible.
Numbers come from `src.dataset.validate` over all 4,226,158 rows.

| Rule | Applied | Rows removed | Why |
|---|---|---:|---|
| Drop malformed rows (`len(x) != 80` or `len(y) != 1`) | yes | **0** | Nothing breaks the rule, but the check stays in case the data changes. |
| Drop duplicate `(character_id, x, y)` rows | **yes** | **1,445** | They add no information. All copies share one `character_id`, so the grouped split already keeps them inside one split — removing them cannot leak anything. |
| Unicode NFC normalisation | **no** | 0 | Measured: 0 non-ASCII characters, so it would change nothing. |
| Collapse repeated spaces | **no** | 0 | Measured: 0 rows with repeated spaces. LEAF already did this. |
| Lowercase / remove punctuation | **no** | — | Upper case, lower case, punctuation and the space are **target classes**. This would destroy the labels. |
| Drop contradictory rows | **no** | 29 | Real ambiguity in Shakespeare's text, not an error. Removing it would make the evaluation look better than it really is. |

Result: **4,226,158 → 4,224,713 rows (−1,445, −0.0342%)**.

### 6.2 The split: grouped by `character_id`

`GroupShuffleSplit` applied twice (`SEED = 42`) over the 1,129 characters, 70/15/15 **of
characters**.

| Split | Characters | Rows | Share of rows |
|---|---:|---:|---:|
| train | 790 | 2,997,217 | 70.94% |
| val | 169 | 583,098 | 13.80% |
| test | 170 | 644,398 | 15.25% |

Checked and reported by the pipeline:

- **No character is in two splits.** All three pairwise overlaps are **0**. Since every
  overlapping window belongs to one character, windows cannot cross between splits — the 98.75%
  overlap cannot leak.
- **The row shares are not exactly 70/15/15** (70.94 / 13.80 / 15.25) because characters range
  from 3 to 66,903 rows. `GroupShuffleSplit` picks whole groups by group count, so exact row
  shares are impossible with any group-based method. The real shares are reported, not forced.
- **The class distribution survives.** The biggest difference in class frequency between any two
  splits is **0.1643 percentage points** (class `t`).

**Why not a temporal split inside each character?** The alternative is to cut each character's
lines into blocks in order (first 70% → train, next 15% → val, last 15% → test) with a **gap of at
least 79 rows** between blocks, which is exactly enough so no window crosses a boundary. It was
**not** chosen because:

- It answers an easier question. It only tests on characters the model already trained on, so a
  model could rely on a character's particular style and still score well. Grouping by character
  tests whether the model works on **characters it has never seen**, which is a stronger claim.
- Its correctness depends on getting the gap arithmetic exactly right. The grouped split is safe
  by construction, and the check (no shared characters) is trivial to verify.
- It matches how the dataset was built: in LEAF each `character_id` is a separate client, and
  holding out whole clients is the benchmark's own way of evaluating.

Its one real advantage is listed honestly in §6.3, because the grouped split pays for it.

### 6.3 Vocabulary and unknown characters

The character vocabulary is built from the **training split only** (`data/vocab.json`). Building
it on the whole dataset would mean the model's input encoding already knows about val and test.
Index 0 is reserved for `<UNK>`.

**The cost of grouping, measured.** The train vocabulary has **75 characters + `<UNK>` = 76
tokens**, but the full dataset has **79**. The characters **`0`, `8`, `9` and `>`** appear only in
val/test: each occurs in a handful of rows that happen to belong to a few characters, and digits
are only 0.0011% of all context characters.

| Split | Unknown characters found in `x` | Rows with an unknown target |
|---|---|---:|
| train | none | 0 |
| val | `8`, `9` (160 occurrences) | **2** |
| test | `0`, `8`, `>` (591 occurrences) | **4** |

All unknown characters become `<UNK>`. Six rows in total have a target the model could never have
learned — about 1 per million. That will not move any overall score, but it **must not be hidden**
when reporting per-class results. A temporal split would have kept all 79 classes in training;
this is the trade-off accepted in §6.2.

There are no other normalisation parameters in this pipeline, so nothing else needs to be fitted
on train only.

### 6.4 Target encoding

`data/processed/*.parquet` contains `character_id`, `x`, `y` and a new **`y_id`** column: the
target's index in the vocabulary, or 0 (`<UNK>`) if it never appeared in train. `x` is kept as
plain text on purpose — turning it into model input is feature extraction, which is out of scope
here.

---

## 7. Reproducibility

- `SEED = 42` is defined once in `src/config.py`. Every script imports it and calls `set_seeds()`
  before anything random. scikit-learn has no global seed, so its splitters are created with
  `random_state=SEED` directly.
- The dataset revision is **pinned to a commit**, and a checksum is written on every run.
- `requirements.txt` pins every version.
- **`data/splits.json` is the source of truth.** It stores the character ids for each split (1,129
  strings), the seed, the date and the resulting counts — not the 4.2 M row numbers, which would
  make a ~40 MB file for no benefit. `src.dataset.load_split(name)` rebuilds any split exactly,
  either from the saved Parquet file (`source="processed"`) or from the raw data
  (`source="raw"`); both return the same rows.
- **Never split the data again once `splits.json` exists.** All later code must use
  `load_split()`. Re-running `--prepare` with the same seed gives the same split, but any new code
  that re-randomises would quietly invalidate everything done before it.

---

## 8. Proposed data augmentation (described, not implemented)

Augmentation is not part of this stage. It is written down here so the modelling stage can decide
about it on purpose. It would apply to **training data only** — never to val/test, and never to
`y`.

**Main idea: random character substitution.** With a small probability (about 1–2% per position),
replace a character in `x` with another one drawn from the *training* character distribution,
leaving `y` unchanged.

- *Why.* The EDA found 4,224,322 different windows out of 4,226,158 rows, so the model sees each
  context essentially once and is free to memorise it instead of learning. Adding noise makes
  memorising exact windows useless, while keeping the local letter patterns that actually carry
  the signal (Section 6 of the notebook: five characters of context remove 62.4% of the
  uncertainty). It also imitates a realistic kind of error for text that was digitised.

**Secondary idea: shorter context.** Replace a random number of characters at the start of `x`
with a padding token, so the model has to predict from less context.

- *Why.* Most of the useful signal is in the last few characters, and the other 75 positions carry
  longer-range structure. Training with different amounts of context stops the model from
  depending on always having exactly 80 characters, and makes it usable at the beginning of a
  passage.

**An important warning if this is used.** Because consecutive rows already share 98.75% of their
characters, changing one window produces something very close to a changed version of its
neighbours. The **real** increase in variety is much smaller than the 4.2 M row count suggests, so
the amount of augmentation should be tuned against the held-out characters (§6.2) rather than
assumed to help just because the dataset is large.

---

## Out of scope / Next steps

Not done at this stage, on purpose:

- [ ] **`src/features.py` — turning `x` into model input.** One-hot or integer encoding, n-gram
      counts, embedding-ready arrays. `data/vocab.json` and the `y_id` column are the starting
      point.
- [ ] **`src/baselines.py` — baseline models.** At least: always predict the most frequent class
      (known score: **18.74%**), a model that looks up the previous character (the k=1 entropy of
      3.5288 bits limits how well it can do), and logistic regression on the last few characters.
- [ ] **Training and evaluation.** Report macro F1, top-k accuracy, per-class recall and
      cross-entropy as well as accuracy, for the reasons in §5.
- [ ] **Implementing the augmentation** described in §8.
- [ ] **The PDF report.** Sections 5 and 6 of this README, plus the figures in `reports/figures/`,
      are the material to write it from.
