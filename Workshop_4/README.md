Workshop 4 — Experimental GRU Autoencoders and Variational Autoencoders
Status: exploratory research branch — not the selected Workshop 4 submission.

This directory documents an alternative implementation developed during Workshop 4. The team selected the implementation on the main branch for its final submission. This experimental GRU-based approach was preserved for reproducibility, diagnostics, and potential future comparison; it should not be interpreted as the team's final reported solution.

Purpose
The experiment investigates whether recurrent autoencoders can reconstruct 80-character Shakespeare text windows and identify unusual windows through reconstruction error. It uses the data preparation, 76-token vocabulary, and speaker-disjoint train/validation/test splits from Workshop 1.

Unlike the team's main-branch implementation, which uses fully connected layers, this branch explores an embedding → GRU encoder → latent bottleneck → GRU decoder → character logits architecture, along with a variational extension producing mu and logvar. The reconstruction objective is categorical cross-entropy, summed across the 80-character window; the VAE objective additionally includes beta × KL.
Experimental status and findings
The following results were obtained from the normal-only anomaly detection workflow:
Configuration	Status	Development ROC-AUC (best scoring method)
GRU AE, latent dimension 32	Trained and evaluated	0.5356 (reconstruction NLL)
GRU AE, latent dimension 64	Trained and evaluated	0.5638 (categorical Brier MSE)
GRU VAE, latent dimension 32, beta=0.5	Notebook prepared; not trained	—
GRU VAE, latent dimension 32, beta=1.0	Notebook prepared; not trained	—
GRU VAE, latent dimension 32, beta=4.0	Notebook prepared; not trained	—


The two GRU AEs were trained on 562,253 normal training windows, with early stopping, numerical safeguards, and checkpoint preservation. Their anomaly scores were compared on the same 7,640 development/validation examples, using a frozen evaluation protocol. The final test split was deliberately kept closed in this workflow.

For context, the team’s main-branch, fully connected normal-only AE reported 0.712 ROC-AUC on a 20,000-window test sample, after training on 37,762 normal windows sampled from its training subset. Although this experiment used more normal training windows, its observed anomaly discrimination on development data was weaker. These figures are not a head-to-head benchmark: the architectures, training samples, and evaluation partitions differ. In particular, no claim is made here about this branch's final test performance.

The main-branch implementation was therefore preferred for the submitted Workshop 4 report because it provided stronger reported results and a more complete set of AE/VAE experiments, reconstructions, and latent-space analyses. More training data alone did not guarantee better anomaly detection in this investigation.
Engineering observations
- Recurrent training required close attention to optimization stability. Earlier runs experienced large gradients and deterioration after initially useful reconstruction performance.
- The experimental training code adds FP64 global gradient-norm calculations, gradient clipping, checkpoint integrity checks, early stopping, and diagnostic logging.
- Increasing the AE latent dimension from 32 to 64 improved development ROC-AUC, but both scores remained relatively close to random ranking (0.5).
- The VAE notebooks were prepared, not completed; no VAE outcome should be inferred from this branch.
- GPU checkpoints and evaluation outputs produced in Google Colab were stored separately in Google Drive and are not necessarily included in this repository.
Key files
Location	Purpose
src/autoencoder.py	GRU AE/VAE architectures and objectives
src/train_ae.py	General training and recovery workflow
src/train_normal_ae.py	Normal-only AE training
src/train_remaining_normal.py	Configuration and training support for remaining models
src/evaluate_normal_models.py	Paired development evaluation using the frozen protocol
notebooks/Workshop_4_NormalOnly_AE_Colab.ipynb	AE z32 normal-only Colab workflow
notebooks/Workshop_4_AE_z64_NormalOnly_Colab.ipynb	AE z64 normal-only Colab workflow
notebooks/Workshop_4_VAE_z32_b0p5_NormalOnly_Colab.ipynb	Prepared VAE beta=0.5 workflow
notebooks/Workshop_4_VAE_z32_b1_NormalOnly_Colab.ipynb	Prepared VAE beta=1.0 workflow
notebooks/Workshop_4_VAE_z32_b4_NormalOnly_Colab.ipynb	Prepared VAE beta=4.0 workflow
tests/	Automated checks for data contracts, models, stability, and evaluation


Detailed experiment logs, caveats, and implementation decisions are documented in [NORMAL_ONLY_ANOMALIES_PLAN.md](NORMAL_ONLY_ANOMALIES_PLAN.md), [REMAINING_NORMAL_MODELS.md](REMAINING_NORMAL_MODELS.md), [AUDIT_REPORT.md](AUDIT_REPORT.md), and [AUDIT_REPORT_PHASE4.md](AUDIT_REPORT_PHASE4.md).
Reproducibility and scope
The experimental notebooks default to disabled training/evaluation switches and require the Workshop 1 processed data, matching vocabulary/split metadata, and the appropriate existing checkpoints. The frozen development protocol must be reused when comparing the completed GRU AEs. Do not train on or tune against the test split.

For the final Workshop 4 deliverable, refer to the team's main branch. This branch is retained as an independent research artifact, not as a replacement for the final implementation or its published metrics.
