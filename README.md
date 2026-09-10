# BTP_1_project_ML_in_Bio_Data

Machine Learning on Biological Data — BTP project by Animesh Kumar.

Predicting **siRNA silencing efficacy** from sequence data using a heterogeneous graph neural network (HinSAGE, via StellarGraph) and gradient boosting (XGBoost).

## Project Overview

Small interfering RNAs (siRNAs) silence genes through RNA interference, but their knockdown efficacy varies widely across sequences. This project models siRNA–mRNA interactions as a **heterogeneous graph** with three node types — `siRNA`, `mRNA`, and `interaction` — and learns efficacy with **HinSAGE** (Heterogeneous GraphSAGE), evaluated over 10 train/dev/test splits.

## Repository Structure

```
├── siRNADiscovery.py / .ipynb          # Main HinSAGE pipeline (10-fold CV)
├── siRNA_Discovery_S.py / .ipynb       # Variant of the discovery pipeline
├── siRNADiscovery _XGBoost.ipynb       # XGBoost baseline
├── siRNADiscovery copy.ipynb           # Working copy of the main notebook
├── utils.py                            # Feature calculation helpers
├── siRNA_param.json                    # Model / training hyperparameters
├── RNA_AGO2/                           # AGO2-binding scores (siRNA & mRNA)
│   ├── siRNA_AGO2.csv
│   └── mRNA_AGO2.csv
├── siRNA_split_datasets/               # 10 splits (split0–split9)
│   └── splitN/{train,dev,test}.csv     # Columns: siRNA, mRNA, siRNA_seq, mRNA_seq, efficacy
└── siRNA_split_preprocess/             # Precomputed feature matrices
    ├── con_matrix.txt                  # Co-fold (siRNA–mRNA) thermodynamic features
    ├── self_siRNA_matrix.txt           # siRNA self-fold features
    └── self_mRNA_matrix.txt            # mRNA self-fold features
```

## Features Used

- **One-hot encoding** of siRNA (21 nt) and mRNA sequences
- **Positional encoding** of the siRNA–mRNA binding position
- **Thermodynamic features** (co-fold and self-fold energies)
- **AGO2-binding scores** for siRNA and mRNA
- **GC content**
- **k-mer frequencies** (k = 1…5)
- **Established siRNA design rules** (positional scores)

## Model

A 2-layer HinSAGE network (`hinsage_layer_sizes: [64, 32]`) over the siRNA–mRNA–interaction graph, with a dense regression head predicting efficacy. Hyperparameters are in `siRNA_param.json` (batch size 64, 26 epochs, Adam lr 0.001, MSE loss).

## Evaluation

For each of the 10 splits the model reports:

- **PCC** — Pearson correlation coefficient
- **SPCC** — Spearman correlation coefficient
- **MSE** — mean squared error
- **AUC** — ROC-AUC with efficacy > 0.7 as the positive class

## Getting Started

1. Set up a Python 3.10 environment (this project used a venv named `sirna_env_3.10`).
2. Install dependencies: `stellargraph`, `tensorflow`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `xgboost`.
3. Run the pipeline:

   ```bash
   python siRNADiscovery.py
   ```

   Results for all 10 splits are printed along with the average of each metric.

## Report & Presentation

- `BTP_Report_Animesh(22CS30009).pdf` — full project report
- `btp _ppt_(22CS30009).pdf` / `.pptx` — presentation slides
