# scDRTrans: Noise-Aware Knowledge Transfer from Single-Cell Knowledge Bases for Drug Response Prediction

## Introduction

**scDRTrans** is a semi-supervised domain adaptation framework that transfers drug-response knowledge from weakly labelled single-cell datasets to sparsely annotated target datasets for single-cell drug response prediction.

The **source domain** comprises scRNA-seq datasets from [scDrugAct](https://github.com/hliulab/scDrugAct), with drug sensitivity and resistance computationally inferred using a Connectivity Map (CMap) strategy based on gene set enrichment analysis (GSEA). The **target domain** contains limited experimentally annotated drug-response labels. To address noisy source annotations and cross-domain distribution shifts, scDRTrans integrates source-label adaptation with minimax entropy-based domain adaptation under limited target-domain supervision.

scDRTrans comprises three main components:

1. **Denoising Autoencoder with Self-Attention** — A shared encoder learns low-dimensional representations of scRNA-seq profiles from both domains while preserving transcriptomic information through reconstruction.
2. **Minimax Entropy Domain Adaptation** — A fully connected classifier coupled with adversarial optimization aligns target representations with the source-derived class structure, promoting compact and discriminative feature distributions across unlabelled target cells.
3. **Prototype-Based Label Adaptation** — After an initial warm-up stage, a prototype module refines noisy source labels via adaptive instance-level feature recalibration, incorporating a label-adaptation loss to down-weight unreliable source annotations.

## Model Architecture

**Key components:**


| Module                                   | Role                                                                              |
| ---------------------------------------- | --------------------------------------------------------------------------------- |
| `DAE` (Denoising AE + 4× Self-Attention) | Shared encoder: noise-robust cross-domain representation learning                 |
| `Predictor`                              | Maps latent embedding → 32-dim feature space                                      |
| `Predictor_adentropy`                    | Temperature-scaled classifier with gradient reversal for MME adversarial training |
| `ProtoClassifier (PPC)`                  | Prototype-based source label adaptation via weighted instance recalibration       |
| `FocalLoss`                              | Handles class imbalance in computationally inferred source labels                 |


**Training objective:**

The total loss function combines classification, source label adaptation, minimax entropy domain adaptation, and reconstruction:

```
L_total = L_cls + L_sla + λ × L_mme + η × L_recon
```

where λ and η are hyperparameters balancing domain adaptation and feature preservation. In the default configuration: λ = 0.02, η = 0.55, and the source label adaptation weight is 0.40 (activated after a 100-epoch warm-up).


| Loss term | Description                                                    |
| --------- | -------------------------------------------------------------- |
| L_cls     | Focal Loss on source + cross-entropy on labeled target cells   |
| L_sla     | Prototype-based soft label adaptation on source (post warm-up) |
| L_mme     | Minimax entropy on unlabeled target cells (λ = 0.02)           |
| L_recon   | MSE reconstruction loss on both domains (η = 0.55)             |


## Requirements

Models were trained using NVIDIA A100 GPU on a Linux system.

- Python 3.9
- PyTorch 2.1.0
- Scanpy 1.10.3
- Pandas 2.3.2
- NumPy 1.26.4
- Scikit-learn 1.6.1
- Shap 0.45.1

Full dependency list: see `requirements.txt`.

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/rabbit-qin/scDRTrans.git

# 2. Enter the project directory
cd scDRTrans/

# 3. Create and activate the environment
conda create --name scDRTrans python=3.9
conda activate scDRTrans
pip install -r requirements.txt
```

## Data Preparation

### Source domain (weakly labelled single-cell)

**scDrugAct** source domain: computationally inferred drug sensitivity labels using CMap + GSEA strategy.

Use `otherCodes/pre_processing_singlecell.py` to:

1. Load raw single-cell count matrices per drug from scDrugAct multi-dataset reference.
2. QC filtering: min_genes=200, min_cells=3, MT% < 20%.
3. Normalize (total count 1e4) → log1p → select HVGs.
4. Intersect with a curated drug-sensitivity gene list (`senstive_gene_list.txt`) and target HVGs.
5. Output aligned expression matrix and weakly labelled metadata.

For the **unknown-drug (cross-drug) setting**, use `otherCodes/pre_processing_singlecell_unknownDrug.py` (source drug differs from target drug).

### Target domain (experimentally annotated single-cell)

**GEO datasets** with experimentally validated drug-response labels (sensitive/resistant).

Use `otherCodes/pre_processing.py` or `otherCodes/pre_processing_DRMref.py` to align raw GEO scRNA-seq count data to the source gene space:

1. Parse GEO count matrix and metadata.
2. Gene name standardization and deduplication.
3. QC filtering and normalization (same pipeline as source).
4. Gene space alignment with source domain via HVG intersection.

## Usage

### Training (semi-supervised, n-shot)

```bash
python experiment_shot.py \
  --drug Erlotinib_GSE149214 \
  --n 15 \
  --gene HVGs \
  --encoder_h_dims "512,256" \
  --bottleneck 256 \
  --predictor_h_dims "64,32" \
  --epochs 300 \
  --lr 0.001 \
  --batch_size 200 \
  --dropout 0.3 \
  --device gpu
```

**Key arguments:**


| Argument           | Default               | Description                                                                                                                                                                                                         |
| ------------------ | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--drug`           | `Erlotinib_GSE149214` | Drug/dataset identifier. Options include `Erlotinib_GSE149214`, `Erlotinib_GSE134839`, `Palbociclib`, `Etoposide`, `Ibrutinib_GSE152469`, `Cisplatin_Cisp568`, `Paclitaxel_GSE163836`, `Crizotinib_ts485to_c`, etc. |
| `--n`              | `15`                  | Number of labeled target cells per class (n-shot).                                                                                                                                                                  |
| `--gene`           | `HVGs`                | Gene feature type: `HVGs` or `all`.                                                                                                                                                                                 |
| `--encoder_h_dims` | `"512,256"`           | Hidden layer sizes of the encoder, comma-separated.                                                                                                                                                                 |
| `--epochs`         | `300`                 | Training epochs.                                                                                                                                                                                                    |
| `--lr`             | `0.001`               | Learning rate (Adagrad optimizer).                                                                                                                                                                                  |
| `--batch_size`     | `200`                 | Mini-batch size.                                                                                                                                                                                                    |
| `--dropout`        | `0.3`                 | Dropout rate.                                                                                                                                                                                                       |
| `--method`         | `adv`                 | Training mode (`adv` = adversarial MME).                                                                                                                                                                            |
| `--device`         | `gpu`                 | `gpu` or `cpu`.                                                                                                                                                                                                     |


The script runs **50 independent trials** with different random seeds and reports mean ± std across trials for AUC, AUPR, F1, ACC, Recall, and Precision.

### Baseline (source-only, no target adaptation)

```bash
python experiment_shot_baseline.py \
  --drug Palbociclib \
  --n 15 \
  --encoder_h_dims "512,256" \
  --predictor_h_dims "64,32" \
  --epochs 300 \
  --lr 0.001 \
  --batch_size 200
```

Results are saved to `save/results/sc/` and model checkpoints to `save/sc/all_path/`.

## Directory Structure

```
scDRTrans/
├── model.py
├── trainer.py
├── experiment_shot.py
├── experiment_shot_baseline.py
├── utils.py
├── framework.jpg
├── requirements.txt
├── Datasets/
│   ├── rawData/
│   ├── processedData/
│   ├── benchmark.py
│   └── benchmark_unknownDrug.py
└── otherCodes/
    ├── pre_processing_singlecell.py
    ├── pre_processing_singlecell_unknownDrug.py
    ├── pre_processing.py
    ├── pre_processing_unknownDrug.py
    ├── pre_processing_DRMref.py
    ├── sourceDataFilter.py
    └── expressionDataFilter.py
```

## Supported Drugs and Datasets


| Drug        | Source Domain             | Target Domain (GEO)  |
| ----------- | ------------------------- | -------------------- |
| Erlotinib   | scDrugAct (multi-dataset) | GSE149214, GSE134839 |
| Palbociclib | scDrugAct (multi-dataset) | GSE131984            |
| Etoposide   | scDrugAct (multi-dataset) | GSE149215            |
| Ibrutinib   | scDrugAct (multi-dataset) | GSE152469            |
| Cisplatin   | scDrugAct (multi-dataset) | GSE117872 (Cisp568)  |
| Paclitaxel  | scDrugAct (multi-dataset) | GSE163836, GSE131984 |
| Crizotinib  | scDrugAct (multi-dataset) | Internal sc datasets |


## Evaluation Metrics

All results are reported as **mean ± std over 50 independent runs**:

- AUC (Area Under ROC Curve)
- AUPR (Area Under Precision-Recall Curve)
- F1 Score
- Accuracy (ACC)
- Recall
- Precision

