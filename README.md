## TransHLA2.0 — Tutorial and README

High-throughput screening on HLA class I epitopes underpins neoantigen vaccines development, T cell therapies, and frontline responses to emerging pathogens. However, practical pipelines still struggle to prioritize true ligands and immunogenic targets at proteome scale. 

We introduce **TransHLA2.0**, a compact three-stage framework that addresses these challenges:

- **TransHLA2.0-PRE**: Enriches epitope-like peptides in peptide-only setting with standardization and preprocessing utilities
- **TransHLA2.0-BIND**: Resolves allele-specific binding/presentation with quantitative supervision integrating eluted ligands and IC50-annotated pairs. A minimal Hugging Face-compatible binding classifier achieving **AUROC of 96.2%** and **AUPRC of 95.2%** on combined BA/EL evaluation
- **TransHLA2.0-IM**: Identifies immunogenic ligands from rigorously curated human T cell assays using cross-attention/CNN branches

### Key Features

- Trained on diverse IEDB ligands and IC50-annotated peptide–HLA pairs, capturing allele-discordant cases that sharpen specificity while maintaining well-calibrated operating points
- Achieves competitive or superior discrimination with markedly fewer trainable parameters through **Low-Rank Adaptation (LoRA)**
- Reduces end-to-end runtime via early peptide-level pruning
- Interpretable attributions and sequence logos recover canonical anchors and align with peptide–HLA structural contacts
- Provides training scripts to train your own models

### Status

- Published on Hugging Face:
  - [TransHLA2.0-PRE](https://huggingface.co/SkywalkerLu/TransHLA2.0-PRE)(Skywalkerlu/TransHLA2.0-PRE)
  - [TransHLA2.0-BIND](https://huggingface.co/SkywalkerLu/TransHLA2.0-BIND) (SkywalkerLu/TransHLA2.0-BIND)
  - [TransHLA2.0-IM](https://huggingface.co/SkywalkerLu/TransHLA2.0-IM) (SkywalkerLu/TransHLA2.0-IM)
- Local repo scripts: models.py, utils.py, train_val.py, infer.py
- Data: all data published in [Google Drive](https://drive.google.com/drive/folders/1gYV1WJGtCXSvKGnfUIkra_P6BwXwS4i4?usp=sharing)



## 1. Environment and Installation

Requirements:
- Python ≥ 3.8 (≥ 3.9 recommended for training scripts)
- torch ≥ 2.0
- transformers ≥ 4.40
- peft
- Optional (training scripts): pandas, numpy, scikit-learn, tqdm, matplotlib, seaborn

Install core libs:
- pip install torch transformers peft

Tip: install torch matching your CUDA first, e.g. CUDA 12.1:
- pip install torch --index-url https://download.pytorch.org/whl/cu121


## 2. Quick Start: TransHLA2.0-PRE

TransHLA2.0-PRE enriches epitope-like peptides in peptide-only setting and provides standardized preprocessing for the pipeline. This stage enables early peptide-level pruning, reducing end-to-end runtime by filtering non-epitope candidates before expensive binding/immunogenicity predictions.



### Usage Example

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "SkywalkerLu/TransHLA2.0-PRE"
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()

# Load tokenizer used in training (ESM2 650M)
tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
peptide = "GILGFVFTL"
PEP_LEN = 16
PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else 1

def pad_to_len(ids_list, target_len, pad_id):
    return ids_list + [pad_id] * (target_len - len(ids_list)) if len(ids_list) < target_len else ids_list[:target_len]
pep_ids = tok(peptide, add_special_tokens=True)["input_ids"]
pep_ids = pad_to_len(pep_ids, PEP_LEN, PAD_ID)
pep_tensor = torch.tensor([pep_ids], dtype=torch.long, device=device)
with torch.no_grad():
    logits, features = model(pep_tensor)
    prob_bind = F.softmax(logits, dim=1)[:, 1]
    pred = int(prob_bind >= 0.5)

print({"peptide": peptide, "pre_prob": round(prob_bind, 6), "label": pred})
```

## 3. Quick Start: TransHLA2.0-BIND
TransHLA2.0-BIND is a minimal Hugging Face-compatible PyTorch model for peptide–HLA binding classification using ESM. It resolves allele-specific binding/presentation with quantitative supervision integrating eluted ligands and IC50-annotated pairs, achieving **AUROC of 96.2%** and **AUPRC of 95.2%** on combined BA/EL evaluation.

Inference workflow:
1) tokenize peptide and HLA pseudosequence with ESM tokenizer
2) pad/truncate to fixed lengths (default peptide=16, HLA=36)
3) forward pass to get logits and features
4) apply softmax to obtain binding probability

### Usage Example

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "SkywalkerLu/TransHLA2.0-BIND"
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()

tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

peptide = "GILGFVFTL"  # 9-mer example
hla_pseudoseq = "YYSEYRNIYAQTDESNLYLSYDYYTWAERAYEWY"  # demo pseudosequence

PEP_LEN = 16
HLA_LEN = 36
PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else 1

def pad_to_len(ids_list, target_len, pad_id):
    return (ids_list + [pad_id] * (target_len - len(ids_list))) if len(ids_list) < target_len else ids_list[:target_len]

pep_ids = tok(peptide, add_special_tokens=True)["input_ids"]
hla_ids = tok(hla_pseudoseq, add_special_tokens=True)["input_ids"]

pep_ids = pad_to_len(pep_ids, PEP_LEN, PAD_ID)
hla_ids = pad_to_len(hla_ids, HLA_LEN, PAD_ID)

pep_tensor = torch.tensor([pep_ids], dtype=torch.long, device=device)
hla_tensor = torch.tensor([hla_ids], dtype=torch.long, device=device)

with torch.no_grad():
    logits, features = model(pep_tensor, hla_tensor)  # logits: [1, 2]
    prob_bind =logits[0][1].item()
    pred = int(prob_bind >= 0.5)

print({"peptide": peptide, "bind_prob": round(prob_bind, 6), "label": pred})
```



## 4. Quick Start: TransHLA2.0-IM

TransHLA2.0-IM identifies immunogenic ligands from rigorously curated human T cell assays. The model architecture leverages:

- **Dual LoRA-ESM encoders** (peptide and HLA) for efficient parameter adaptation using Low-Rank Adaptation (LoRA)
- **Per-stream Transformer encoders** for sequence representation
- **Stacked bi-directional cross-attention** mechanisms for peptide–HLA interaction modeling
- **Optional CNN branches** for local pattern recognition
- **MLP classifier head** for final prediction

The model achieves competitive performance with markedly fewer trainable parameters through LoRA, reducing memory footprint while maintaining discrimination power. Training/inference scripts are included locally (models.py, train_val.py, infer.py).


### Usage Example

```python
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "SkywalkerLu/TransHLA2.0-IM"
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

# Tokenize and pad as in Section 3 (TransHLA2.0-BIND)
peptide = "GILGFVFTL"
hla_pseudoseq = "YYSEYRNIYAQTDESNLYLSYDYYTWAERAYEWY"

PEP_LEN = 16
HLA_LEN = 36
PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else 1

def pad_to_len(ids_list, target_len, pad_id):
    return (ids_list + [pad_id] * (target_len - len(ids_list))) if len(ids_list) < target_len else ids_list[:target_len]

pep_ids = tok(peptide, add_special_tokens=True)["input_ids"]
hla_ids = tok(hla_pseudoseq, add_special_tokens=True)["input_ids"]
pep_ids = pad_to_len(pep_ids, PEP_LEN, PAD_ID)
hla_ids = pad_to_len(hla_ids, HLA_LEN, PAD_ID)

pep_tensor = torch.tensor([pep_ids], dtype=torch.long, device=device)
hla_tensor = torch.tensor([hla_ids], dtype=torch.long, device=device)

with torch.no_grad():
    logits, features = model(pep_tensor, hla_tensor)
    prob_immunogenic = logits[0][1].item()

print({"peptide": peptide, "immunogenic_prob": round(prob_immunogenic, 6)})
```


## 5. Data Conventions (PRE)(For Trainging)

TransHLA2.0 uses standardized inputs for consistent processing across all stages:

### Input Format

- **peptide**: amino-acid string (uppercased), typically 8–14 residues
- **pseudosequence**: HLA pseudo-sequence (length typically 34)
- **label**: binary (0/1) for binding/immunogenicity classification

### Tokenization and Lengths

- **Tokenizer**: `facebook/esm2_t33_650M_UR50D` (ESM-2 tokenizer)
- **Fixed lengths** (defaults): peptide = 16, HLA = 36
- Add special tokens from ESM tokenizer (CLS, SEP), then pad/truncate to fixed lengths
- **Pad token id**: `tokenizer.pad_token_id` (fallback to 1 if None)

### File Format (TSV)

Required columns: `peptide`, `pseudosequence` (or `hla_pseudo`), `label`

Example data files:
  - `data/Example_train.txt`


## 6. Local Project Structure

project/
├─ models.py # Model definitions (TransHLA2_0_IM + ablations)
├─ utils.py # Data loading, tokenization, samplers, loss, metrics
├─ train_val.py # Training + validation (early stopping, best checkpoint)
├─ infer.py # Inference/testing (metrics, plots, CSV)
├─ requirements.txt # Dependencies
├─ data/
│ ├─ Example_train.txt
│ ├─ Example_val.txt
│ └─ Example_test.txt
├─ checkpoints/ # Saved model weights
└─ output/ # Plots and CSV outputs


On first run, pretrained assets are auto-downloaded:
- Tokenizer: facebook/esm2_t33_650M_UR50D
- Backbone (example for IM ablations): facebook/esm2_t12_35M_UR50D

## 7. Training Your Own Model (Local)

TransHLA2.0 models are trained on diverse IEDB ligands and IC50-annotated peptide–HLA pairs, capturing allele-discordant cases that sharpen specificity while maintaining well-calibrated operating points. You can train custom models using the provided scripts.

### Training TransHLA2.0-BIND

Run training + validation with the BIND model:

```bash
python train_val.py \
  --model_name TransHLA2_0_BIND \
  --epochs 100 \
  --batch_size 32 \
  --lr 1e-5 \
  --save_dir checkpoints \
  --save_prefix TransHLA2_0_BIND_best.pt
```

### Training TransHLA2.0-IM

Train with the IM model (supports LoRA for efficient training):

```bash
python train_val.py \
  --model_name TransHLA2_0_IM \
  --epochs 100 \
  --batch_size 32 \
  --lr 1e-5 \
  --save_dir checkpoints \
  --save_prefix TransHLA2_0_IM_best.pt
```

Use standard shuffle (no weighted sampler):

```bash
python train_val.py --model_name TransHLA2_0_BIND --balanced --epochs 100 --batch_size 32
```


### Training Arguments

Key arguments for `train_val.py`:

**Data paths:**
- `--train_path`, `--val_path`, `--test_path`: TSV file paths

**Model selection:**
- `--model_name`: Choose from `{TransHLA2_0_IM, TransHLA2_0_BIND, NoCNN, NoTransformer, NoCrossAttention}`

**Training hyperparameters:**
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-5)
- `--weight_decay`: Weight decay for regularization
- `--patience`: Early stopping patience on validation AUC (default: 5)

**Training options:**
- `--balanced`: Use standard shuffle instead of weighted sampler (for balanced datasets)
- `--save_dir`: Directory to save checkpoints
- `--save_prefix`: Checkpoint filename prefix
- `--device`: `cuda` or `cpu` (auto-detected if omitted)

### Training Tips

**Loss function**: CrossEntropy with optional entropy regularization and smoothing (see `utils.get_val_loss`)

**Output format**: Prefer raw logits during training; apply softmax only at evaluation/inference time

**Fixed lengths**: Keep `pep_len=16, hla_len=36` unless retraining from scratch with different lengths

**Memory optimization**: 
- Try ablation variants (NoCNN/NoCrossAttention) for lower memory usage
- Use mixed precision training (`torch.cuda.amp`)
- Implement gradient accumulation for effective larger batch sizes

**Class imbalance**: Weighted sampler is default; use `--balanced` flag to disable if your dataset is balanced

**Transfer learning**: Start by training only the classifier head, then unfreeze encoders later for fine-tuning

**LoRA efficiency**: TransHLA2.0-IM uses Low-Rank Adaptation (LoRA) to reduce trainable parameters while maintaining performance

## 8. Inference and Evaluation (Local)

Evaluate trained models on test sets to obtain performance metrics including accuracy, AUC, MCC, F1, recall, and precision. The inference script generates comprehensive outputs including CSV files with predictions and visualization plots.

### Inference with TransHLA2.0-BIND

Run inference on the test set:

```bash
python infer.py \
  --model_name TransHLA2_0_BIND \
  --checkpoint checkpoints/TransHLA2_0_BIND_best.pt \
  --output_dir output
```

### Inference with TransHLA2.0-IM

Run inference with the IM model:

```bash
python infer.py \
  --model_name TransHLA2_0_IM \
  --checkpoint checkpoints/TransHLA2_0_IM_best.pt \
  --output_dir output
```

### Evaluation Outputs

The inference script generates comprehensive evaluation results:

**Console metrics**: Accuracy, AUC-ROC, MCC, F1-score, Recall, Precision

**CSV file**: `output/results_<test_filename>.csv` containing:
- `Pred_Prob_0`: Predicted probability for class 0
- `Pred_Prob_1`: Predicted probability for class 1
- `Pred_Label`: Predicted binary label
- `True_Label`: Ground truth label
- All original columns from the input file

**Visualization plots** (in `output/` directory):
- `roc_curve.png`: Receiver Operating Characteristic curve
- `pr_curve.png`: Precision-Recall curve
- `confusion_matrix.png`: Confusion matrix visualization

These outputs enable detailed analysis of model performance and interpretability.
## 9. Reproducibility

To ensure reproducible results across different runs:

**Random seed control**: Use `utils.set_seed()` to fix random seeds for PyTorch, NumPy, and Python's random module

**Environment documentation**: 
- Record dependency versions using `pip freeze > requirements.txt`
- Document all CLI arguments used for training and inference

**Checkpoint management**: 
- Place trained checkpoints (e.g., `TransHLA2_0_BIND_best.pt`, `TransHLA2_0_IM_best.pt`) under `checkpoints/` directory
- Reference checkpoints via `--checkpoint` argument during inference
- Include checkpoint metadata (training date, hyperparameters, dataset version) in checkpoint filenames or separate log files

**Data versioning**: Keep track of dataset versions and preprocessing steps to enable exact replication of results



