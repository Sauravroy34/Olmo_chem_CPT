# OLMo-7B Chemistry Continual Pre-Training (CPT)

Continual pre-training of [OLMo-7B](https://huggingface.co/allenai/OLMo-7B-hf) on chemistry SMILES strings using three training strategies: **Full Fine-Tuning (FSDP)**, **LoRA**, and **QLoRA**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [File Structure](#file-structure)
- [Pipeline Overview](#pipeline-overview)
- [Prerequisites](#prerequisites)
- [How to Run](#how-to-run)
- [How Training Sample Size Is Calculated](#how-training-sample-size-is-calculated)
- [Configuration Reference](#configuration-reference)
- [Weights & Biases Logging](#weights--biases-logging)

---

## Project Overview

This project adapts OLMo-7B for chemistry by:

1. **Extending the tokenizer** with ~300 SMILES Pair Encoding (SPE) tokens learned from ZINC20 + ChEMBL, plus two special tokens (`<|start_of_smiles|>`, `<|end_of_smiles|>`).
2. **Resizing the model** (`embed_tokens` and `lm_head`) and **mean-initializing** new token embeddings from the original OLMo sub-token embeddings.
3. **Continual pre-training** on a mixed chemistry corpus (SMILES, abstracts, SMILES-description pairs) using one of three methods.

The base model after steps 1–2 is published as [`Codemaster67/Olmo-7b-spe`](https://huggingface.co/Codemaster67/Olmo-7b-spe) and serves as the starting checkpoint for all three training scripts.

---

## File Structure

```
Olmo_chem_CPT/
│
├── intial_pre/
│   ├── DataPrep.py                     # Dataset preparation — combines 4 sources, pushes to HF Hub
│   └── Model_init.py                   # Resizes model embeddings for the new tokenizer (mean init)
│
├── SPE_Tokenizer_generation/
│   ├── OLMO_NLP_tokenizer_extended.py  # Trains SPE tokenizer on ZINC20+ChEMBL, extends OLMo tokenizer
│   └── Spec_300.txt                    # The 300 learned SPE chemistry tokens
│
├── training_scripts/
│   ├── FullFinetune.py                 # Full-parameter fine-tuning with FSDP
│   ├── PreTrainLora.py                 # LoRA adapter training (r=64, alpha=128)
│   └── PreTrainQlora.py               # QLoRA adapter training (4-bit NF4 + LoRA)
│
└── README.md                           # This file
```

### File Descriptions

| File | Purpose |
|---|---|
| **`DataPrep.py`** | Combines 4 data sources into a single HF dataset: 1M SMILES from UniChem, 30k ChemRxiv abstracts, 500k ChEBI-20 SMILES-description pairs (5 prompt templates), and 20k PubMed abstracts. Pushes to `Codemaster67/Causal_lm_chemistry_1M_rows`. |
| **`SPE_Tokenizer_generation/OLMO_NLP_tokenizer_extended.py`** | Trains a SMILES Pair Encoding (SPE) tokenizer on 2M ZINC20 + 2M ChEMBL molecules, selects the top 300 SPE tokens, and merges them into the OLMo-7B tokenizer. Pushed to `Codemaster67/Olmo_spe_tokenizer_300SPE_TOKENS`. |
| **`SPE_Tokenizer_generation/Spec_300.txt`** | The 300 SPE tokens selected for the extended tokenizer. |
| **`Model_init.py`** | Loads the vanilla OLMo-7B, resizes `embed_tokens` and `lm_head` for the new vocabulary, and initializes each new token's embedding as the **mean** of its original sub-token embeddings. Pushed to `Codemaster67/Olmo-7b-spe`. |
| **`FullFinetune.py`** | Full-parameter CPT using HuggingFace Trainer with **FSDP** (Fully Sharded Data Parallel). All model weights are updated. Uses a conservative learning rate (5e-6). |
| **`PreTrainLora.py`** | LoRA (Low-Rank Adaptation) CPT. Trains small adapter matrices (r=64, alpha=128) on all linear layers. `embed_tokens` and `lm_head` are saved via `modules_to_save`. |
| **`PreTrainQlora.py`** | QLoRA CPT. The base model is loaded in 4-bit (NF4 via bitsandbytes) and LoRA adapters are trained on top. Most memory-efficient option. |

---

## Pipeline Overview

```
┌─────────────────────┐     ┌──────────────────────────────────┐     ┌──────────────┐
│  DataPrep.py        │     │  SPE_Tokenizer_generation/       │     │ Model_init.py│
│  (Build dataset)    │     │  (Train SPE + extend tokenizer)  │     │ (Resize +    │
│                     │     │                                  │     │  mean init)  │
└────────┬────────────┘     └───────────────┬──────────────────┘     └──────┬───────┘
         │                                  │                              │
         │  Codemaster67/                   │  Codemaster67/               │ Codemaster67/
         │  Causal_lm_chemistry_1M_rows     │  Olmo_spe_tokenizer_...      │ Olmo-7b-spe (already in hugging face)
         ▼   (already in hugging face)      ▼  (already in hugging face)   ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         Training Scripts (pick one)                             │
│                                                                                │
│   FullFinetune.py  ─── FSDP, all params, lr=5e-6                               │
│   PreTrainLora.py  ─── LoRA r=64 α=128, bf16, gradient checkpointing          │
│   PreTrainQlora.py ─── QLoRA 4-bit NF4 + LoRA r=64 α=128                      │
│                                                                                │
│                     ↓ push to HuggingFace Hub ↓                                │
│                                              
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Log in to HuggingFace and Weights & Biases:

```bash
huggingface-cli login
wandb login
```

---

## How to Run

### Step 1 — Prepare the Dataset

```bash
python DataPrep.py
```

This builds the combined chemistry dataset and pushes it to `Codemaster67/Causal_lm_chemistry_1M_rows` on HuggingFace Hub.


#### Multi-GPU with `torchrun`

Use `torchrun` (PyTorch's distributed launcher) to run across multiple GPUs on a single node:

```bash
# Full fine-tuning with FSDP (e.g., 4 GPUs)
torchrun --nproc_per_node=4 FullFinetune.py

# LoRA (e.g., 2 GPUs)
torchrun --nproc_per_node=2 PreTrainLora.py

# QLoRA (e.g., 2 GPUs)
torchrun --nproc_per_node=2 PreTrainQlora.py
```

**`torchrun` flags:**

| Flag | Description |
|---|---|
| `--nproc_per_node=N` | Number of GPU processes per machine. Set to the number of GPUs you want to use. |
| `--nnodes=N` | Number of machines (default: 1). For multi-node, set >1. |
| `--node_rank=R` | Rank of this machine (0-indexed). Only needed for multi-node. |
| `--master_addr=ADDR` | IP of the rank-0 node. Only needed for multi-node. |
| `--master_port=PORT` | Port for distributed communication (default: 29500). |

**Example — Multi-node (2 machines × 4 GPUs each):**

```bash
# On machine 0 (master):
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
         --master_addr=192.168.1.100 --master_port=29500 \
         FullFinetune.py

# On machine 1:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
         --master_addr=192.168.1.100 --master_port=29500 \
         FullFinetune.py
```

> **Note:** `FullFinetune.py` uses FSDP (Fully Sharded Data Parallel) for multi-GPU training, which shards model parameters, gradients, and optimizer states across GPUs. `PreTrainLora.py` and `PreTrainQlora.py` use standard DDP via HuggingFace Trainer with gradient checkpointing.

---

## How Training Sample Size Is Calculated

Understanding the final training sample count requires following four stages:

### Stage 1: Dataset Loading & Split

The full dataset (`Codemaster67/Causal_lm_chemistry_1M_rows`) is loaded by combining all splits:

```python
full_dataset = load_dataset(DATASET_NAME, split="train+test")  # ~1.5M rows
```

It is then split **90/10** into train and validation:

```python
split_dataset = shuffled_dataset.train_test_split(test_size=0.1, seed=42)
# train: ~1.35M rows | val: ~150K rows
```

### Stage 2: Subsetting (optional)

If `USE_SUBSET = True` (the default), only the first N rows are kept:

```
Train: TRAIN_SUBSET_SIZE rows (default: 10,000)
Val:   EVAL_SUBSET_SIZE  rows (default: 1,000)
```

If `USE_SUBSET = False`, the entire train/val split is used.

### Stage 3: SMILES Augmentation (optional)

When `AUGMENT = True`, only rows from the **"unichem"** source (~65% of the dataset) are augmented. For each unichem SMILES, RDKit generates up to `NUM_AUGMENT` (default: 4) randomized SMILES variants (random atom orderings of the same molecule). These variants are **concatenated** with the original text in-place.

**Augmentation is applied BEFORE tokenization, so it increases the total token count, not the row count.**

The approximate **token multiplication factor** from augmentation:

```
effective_tokens ≈ original_tokens + (unichem_fraction × NUM_AUGMENT × avg_smiles_tokens)
                 ≈ original_tokens × (1 + 0.65 × NUM_AUGMENT)
```

| Script | `AUGMENT` default | Effect |
|---|---|---|
| `FullFinetune.py` | `False` | No augmentation — raw rows only |
| `PreTrainLora.py` | `False` | No augmentation — raw rows only |
| `PreTrainQlora.py` | `True` | ~3.6× more tokens from unichem rows |

### Stage 4: Tokenization & Sequence Packing

After tokenization, all token sequences are **packed** (concatenated end-to-end and chunked into fixed-length blocks):

```python
MAX_SEQ_LENGTH = 512  # tokens per packed chunk
```

The packing algorithm:

1. Concatenate all `input_ids` from every row into one long flat list.
2. Divide into chunks of exactly `MAX_SEQ_LENGTH` tokens.
3. Discard any remainder shorter than `MAX_SEQ_LENGTH`.

```
final_training_examples = floor(total_tokens / MAX_SEQ_LENGTH)
```

### Putting It All Together

```
┌──────────────────────────────────────────────────────────────────────┐
│  Full dataset (~1.5M rows)                                         │
│       ↓  90/10 split                                               │
│  Train: ~1.35M rows    Val: ~150K rows                             │
│       ↓  subset (if USE_SUBSET=True)                               │
│  Train: 10,000 rows    Val: 1,000 rows                            │
│       ↓  augment (if AUGMENT=True, unichem rows only)              │
│  Token count ≈ rows × avg_tokens × (1 + 0.65 × NUM_AUGMENT)       │
│       ↓  tokenize                                                  │
│  Total tokens across all rows                                      │
│       ↓  pack into chunks of MAX_SEQ_LENGTH (512)                  │
│  final_examples = floor(total_tokens / 512)                        │
└──────────────────────────────────────────────────────────────────────┘
```

**Example calculation (default settings, no augmentation):**

```
10,000 rows × ~40 avg tokens per SMILES ≈ 400,000 total tokens
400,000 / 512 ≈ 781 packed training examples
```

**Example calculation (QLoRA defaults, augmentation ON):**

```
10,000 rows
 → ~6,500 unichem rows × 4 augmented variants each = +26,000 SMILES strings in-text
 → ~3,500 non-unichem rows unchanged
Total tokens ≈ 400,000 + (26,000 × 40) ≈ 1,440,000
1,440,000 / 512 ≈ 2,812 packed training examples
```

> **Important:** The final number of training examples seen by the Trainer is the **packed chunk count**, not the original row count. This is printed at startup:
> ```
> Packed training sequences:   781
> Packed validation sequences: 78
> ```

---

## Configuration Reference

All hyperparameters are defined as constants at the top of each training script. Key parameters:

| Parameter | FullFinetune | LoRA | QLoRA |
|---|---|---|---|
| Learning Rate | 5e-6 | 1e-5 | 1e-5 |
| Batch Size | 32 | 32 | 32 |
| Max Seq Length | 512 | 512 | 512 |
| Epochs | 1 | 1 | 1 |
| Warmup Ratio | 0.1 | 0.1 | 0.1 |
| Weight Decay | 0.01 | 0.01 | 0.01 |
| Optimizer | AdamW | AdamW 8-bit | AdamW 8-bit |
| Precision | bf16 | bf16 | bf16 adapters / 4-bit NF4 base |
| LoRA Rank | — | 64 | 64 |
| LoRA Alpha | — | 128 | 128 |
| FSDP | ✅ | ❌ | ❌ |
| Gradient Checkpointing | ✅ (activation) | ✅ | ✅ |
| AUGMENT | False | False | True |

---

## Weights & Biases Logging

All scripts log to W&B with contextual, timestamped project and run names:

| Script | Example Project Name | Example Run Name |
|---|---|---|
| `FullFinetune.py` | `OLMo_Full_Fine_tune_20260628_044100` | `Full_Fine_tune_lr5e-06_samples10000_20260628_044100` |
| `PreTrainLora.py` | `OLMo_LoRA_r64_a128_20260628_044100` | `lora_r64_a128_lr1e-05_20260628_044100` |
| `PreTrainQlora.py` | `OLMo_QLoRA_r64_a128_20260628_044100` | `qlora_r64_a128_lr1e-05_20260628_044100` |

**Tracked metrics:**
- `train/loss` — training loss (every `LOGGING_STEPS` steps)
- `eval_loss` — evaluation loss (every `EVAL_STEPS` steps)
- `eval_perplexity` — evaluation perplexity (explicitly logged via custom `CLMTrainerWithPerplexity` class)

Each training script also generates and pushes a **Model Card** to the HuggingFace Hub alongside the model/adapter, documenting the training configuration, results, and usage instructions.
