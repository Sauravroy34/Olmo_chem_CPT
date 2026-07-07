# OLMo-7B Chemistry Continual Pre-Training (CPT)

Continual pre-training of [OLMo-7B](https://huggingface.co/allenai/OLMo-7B-hf) on chemistry SMILES strings. Three training strategies are supported: **Full Fine-Tuning (FSDP)**, **LoRA**, and **QLoRA**.

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

This project teaches OLMo-7B chemistry by:

1. **Extending the tokenizer** — adds ~300 SMILES Pair Encoding (SPE) tokens learned from ZINC20 + ChEMBL, plus two special tokens (`<|start_of_smiles|>`, `<|end_of_smiles|>`).
2. **Resizing the model** — expands `embed_tokens` and `lm_head` for the new vocabulary. Each new token embedding is set to the **mean** of its original sub-token embeddings.
3. **Continual pre-training** — trains on a mixed chemistry corpus (SMILES, abstracts, SMILES-description pairs) using one of three methods.

The base model (after steps 1–2) is published at [`Codemaster67/Olmo-7b-spe`](https://huggingface.co/Codemaster67/Olmo-7b-spe). All three training scripts start from this checkpoint.

---

## File Structure

```
Olmo_chem_CPT/
│
├── intial_pre/
│   ├── DataPrep.py                     # Builds the dataset, pushes to HF Hub
│   └── Model_init.py                   # Resizes model embeddings (mean init)
│
├── SPE_Tokenizer_generation/
│   ├── OLMO_NLP_tokenizer_extended.py  # Trains SPE tokenizer, extends OLMo tokenizer
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

| File | What It Does |
|---|---|
| **`DataPrep.py`** | Combines 4 data sources into one HF dataset: 1M SMILES from UniChem, 30k ChemRxiv abstracts, 500k ChEBI-20 SMILES-description pairs (5 prompt templates), and 20k PubMed abstracts. Pushes to `Codemaster67/Causal_lm_chemistry_1M_rows`. |
| **`OLMO_NLP_tokenizer_extended.py`** | Trains an SPE tokenizer on 2M ZINC20 + 2M ChEMBL molecules, picks the top 300 SPE tokens, and merges them into the OLMo-7B tokenizer. Pushes to `Codemaster67/Olmo_spe_tokenizer_300SPE_TOKENS`. |
| **`Spec_300.txt`** | The 300 SPE tokens used in the extended tokenizer. |
| **`Model_init.py`** | Loads vanilla OLMo-7B, resizes `embed_tokens` and `lm_head` for the new vocabulary, and sets each new token's embedding to the **mean** of its original sub-token embeddings. Pushes to `Codemaster67/Olmo-7b-spe`. |
| **`FullFinetune.py`** | Full-parameter CPT using HuggingFace Trainer with **FSDP**. All model weights are updated. Uses a low learning rate (5e-6). |
| **`PreTrainLora.py`** | LoRA CPT. Trains small adapter matrices (r=64, alpha=128) on all linear layers. `embed_tokens` and `lm_head` are saved via `modules_to_save`. |
| **`PreTrainQlora.py`** | QLoRA CPT. Loads the base model in 4-bit (NF4 via bitsandbytes) and trains LoRA adapters on top. Most memory-efficient option. |

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
         │  Causal_lm_chemistry_1M_rows     │  Olmo_spe_tokenizer_...      │ Olmo-7b-spe
         ▼  (on HuggingFace)               ▼  (on HuggingFace)            ▼ (on HuggingFace)
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

Log in to HuggingFace and Weights & Biases before running anything:

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

### Step 2 — Train (Multi-GPU with `torchrun`)

Use `torchrun` to run across multiple GPUs on a single machine:

```bash
# Full fine-tuning with FSDP (e.g., 4 GPUs)
torchrun --nproc_per_node=4 FullFinetune.py

# LoRA (e.g., 2 GPUs)
torchrun --nproc_per_node=2 PreTrainLora.py

# QLoRA (e.g., 2 GPUs)
torchrun --nproc_per_node=2 PreTrainQlora.py
```

**`torchrun` flags:**

| Flag | What It Does |
|---|---|
| `--nproc_per_node=N` | Number of GPU processes per machine. Set to your GPU count. |
| `--nnodes=N` | Number of machines (default: 1). Set >1 for multi-node. |
| `--node_rank=R` | Rank of this machine (0-indexed). Only for multi-node. |
| `--master_addr=ADDR` | IP of the rank-0 node. Only for multi-node. |
| `--master_port=PORT` | Port for distributed communication (default: 29500). |

**Multi-node example (2 machines × 4 GPUs each):**

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

> **Note:** `FullFinetune.py` uses FSDP (shards model weights, gradients, and optimizer states across GPUs). `PreTrainLora.py` and `PreTrainQlora.py` use standard DDP via HuggingFace Trainer with gradient checkpointing.

---

## How Training Sample Size Is Calculated

The final training sample count depends on four stages:

### Stage 1: Load and Split

The full dataset (`Codemaster67/Causal_lm_chemistry_1M_rows`) is loaded by combining all splits:

```python
full_dataset = load_dataset(DATASET_NAME, split="train+test")  # ~1.5M rows
```

Then split **90/10** into train and validation:

```python
split_dataset = shuffled_dataset.train_test_split(test_size=0.1, seed=42)
# train: ~1.35M rows | val: ~150K rows
```

### Stage 2: Subset (optional)

If `USE_SUBSET = True` (the default), only the first N rows are kept:

```
Train: TRAIN_SUBSET_SIZE rows (default: 10,000)
Val:   EVAL_SUBSET_SIZE  rows (default: 1,000)
```

If `USE_SUBSET = False`, the full train/val split is used.

### Stage 3: SMILES Augmentation (optional)

When `AUGMENT = True`, only rows from the **"unichem"** source (~65% of the dataset) are augmented. For each unichem SMILES, RDKit generates up to `NUM_AUGMENT` (default: 4) randomized SMILES variants (random atom orderings of the same molecule). These variants are added to the original text in-place.

**Augmentation happens before tokenization, so it increases total tokens, not row count.**

Approximate token multiplication:

```
effective_tokens ≈ original_tokens × (1 + 0.65 × NUM_AUGMENT)
```

| Script | `AUGMENT` default | Effect |
|---|---|---|
| `FullFinetune.py` | `False` | No augmentation |
| `PreTrainLora.py` | `False` | No augmentation |
| `PreTrainQlora.py` | `True` | ~3.6× more tokens from unichem rows |

### Stage 4: Tokenize and Pack

After tokenization, all token sequences are **packed** — concatenated end-to-end and chunked into fixed-length blocks:

```python
MAX_SEQ_LENGTH = 512  # tokens per packed chunk
```

How packing works:

1. Concatenate all `input_ids` from every row into one flat list.
2. Split into chunks of exactly `MAX_SEQ_LENGTH` tokens.
3. Drop any leftover shorter than `MAX_SEQ_LENGTH`.

```
final_training_examples = floor(total_tokens / MAX_SEQ_LENGTH)
```

### Full Picture

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

**Example — no augmentation (default):**

```
10,000 rows × ~40 avg tokens ≈ 400,000 total tokens
400,000 / 512 ≈ 781 packed training examples
```

**Example — QLoRA with augmentation ON:**

```
10,000 rows
 → ~6,500 unichem rows × 4 variants each = +26,000 SMILES strings
 → ~3,500 non-unichem rows unchanged
Total tokens ≈ 400,000 + (26,000 × 40) ≈ 1,440,000
1,440,000 / 512 ≈ 2,812 packed training examples
```

> **Important:** The Trainer sees the **packed chunk count**, not the original row count. This is printed at startup:
> ```
> Packed training sequences:   781
> Packed validation sequences: 78
> ```

---

## Configuration Reference

All hyperparameters are set as constants at the top of each training script. Key settings:

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

All scripts log to W&B with timestamped project and run names:

| Script | Example Project Name | Example Run Name |
|---|---|---|
| `FullFinetune.py` | `OLMo_Full_Fine_tune_20260628_044100` | `Full_Fine_tune_lr5e-06_samples10000_20260628_044100` |
| `PreTrainLora.py` | `OLMo_LoRA_r64_a128_20260628_044100` | `lora_r64_a128_lr1e-05_20260628_044100` |
| `PreTrainQlora.py` | `OLMo_QLoRA_r64_a128_20260628_044100` | `qlora_r64_a128_lr1e-05_20260628_044100` |

**Tracked metrics:**
- `train/loss` — training loss (every `LOGGING_STEPS` steps)
- `eval_loss` — validation loss (every `EVAL_STEPS` steps)
- `eval_perplexity` — validation perplexity (logged by the custom `CLMTrainerWithPerplexity` class)

Each training script also generates a **Model Card** and pushes it to HuggingFace Hub with the model/adapter.
