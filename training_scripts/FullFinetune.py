"""
FullFinetune.py — Full-parameter continual pre-training for OLMo-7B on chemistry SMILES.

Usage examples with torchrun
============================

    torchrun --nproc_per_node=<NUM_GPUS> FullFinetune.py \
        --hf_token "hf_XXXXX" \
        --wandb_key "your_wandb_key_here" \
        --hf_repo_id "YourUser/YourModel" \
        --train_subset_size 250000

"""
import os
import math
import random
import argparse
import torch
from functools import partial
from itertools import chain
from datetime import datetime

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from huggingface_hub import ModelCard, ModelCardData
from rdkit import Chem 
import wandb

from huggingface_hub import login

# ── Special tokens (not configurable) ──────────────────────────────
SMILES_START = "<|start_of_smiles|>"
SMILES_END = "<|end_of_smiles|>"
EOS = "<|endoftext|>"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments. Tokens/repo are user-provided; everything else has sensible defaults."""
    parser = argparse.ArgumentParser(
        description="Full-parameter continual pre-training for OLMo-7B on chemistry SMILES.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--cpu_offload", type=bool, default=False,
                        help="Enable CPU offloading.")

    parser.add_argument("--activation_checkpointing", type=bool, default=False,
                        help="Enable activation checkpointing.")
                        
    # ── Credentials & repo (user must supply these) ────────────────
    parser.add_argument("--hf_token", type=str, required=True,
                        help="HuggingFace API token for login and pushing to hub.")
    parser.add_argument("--wandb_key", type=str, required=True,
                        help="Weights & Biases API key.")
    parser.add_argument("--hf_repo_id", type=str, required=True,
                        help="HuggingFace Hub repo ID to push the model (e.g. 'YourUser/YourModel').")

    # ── Dataset / model ────────────────────────────────────────────
    parser.add_argument("--dataset_name", type=str,
                        default="Codemaster67/Causal_lm_chemistry_1M_rows",
                        help="HuggingFace dataset ID (default: Codemaster67/Causal_lm_chemistry_1M_rows).")
    parser.add_argument("--base_model", type=str,
                        default="Codemaster67/Olmo-7b-spe",
                        help="Base model ID with extended tokenizer (default: Codemaster67/Olmo-7b-spe).")
    parser.add_argument("--output_dir", type=str,
                        default="./olmo_chem_full_cpt_5e-6_lr",
                        help="Local directory for checkpoints and model (default: ./olmo_chem_full_cpt_5e-6_lr).")

    # ── Training hyperparameters ───────────────────────────────────
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs (default: 1).")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Learning rate (default: 5e-6, conservative for full fine-tuning).")
    parser.add_argument("--batch_size", type=int, default=32, help="Per-device batch size (default: 32).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps (default: 1).")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Max packed sequence length (default: 512).")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio (default: 0.1).")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay (default: 0.01).")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging interval in steps (default: 10).")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluation interval in steps (default: 1000).")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save interval in steps (default: 1000).")
    parser.add_argument("--save_total_limit", type=int, default=1,
                        help="Maximum number of checkpoint saves to keep (default: 1).")

    # ── Augmentation ───────────────────────────────────────────────
    parser.add_argument("--augment", action="store_true", default=False,
                        help="Enable SMILES augmentation (default: disabled).")
    parser.add_argument("--num_augment", type=int, default=4,
                        help="Number of SMILES augmentations per molecule (default: 4).")

    # ── Subset control ─────────────────────────────────────────────
    parser.add_argument("--use_subset", action="store_true", default=True,
                        help="Use a subset of the dataset (default: True).")
    parser.add_argument("--no_use_subset", action="store_false", dest="use_subset",
                        help="Use the full dataset instead of a subset.")
    parser.add_argument("--train_subset_size", type=int, default=250000,
                        help="Training subset size (default: 250000). Ignored when --no_use_subset.")
    parser.add_argument("--eval_subset_size", type=int, default=25000,
                        help="Eval subset size (default: 25000). Ignored when --no_use_subset.")
    parser.add_argument("--save_strategy", action="store_true", default=False,
                        help="Whether to have checkpoints or not")
                        
    return parser.parse_args()


def print_main(message):
    """prints only for rank 0"""
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        print(message)


def wandb_log(metrics_dict, step):
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        wandb.log(metrics_dict, step=step)


def setup_tokenizer(tokenizer_id: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def augment_smiles(smiles, num_augmentations: int):
    smiles = smiles.replace("<|start_of_smiles|>", "").replace("<|end_of_smiles|>", "")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]  

    augmented_set = set()
    attempts = 0
    max_attempts = num_augmentations * 2  

    while len(augmented_set) < num_augmentations and attempts < max_attempts:
        rand_smiles = Chem.MolToSmiles(mol, doRandom=True)
        augmented_set.add(rand_smiles)
        attempts += 1

    del mol 
    return list(augmented_set)


def setup_model(base_model: str):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # requires flash attention library
    )
    return model


def tokenize_function(examples: dict, tokenizer: AutoTokenizer) -> dict:
    texts = [t for t in examples["text"]] 
    return tokenizer(
        texts,
        truncation=False,  
        return_attention_mask=False,
    )


def pack_sequences(tokenized_dataset, tokenizer, max_seq_length: int):
    def group_texts(examples):
        # source https://huggingface.co/docs/transformers/en/tasks/language_modeling?utm_source=chatgpt.com
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])

        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length

        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated.items()
        }
        # result["labels"] = result["input_ids"].copy() data_collator handles the label part 
        return result

    packed = tokenized_dataset.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
        remove_columns=tokenized_dataset.column_names,
        desc="Packing sequences",
    )
    return packed


def prepare_datasets(tokenizer, args: argparse.Namespace):
    full_dataset = load_dataset(args.dataset_name, split="train+test") 
    shuffled_dataset = full_dataset.shuffle(seed=args.seed)

    split_dataset = shuffled_dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    if args.use_subset:
        print_main(f"Using a subset of the training data: {args.train_subset_size} examples")
        train_dataset = train_dataset.select(range(args.train_subset_size))
        
        print_main(f"Using a subset of the validation data: {args.eval_subset_size} examples")
        val_dataset = val_dataset.select(range(int(args.eval_subset_size)))

    Source_name = "unichem"  
    if args.augment:
        num_aug = args.num_augment
        def format_smiles(example):
            input_smiles = example["text"] if example["source"] == Source_name else ""
            if input_smiles != "":
                augmented_list = augment_smiles(input_smiles, num_aug)
                formatted_smiles = [f"{SMILES_START}{smiles}{SMILES_END}{EOS}" for smiles in augmented_list]
                formatted_smiles.append(input_smiles)  
                example["text"] = "".join(formatted_smiles) 
            return example

        train_dataset = train_dataset.map(format_smiles, num_proc=4, desc="Formatting smiles")
        
    random.seed(args.seed)

    tokenize_fn = partial(tokenize_function, tokenizer=tokenizer)

    dataset = DatasetDict({
        "train": train_dataset,
        "test": val_dataset
    })

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        num_proc=4,
        remove_columns=["text", "source"],
        desc="Tokenizing",
    )

    train_packed = pack_sequences(tokenized["train"], tokenizer, args.max_seq_length)
    val_packed = pack_sequences(tokenized["test"], tokenizer, args.max_seq_length)

    print_main(f"Packed training sequences:   {len(train_packed)}")
    print_main(f"Packed validation sequences: {len(val_packed)}")

    sample_ids = train_packed[0]["input_ids"][:200]
    print_main(f"Sample packed text (first 200 tokens):\n{tokenizer.decode(sample_ids)}")

    return train_packed, val_packed

class CLMTrainerWithPerplexity(Trainer):
    """Extends HuggingFace Trainer to calculate perplexity and explicitly log metrics manually."""

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        eval_loss_key = f"{metric_key_prefix}_loss"
        if eval_loss_key in metrics:
            try:
                perplexity = math.exp(metrics[eval_loss_key])
            except OverflowError:
                perplexity = float("inf")
            metrics[f"{metric_key_prefix}_perplexity"] = perplexity

            # Explicitly log eval perplexity to wandb (rank 0 only)
            if int(os.getenv("LOCAL_RANK", "0")) == 0 and wandb.run is not None:
                wandb.log({
                    f"{metric_key_prefix}_perplexity": perplexity,
                    f"{metric_key_prefix}_loss": metrics[eval_loss_key],
                }, step=self.state.global_step)

        return metrics



def main():
    args = parse_args()

    # ── Login with user-provided tokens ────────────────────────────
    login(token=args.hf_token)
    wandb.login(key=args.wandb_key)
    os.environ["WANDB_LOG_MODEL"] = "false"

    set_seed(args.seed)
    is_main_process = int(os.getenv("LOCAL_RANK", "0")) == 0

    wandb_project = f"Saurav_OLMo_Full_Fine_tune_and_lora"
    wandb_run_name = f"Full_Fine_tune_lr{args.learning_rate}_samples{args.train_subset_size}_batch_size{args.batch_size}"

    if is_main_process:
        wandb.init(project=wandb_project, name=wandb_run_name)

     
    if is_main_process:
        wandb.log({"Train subset size": args.train_subset_size})

    tokenizer = setup_tokenizer(args.base_model)
    model = setup_model(args.base_model)

    train_dataset, val_dataset = prepare_datasets(tokenizer, args)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    transformer_layer_cls_name = type(model.model.layers[0]).__name__

    # more about the args https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L180
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_torch",  
        bf16=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps" if args.save_strategy else "no",
        save_total_limit=args.save_total_limit, #only save last N models 
        save_steps=args.save_steps ,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to="wandb", 
        run_name=wandb_run_name,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        push_to_hub=False,
        seed=args.seed,
        remove_unused_columns=False,
        #More about fsdp https://github.com/huggingface/transformers/blob/e0e7504bca2bfd1b85bb0eedb148f7b250226f06/src/transformers/training_args.py#L650
        fsdp=True,
        fsdp_config={
            "transformer_layer_cls_to_wrap": transformer_layer_cls_name,
            "activation_checkpointing": args.activation_checkpointing,
            "cpu_offload" : args.cpu_offload
        }
    )

    trainer = CLMTrainerWithPerplexity(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    train_result = trainer.train()

    
    print_main("Running final evaluation...")
    final_metrics = trainer.evaluate()
    FINAL_SAVE_DIR = os.path.join(args.output_dir, "final_model_push")
    trainer.save_model(FINAL_SAVE_DIR)

    if is_main_process: 
        print_main("Generating Model Card...")
    
        tokenizer.save_pretrained(args.output_dir)

        final_ppl = final_metrics.get("eval_perplexity", "N/A")
        final_loss = final_metrics.get("eval_loss", "N/A")

        # ── Model Card ──────────────────────────────────────────────
        card_data = ModelCardData(
            language="en",
            license="apache-2.0",
            library_name="transformers",
            tags=["chemistry", "smiles", "olmo", "causal-lm", "full-finetune", "fsdp"],
            datasets=[args.dataset_name],
            base_model=args.base_model,
        )
        card_content = f"""---
{card_data.to_yaml()}
---

# OLMo-7B Full Fine-Tune — Chemistry SMILES CPT

## Model Description

This model is a **full-parameter fine-tuned** version of
[{args.base_model}](https://huggingface.co/{args.base_model}) trained on chemistry
SMILES strings from the
[{args.dataset_name}](https://huggingface.co/datasets/{args.dataset_name}) dataset.

The base model's tokenizer was pre-extended with ~300 SPE (SMILES Pair
Encoding) chemistry tokens plus `<|start_of_smiles|>` / `<|end_of_smiles|>`
special tokens, and its embedding & LM-head layers were resized with
mean-initialised vectors for the new tokens.

## Training Details

| Parameter | Value |
|---|---|
| **Method** | Full Fine-Tune (all weights updated) |
| **Parallelism** | FSDP (Fully Sharded Data Parallel) |
| **Epochs** | {args.num_epochs} |
| **Learning Rate** | {args.learning_rate} |
| **Batch Size (per device)** | {args.batch_size} |
| **Gradient Accumulation** | {args.gradient_accumulation_steps} |
| **Max Sequence Length** | {args.max_seq_length} |
| **Warmup Ratio** | {args.warmup_ratio} |
| **Weight Decay** | {args.weight_decay} |
| **Effective Batch Size (Batch Size {args.batch_size} x Gradient Accumulation {args.gradient_accumulation_steps})** | {args.batch_size * args.gradient_accumulation_steps} |
| **Scheduler** | Cosine |
| **Precision** | bf16 |
| **Augmentation** | {"ON (×" + str(args.num_augment) + " per unichem SMILES)" if args.augment else "OFF"} |
| **Training Samples** | {args.train_subset_size if args.use_subset else "Full dataset"} |
| **Eval Samples** | {args.eval_subset_size if args.use_subset else "Full dataset (10%)"} |

## Evaluation Results

| Metric | Value |
|---|---|
| **Final Eval Loss** | {final_loss} |
| **Final Eval Perplexity** | {final_ppl} |
| **Training Loss** | {train_result.training_loss:.4f} |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{args.hf_repo_id}", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("{args.hf_repo_id}", trust_remote_code=True)

smiles_input = "<|start_of_smiles|>CC(=O)Oc1ccccc1C(=O)O<|end_of_smiles|>"
inputs = tokenizer(smiles_input, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```

## Intended Use

Chemistry-domain language modelling, SMILES generation and completion,
and downstream molecular property prediction via fine-tuning.

## Limitations

- Trained primarily on SMILES strings; natural-language instruction-following
  ability may degrade compared to the base OLMo checkpoint.
- Augmentation was {"enabled" if args.augment else "disabled"} for this run.
"""
        model_card = ModelCard(card_content)
        # ────────────────────────────────────────────────────────────

        try:
            push_model = AutoModelForCausalLM.from_pretrained(
                FINAL_SAVE_DIR,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            push_model.push_to_hub(args.hf_repo_id, commit_message="Upload full fine-tuned OLMo model")
            tokenizer.push_to_hub(args.hf_repo_id)
            model_card.push_to_hub(args.hf_repo_id, commit_message="Add model card")
            print_main(f"Successfully pushed model + card to {args.hf_repo_id}")
            
        except Exception as e:
            print_main(f"Failed to push model: {e}")

        print("\n" + "=" * 70)
        print(f"  Final eval loss:     {final_loss}")
        print(f"  Final perplexity:    {final_ppl}")
        print(f"  Train loss:          {train_result.training_loss:.4f}")
        print(f"  Output saved to:     {args.output_dir}")
        print("=" * 70 + "\n")
        
        wandb.finish()


if __name__ == "__main__":
    main()