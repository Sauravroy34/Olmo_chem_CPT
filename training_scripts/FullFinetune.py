"""
    Multi GPU:   torchrun --nproc_per_node=<NUM_GPUS> FullFinetune.py
"""
import os
import math
import random
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


login(token="hf token")

wandb.login(key="wandb key")
os.environ["WANDB_LOG_MODEL"] = "false" 

DATASET_NAME = "Codemaster67/Causal_lm_chemistry_1M_rows"         
OUTPUT_DIR = "./olmo_chem_full_cpt_5e-6_lr"
HF_REPO_ID = "Codemaster67/Olmo-7b_1M_Smiles_fullfinetune" 
SEED = 42

NUM_EPOCHS = 1
LEARNING_RATE = 5e-6 #conservative learning rate  
BATCH_SIZE = 32                             
GRADIENT_ACCUMULATION_STEPS = 1             
MAX_SEQ_LENGTH = 512                        
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 10
EVAL_STEPS = 50                             
SAVE_STEPS = 100

AUGMENT = False
NUM_AUGMENT = 4  

USE_SUBSET = True # if set to false then whole training dataset is used  
TRAIN_SUBSET_SIZE = 10000   
EVAL_SUBSET_SIZE = 1000


# note if augment is true is number of training samples = TRAIN_SUBSET_SIZE * NUM_AUGMENT * 0.65 (65 Percent of dataset is smiles strings)
BASE_MODEL = "Codemaster67/Olmo-7b-spe" # Note this model has the new tokenizer 

SMILES_START = "<|start_of_smiles|>"
SMILES_END = "<|end_of_smiles|>"
EOS = "<|endoftext|>"


def print_main(message):
    """prints only for rank 0"""
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        print(message)


def wandb_log(metrics_dict, step):
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        wandb.log(metrics_dict, step=step)


def setup_tokenizer(tokenizer_id=BASE_MODEL) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=True,
    )


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def augment_smiles(smiles, num_augmentations=NUM_AUGMENT):
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


def setup_model(model_id=BASE_MODEL):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
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


def prepare_datasets(tokenizer, augment=AUGMENT):
    full_dataset = load_dataset(DATASET_NAME, split="train+test") 
    shuffled_dataset = full_dataset.shuffle(seed=SEED)

    split_dataset = shuffled_dataset.train_test_split(test_size=0.1, seed=SEED)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    if USE_SUBSET:
        print_main(f"Using a subset of the training data: {TRAIN_SUBSET_SIZE} examples")
        train_dataset = train_dataset.select(range(TRAIN_SUBSET_SIZE))
        
        print_main(f"Using a subset of the validation data: {EVAL_SUBSET_SIZE} examples")
        val_dataset = val_dataset.select(range(EVAL_SUBSET_SIZE))

    Source_name = "unichem"  
    if augment:
        def format_smiles(example):
            input_smiles = example["text"] if example["source"] == Source_name else ""
            if input_smiles != "":
                augmented_list = augment_smiles(input_smiles)
                formatted_smiles = [f"{SMILES_START}{smiles}{SMILES_END}{EOS}" for smiles in augmented_list]
                formatted_smiles.append(input_smiles)  
                example["text"] = "".join(formatted_smiles) 
            return example

        train_dataset = train_dataset.map(format_smiles, num_proc=4, desc="Formatting smiles")
        
    random.seed(SEED)

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

    train_packed = pack_sequences(tokenized["train"], tokenizer, MAX_SEQ_LENGTH)
    val_packed = pack_sequences(tokenized["test"], tokenizer, MAX_SEQ_LENGTH)

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
    set_seed(SEED)
    is_main_process = int(os.getenv("LOCAL_RANK", "0")) == 0

    if is_main_process:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wandb_project = f"OLMo_Full_Fine_tune_{run_timestamp}"
        wandb_run_name = f"Full_Fine_tune_lr{LEARNING_RATE}_samples{TRAIN_SUBSET_SIZE}_{run_timestamp}"
        wandb.init(project=wandb_project, name=wandb_run_name)

    tokenizer = setup_tokenizer()
    model = setup_model()

    train_dataset, val_dataset = prepare_datasets(tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    transformer_layer_cls_name = type(model.model.layers[0]).__name__

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        optim="adamw_torch",  
        bf16=True,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_total_limit=2, #only save last 2 models 
        save_steps=SAVE_STEPS,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        report_to="wandb", 
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        push_to_hub=False,
        seed=SEED,
        remove_unused_columns=False,
        #More about fsdp https://github.com/huggingface/transformers/blob/e0e7504bca2bfd1b85bb0eedb148f7b250226f06/src/transformers/training_args.py#L650
        fsdp=True,
        fsdp_config={
            "transformer_layer_cls_to_wrap": transformer_layer_cls_name,
            "activation_checkpointing": True,
            "sync_module_states": True,
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
    FINAL_SAVE_DIR = os.path.join(OUTPUT_DIR, "final_model_push")
    trainer.save_model(FINAL_SAVE_DIR)

    if is_main_process: 
        print_main("Generating Model Card...")
    
        tokenizer.save_pretrained(OUTPUT_DIR)

        final_ppl = final_metrics.get("eval_perplexity", "N/A")
        final_loss = final_metrics.get("eval_loss", "N/A")

        # ── Model Card ──────────────────────────────────────────────
        card_data = ModelCardData(
            language="en",
            license="apache-2.0",
            library_name="transformers",
            tags=["chemistry", "smiles", "olmo", "causal-lm", "full-finetune", "fsdp"],
            datasets=[DATASET_NAME],
            base_model=BASE_MODEL,
        )
        card_content = f"""---
{card_data.to_yaml()}
---

# OLMo-7B Full Fine-Tune — Chemistry SMILES CPT

## Model Description

This model is a **full-parameter fine-tuned** version of
[{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}) trained on chemistry
SMILES strings from the
[{DATASET_NAME}](https://huggingface.co/datasets/{DATASET_NAME}) dataset.

The base model's tokenizer was pre-extended with ~300 SPE (SMILES Pair
Encoding) chemistry tokens plus `<|start_of_smiles|>` / `<|end_of_smiles|>`
special tokens, and its embedding & LM-head layers were resized with
mean-initialised vectors for the new tokens.

## Training Details

| Parameter | Value |
|---|---|
| **Method** | Full Fine-Tune (all weights updated) |
| **Parallelism** | FSDP (Fully Sharded Data Parallel) |
| **Epochs** | {NUM_EPOCHS} |
| **Learning Rate** | {LEARNING_RATE} |
| **Batch Size (per device)** | {BATCH_SIZE} |
| **Gradient Accumulation** | {GRADIENT_ACCUMULATION_STEPS} |
| **Max Sequence Length** | {MAX_SEQ_LENGTH} |
| **Warmup Ratio** | {WARMUP_RATIO} |
| **Weight Decay** | {WEIGHT_DECAY} |
| **Scheduler** | Cosine |
| **Precision** | bf16 |
| **Augmentation** | {"ON (×" + str(NUM_AUGMENT) + " per unichem SMILES)" if AUGMENT else "OFF"} |
| **Training Samples** | {TRAIN_SUBSET_SIZE if USE_SUBSET else "Full dataset"} |
| **Eval Samples** | {EVAL_SUBSET_SIZE if USE_SUBSET else "Full dataset (10%)"} |

## Evaluation Results

| Metric | Value |
|---|---|
| **Final Eval Loss** | {final_loss} |
| **Final Eval Perplexity** | {final_ppl} |
| **Training Loss** | {train_result.training_loss:.4f} |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{HF_REPO_ID}", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("{HF_REPO_ID}", trust_remote_code=True)

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
- Augmentation was {"enabled" if AUGMENT else "disabled"} for this run.
"""
        model_card = ModelCard(card_content)
        # ────────────────────────────────────────────────────────────

        try:
            push_model = AutoModelForCausalLM.from_pretrained(
                FINAL_SAVE_DIR,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            push_model.push_to_hub(HF_REPO_ID, commit_message="Upload full fine-tuned OLMo model")
            tokenizer.push_to_hub(HF_REPO_ID)
            model_card.push_to_hub(HF_REPO_ID, commit_message="Add model card")
            print_main(f"Successfully pushed model + card to {HF_REPO_ID}")
            
        except Exception as e:
            print_main(f"Failed to push model: {e}")

        print("\n" + "=" * 70)
        print(f"  Final eval loss:     {final_loss}")
        print(f"  Final perplexity:    {final_ppl}")
        print(f"  Train loss:          {train_result.training_loss:.4f}")
        print(f"  Output saved to:     {OUTPUT_DIR}")
        print("=" * 70 + "\n")
        
        wandb.finish()


if __name__ == "__main__":
    main()