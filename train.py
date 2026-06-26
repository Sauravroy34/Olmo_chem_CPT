import os
import math
import random
import torch
from functools import partial
from itertools import chain

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import rdkit
import wandb

# We don't need this env variable anymore since we are initializing wandb manually
# os.environ["WANDB_PROJECT"] = "Olmo_run" 
os.environ["WANDB_LOG_MODEL"] = "false" 

DATASET_NAME = "Codemaster67/Causal_lm_chemistry_1M_rows"         
OUTPUT_DIR = "./olmo_chem_lora_cpt_LoRA_r64_alpha128"
HF_REPO_ID = "Codemaster67/Olmo-7b_1M_Smiles_lora" 
SEED = 42

NUM_EPOCHS = 1
LEARNING_RATE = 1e-5 
BATCH_SIZE = 32                             
GRADIENT_ACCUMULATION_STEPS = 1             
MAX_SEQ_LENGTH = 512                        
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 10
EVAL_STEPS = 50                             
SAVE_STEPS = 100
AUGMENT = True
NUM_AUGMENT = 4  

USE_SUBSET = True  

TRAIN_SUBSET_SIZE = 10000   
EVAL_SUBSET_SIZE = 1000

BASE_MODEL = "Codemaster67/Olmo-7b-spe" 

SMILES_START = "<|start_of_smiles|>"
SMILES_END = "<|end_of_smiles|>"
EOS = "<|endoftext|>"


def print_main(message):
    """prints only for rank 0"""
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        print(message)


# ─────────────────────────────────────────────────────────
# 1. Custom Manual Logging Function
# ─────────────────────────────────────────────────────────
def manual_wandb_log(metrics_dict, step):
    """Manually sends a dictionary of metrics to WandB on the main process."""
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        wandb.log(metrics_dict, step=step)


def setup_tokenizer(tokenizer_id="Codemaster67/Olmo-7b-spe") -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=True,
    )
    return tokenizer


def augment_smiles(smiles, num_augmentations=NUM_AUGMENT):
    smiles = smiles.replace("<|start_of_smiles|>", "").replace("<|end_of_smiles|>", "")
    mol = rdkit.Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]  

    augmented_set = set()
    attempts = 0
    max_attempts = num_augmentations * 2  

    while len(augmented_set) < num_augmentations and attempts < max_attempts:
        rand_smiles = rdkit.Chem.MolToSmiles(mol, doRandom=True)
        augmented_set.add(rand_smiles)
        attempts += 1

    del mol 
    return list(augmented_set)


def setup_model(model_id=BASE_MODEL, lora=True):
    print_main(f"Loading base model in bfloat16: {model_id}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  
    )
    if lora:
        peft_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules="all-linear", 
            lora_dropout=0.01,
            bias="none",
            use_rslora=True, 
            task_type=TaskType.CAUSAL_LM,
        )
        
        print_main("Applying LoRA adapter configuration...")
        model = get_peft_model(model, peft_config)
        
        if hasattr(model.base_model.model, 'model') and hasattr(model.base_model.model.model, 'embed_tokens'):
            model.base_model.model.model.embed_tokens.weight.requires_grad = True
        elif hasattr(model.base_model.model, 'embed_tokens'):
            model.base_model.model.embed_tokens.weight.requires_grad = True
            
        if hasattr(model.base_model.model, 'lm_head'):
            model.base_model.model.lm_head.weight.requires_grad = True

        for name, param in model.named_parameters():
            if "embed_tokens" in name or "lm_head" in name:
                param.requires_grad = True
                print_main(f"Lm_head and embed_tokens unfrozen")

    model.print_trainable_parameters()
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
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])

        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length

        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
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
            
        return metrics

    def log(self, logs: dict):
        super().log(logs)
        
        metrics_to_log = {}
        
        if "loss" in logs:
            metrics_to_log["train/loss"] = logs["loss"]
        if "learning_rate" in logs:
            metrics_to_log["train/learning_rate"] = logs["learning_rate"]
            
        if "eval_loss" in logs:
            metrics_to_log["eval/loss"] = logs["eval_loss"]
        if "eval_perplexity" in logs:
            metrics_to_log["eval/perplexity"] = logs["eval_perplexity"]

        if metrics_to_log:
            manual_wandb_log(metrics_to_log, step=self.state.global_step)


def main():
    set_seed(SEED)
    is_main_process = int(os.getenv("LOCAL_RANK", "0")) == 0

    # Initialize wandb manually on the main process
    if is_main_process:
        wandb.init(project="Olmo_run", name="olmo_lora_r64_alpha128")

    tokenizer = setup_tokenizer()
    model = setup_model()

    train_dataset, val_dataset = prepare_datasets(tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

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
        optim="adamw_8bit", 
        bf16=True,
        tf32=True,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="no",
        save_steps=SAVE_STEPS,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        report_to="none",  
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        push_to_hub=False,
        seed=SEED,
        remove_unused_columns=False,
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

    ADAPTER_DIR = os.path.join(OUTPUT_DIR, "adapter")
    
    print_main("Running final evaluation...")
    final_metrics = trainer.evaluate()

    if is_main_process:
        tokenizer.save_pretrained(ADAPTER_DIR)
        
        print_main("Attempting to push adaptors to HF Hub...")
        try:
            trainer.model.push_to_hub(HF_REPO_ID)
            tokenizer.push_to_hub(HF_REPO_ID)
            print_main(f"Successfully pushed merged model to {HF_REPO_ID}")
        except Exception as e:
            print_main(f"Failed to push merged model: {e}")

        final_ppl = final_metrics.get("eval_perplexity", "N/A")
        final_loss = final_metrics.get("eval_loss", "N/A")

        print("\n" + "=" * 70)
        print(f"  Final eval loss:     {final_loss}")
        print(f"  Final perplexity:    {final_ppl}")
        print(f"  Train loss:          {train_result.training_loss:.4f}")
        print(f"  Output saved to:     {OUTPUT_DIR}")
        print("=" * 70 + "\n")
        
        wandb.finish()


if __name__ == "__main__":
    main()