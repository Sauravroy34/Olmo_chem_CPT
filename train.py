import os
import math
import random
import torch
import logging
from functools import partial
from itertools import chain

from datasets import load_dataset , DatasetDict
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




logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


DATASET_NAME = "Codemaster67/Causal_lm_chemistry_1M_rows"         
OUTPUT_DIR = "./olmo_chem_lora_cpt_LoRA_r64_alpha128"
HF_REPO_ID = "Codemaster67/Olmo-7b_1M_Smiles_lora" # This is where the final model will be pushed
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


BASE_MODEL = "Codemaster67/Olmo-7b-spe"

SMILES_START = "<|start_of_smiles|>"
SMILES_END = "<|end_of_smiles|>"
EOS = "<|endoftext|>"

def setup_tokenizer(tokenizer_id = "Codemaster67/Olmo-7b-spe") -> AutoTokenizer:
    """Load tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=True,
    )

    return tokenizer


def augment_smiles(smiles, num_augmentations=10):
    """
    Generates random SMILES augmentations for a given SMILES string.
    """

    smiles = smiles.replace("<|start_of_smiles|>", "").replace("<|end_of_smiles|>", "")
    mol = rdkit.Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]  

    augmented_set = set()
    attempts = 0
    max_attempts = num_augmentations * 2  # Prevent infinite loops if diversity is low

    # Generate random SMILES until we have 10 unique ones
    while len(augmented_set) < num_augmentations and attempts < max_attempts:
        rand_smiles = rdkit.Chem.MolToSmiles(mol, doRandom=True)
        augmented_set.add(rand_smiles)
        attempts += 1

    del mol 
    return list(augmented_set)



def setup_model(model_id=BASE_MODEL,lora=True):
    logger.info(f"Loading base model in bfloat16: {model_id}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # Requires flash attention libraray (pip install flash-attn --no-cache-dir --no-build-isolation)
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
        
        logger.info("Applying LoRA adapter configuration...")
        model = get_peft_model(model, peft_config)
        
        if hasattr(model.base_model.model, 'model') and hasattr(model.base_model.model.model, 'embed_tokens'):
            model.base_model.model.model.embed_tokens.weight.requires_grad = True
        elif hasattr(model.base_model.model, 'embed_tokens'):
            model.base_model.model.embed_tokens.weight.requires_grad = True
            
        if hasattr(model.base_model.model, 'lm_head'):
            model.base_model.model.lm_head.weight.requires_grad = True

        # Print out confirmations to logs
        for name, param in model.named_parameters():
            if "embed_tokens" in name or "lm_head" in name:
                param.requires_grad = True
                logger.info(f" -> Enforced trainable: {name} ({param.dtype})")

    model.print_trainable_parameters()
    return model



def tokenize_function(examples: dict, tokenizer: AutoTokenizer) -> dict:
    "Tokenize text "
    texts = [t for t in examples["text"]] #For some reason olmo's tokenizer adds a default eos to end
    return tokenizer(
        texts,
        truncation=False,  
        return_attention_mask=False,
    )


def pack_sequences(tokenized_dataset, tokenizer, max_seq_length: int):
    """
    Pack (concatenate + chunk) tokenized sequences into fixed-length blocks.

    This is the standard "packing" approach for CLM:
      1. Concatenate all token sequences with EOS separators
      2. Chunk into blocks of max_seq_length
      3. Labels = shifted input_ids (handled by DataCollatorForLanguageModeling)

    Benefits:
      - No padding waste → maximum GPU utilization
      - Each batch element contains ~max_seq_length real tokens
      - Multiple short examples are packed into one sequence
    """

    def group_texts(examples):
        # Concatenate all tokenized texts
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])

        # Drop the remainder that doesn't fill a complete block
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length

        # Split into chunks of max_seq_length
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated.items()
        }

        # For CLM, labels are the same as input_ids (shifted internally by the model)
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


def prepare_datasets(tokenizer: AutoTokenizer):


    full_dataset = load_dataset(DATASET_NAME,split ="train+test") 

    shuffled_dataset = full_dataset.shuffle(seed=SEED)


    split_dataset = shuffled_dataset.train_test_split(test_size=0.1, seed=SEED)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    Source_name = "unichem" #dataset contains 2 cols , text and source 

    def format_smiles(example):
        input_smiles = example["text"] if example["source"] == Source_name else ""
        if input_smiles != "":
            augmented_list = augment_smiles(input_smiles)
            formatted_smiles = [f"{SMILES_START}{smiles}{SMILES_END}{EOS}" for smiles in augmented_list]
            formatted_smiles.append(input_smiles) #add original 
            example["text"] = "".join(formatted_smiles) 
        return example

    train_dataset = train_dataset.map(format_smiles, num_proc=4, desc="Formatting smiles")
    random.seed(SEED)


    logger.info("Tokenizing...")
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
        remove_columns=["text","source"],
        desc="Tokenizing",
    )

    # Pack sequences for efficient training
    logger.info(f"Packing sequences to length {MAX_SEQ_LENGTH}...")
    train_packed = pack_sequences(tokenized["train"], tokenizer, MAX_SEQ_LENGTH)
    val_packed = pack_sequences(tokenized["test"], tokenizer, MAX_SEQ_LENGTH)

    logger.info(f"Packed training sequences:   {len(train_packed)}")
    logger.info(f"Packed validation sequences: {len(val_packed)}")

    # Log a sample to verify formatting
    sample_ids = train_packed[0]["input_ids"][:200]
    logger.info(f"Sample packed text (first 200 tokens):\n{tokenizer.decode(sample_ids)}")

    return train_packed, val_packed


# ─────────────────────────────────────────────────────────
# 4. Custom Trainer with Perplexity Logging
# ─────────────────────────────────────────────────────────
class CLMTrainerWithPerplexity(Trainer):
    """Extends HuggingFace Trainer to log perplexity during evaluation."""

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # Compute perplexity from eval loss
        eval_loss_key = f"{metric_key_prefix}_loss"
        if eval_loss_key in metrics:
            try:
                perplexity = math.exp(metrics[eval_loss_key])
            except OverflowError:
                perplexity = float("inf")
            metrics[f"{metric_key_prefix}_perplexity"] = perplexity
            logger.info(f"Perplexity: {perplexity:.2f}")

        return metrics


# ─────────────────────────────────────────────────────────
# 5. Main Training Loop
# ─────────────────────────────────────────────────────────
def main():
    set_seed(SEED)

    # Setup tokenizer
    tokenizer = setup_tokenizer()

    model = setup_model()

    # Prepare datasets (format with template randomization → tokenize → pack)
    train_dataset, val_dataset = prepare_datasets(tokenizer)

    # Data collator for CLM (no masking, just handles labels alignment)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        # Training schedule
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        # Optimizer
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        optim="adamw_8bit", # requires bitsandbytes

        # Precision
        bf16=True,
        tf32=True,

        # Evaluation & saving
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="no",
        save_steps=SAVE_STEPS,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Logging
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        report_to="none",  

        # Memory optimization
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        # HuggingFace Hub push
        push_to_hub=False,

        # Misc
        seed=SEED,
        remove_unused_columns=False,
    )

    # Create trainer
    trainer = CLMTrainerWithPerplexity(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # ── Run initial evaluation (baseline) ──
    logger.info("Running baseline evaluation before training...")
    baseline_metrics = trainer.evaluate()
    logger.info(f"Baseline metrics: {baseline_metrics}")

    # ── Train ──
    train_result = trainer.train()

    # ── Save final model locally ──


    ADAPTER_DIR = os.path.join(OUTPUT_DIR, "adapter")

    logger.info(f"Saving LoRA adapter to {ADAPTER_DIR}")

    trainer.save_model(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    # ==========================================================
    # Merge LoRA into base model
    # ==========================================================

    logger.info("Reloading base model for merge...")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    logger.info("Loading adapter and merging weights...")

    merged_model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_DIR,
    )

    merged_model = merged_model.merge_and_unload()

    # ==========================================================
    # Save merged model
    # ==========================================================

    MERGED_DIR = os.path.join(OUTPUT_DIR, "merged")

    logger.info(f"Saving merged model to {MERGED_DIR}")

    merged_model.save_pretrained(
        MERGED_DIR,
    )

    tokenizer.save_pretrained(MERGED_DIR)

    # ==========================================================
    # Push merged model to HF Hub
    # ==========================================================

    logger.info(f"Pushing merged model to {HF_REPO_ID}")

    try:

        merged_model.push_to_hub(
            HF_REPO_ID,
            commit_message=(
                f"OLMo-7B domain adaptation on {DATASET_NAME}\n"
                f"LoRA merged into base model\n"
                f"Epochs={NUM_EPOCHS}, LR={LEARNING_RATE}"
            ),
        )

        tokenizer.push_to_hub(HF_REPO_ID)

        logger.info(
            f"Successfully pushed merged model to {HF_REPO_ID}"
        )

    except Exception as e:
        logger.error(f"Failed to push merged model: {e}")

        logger.info(
            f"Merged model saved locally at {MERGED_DIR}"
        )

    # ── Final evaluation ──
    logger.info("Running final evaluation...")
    final_metrics = trainer.evaluate()


    baseline_ppl = baseline_metrics.get("eval_perplexity", "N/A")
    final_ppl = final_metrics.get("eval_perplexity", "N/A")
    baseline_loss = baseline_metrics.get("eval_loss", "N/A")
    final_loss = final_metrics.get("eval_loss", "N/A")

    print(f"  Baseline eval loss:  {baseline_loss}")
    print(f"  Baseline perplexity: {baseline_ppl}")
    print(f"  Final eval loss:     {final_loss}")
    print(f"  Final perplexity:    {final_ppl}")
    print(f"  Train loss:          {train_result.training_loss:.4f}")
    print(f"  Output saved to:     {OUTPUT_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()


