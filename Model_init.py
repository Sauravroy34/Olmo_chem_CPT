import os
import math
import random
import torch
import logging
from functools import partial
from itertools import chain

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
SMILES_START = "<|start_of_smiles|>"
SMILES_END = "<|end_of_smiles|>"

def setup_model(model_id="allenai/OLMo-7B-hf", new_tokenizer_id="Codemaster67/OLMO_Smiles_aware_tokenizer", token_file="spe_codes.txt") -> AutoModelForCausalLM:

    print("Loading tokenizers and model...")
    base_tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True,)
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                torch_dtype=torch.bfloat16,
                                                trust_remote_code=True,
                                                )

    new_tokenizer = AutoTokenizer.from_pretrained(new_tokenizer_id,trust_remote_code=True,)

    with open(token_file, "r") as f:
        new_smiles_tokens = [line.strip() for line in f if line.strip()]

    new_smiles_tokens = new_smiles_tokens + [SMILES_START, SMILES_END]

    print(len(new_smiles_tokens))
    model.resize_token_embeddings(len(new_tokenizer))

    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data  # ← LM head weights

    are_tied = input_embeddings.data_ptr() == output_embeddings.data_ptr()
    print(f"Tied: {are_tied}")

    # Or check config
    print(f"Config says tied: {model.config.tie_word_embeddings}")
    print("Starting Mean Initialization...")
    with torch.no_grad():
        for token in new_smiles_tokens:
            token = "".join(token.split())

            print(token) if new_smiles_tokens.index(token) % 100 == 0 else None
            
            sub_token_ids = base_tokenizer.encode(token, add_special_tokens=False)

            if len(sub_token_ids) > 0:
                # --- Input embeddings ---
                sub_token_vectors = input_embeddings[sub_token_ids]
                mean_vector = torch.mean(sub_token_vectors, dim=0)

                new_target_id = new_tokenizer.convert_tokens_to_ids(token) 
                input_embeddings[new_target_id] = mean_vector

                # --- Output embeddings (LM head) ---
                lm_sub_vectors = output_embeddings[sub_token_ids]
                lm_mean_vector = torch.mean(lm_sub_vectors, dim=0)
                output_embeddings[new_target_id] = lm_mean_vector  

    return model , new_tokenizer 



model , tokenizer = setup_model()   

repo_id = "Codemaster67/Olmo-7b-spe-200ksmiles-10-smiles_text-30kchempapers"
print("Pushing to Hub...")

# model.push_to_hub(
#     repo_id,
# )

# tokenizer.push_to_hub(repo_id)

# print("Done!")


