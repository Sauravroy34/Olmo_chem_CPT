#!/usr/bin/env python3
"""
Train SPE Tokenizer on ZINC20 + ChEMBL SMILES, extend OLMo-7B-hf tokenizer, push to Hub.

Usage (PyTorch Lightning Studio):
    pip install SmilesPE rdkit-pypi transformers huggingface_hub datasets tqdm fastprogress
    python train_spe_tokenizer.py
"""

import os
import random
import codecs
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from datasets import load_dataset
from rdkit import Chem
from rdkit import RDLogger

# Suppress RDKit warnings for clean output
RDLogger.logger().setLevel(RDLogger.ERROR)

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════
ZINC_TARGET   = 2_000_000
CHEMBL_TARGET = 2_000_000
SPE_VOCAB_SIZE   = 300
SPE_MIN_FREQ     = 4000
SPE_AUGMENTATION = 0       # no augmentation

DATA_DIR     = Path("./spe_data")
OUTPUT_DIR   = Path("./spe_output")
SMILES_CORPUS = DATA_DIR / "combined_canonical_smiles.txt"
SPE_VOC_FILE  = OUTPUT_DIR / "spe_codes.txt"

HF_REPO    = "Codemaster67/Olmo_chemical_aware_tokenizer"
BASE_MODEL = "allenai/OLMo-7B-hf"

# HuggingFace dataset sources (much faster & more reliable than raw servers)
ZINC_HF_DATASET   = "haydn-jones/ZINC20"     # ~1.5B SMILES, column: "smiles"
CHEMBL_HF_DATASET = "antoinebcx/smiles-molecules-chembl"  # ~1.94M SMILES, column: "smiles"


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1a: Download ZINC20 SMILES (2M) from HuggingFace
# ═══════════════════════════════════════════════════════════════════════════════
def download_zinc20_smiles(target: int = ZINC_TARGET) -> list[str]:
    """
    Stream ~2M SMILES from the haydn-jones/ZINC20 dataset on HuggingFace.
    
    The dataset contains ~1.5B SMILES in parquet format. We use streaming
    mode to avoid downloading the full ~174GB dataset — only the first
    `target` rows are fetched.
    """
    print("=" * 70)
    print(f"STEP 1a: Streaming {target:,} SMILES from HuggingFace ({ZINC_HF_DATASET})")
    print("=" * 70)

    cache_file = DATA_DIR / "zinc20_raw_smiles.txt"
    if cache_file.exists():
        print(f"  → Found cached file: {cache_file}")
        with open(cache_file) as f:
            smiles = [line.strip() for line in f if line.strip()]
        if len(smiles) >= target:
            print(f"  → Loaded {len(smiles):,} cached ZINC SMILES")
            return smiles[:target]
        print(f"  → Cache has only {len(smiles):,}, need more...")

    smiles = []
    
    print(f"  → Loading dataset in streaming mode...")
    ds = load_dataset(ZINC_HF_DATASET, split="train", streaming=True)
    
    for example in tqdm(ds, desc="  Streaming ZINC20", total=target):
        smi = example.get("smiles", "")
        if smi and isinstance(smi, str) and len(smi) > 1:
            smiles.append(smi.strip())
        if len(smiles) >= target:
            break
    
    print(f"  → Streamed {len(smiles):,} raw ZINC SMILES")
    
    # Cache to disk
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        for smi in smiles:
            f.write(smi + "\n")
    
    return smiles[:target]


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1b: Download ChEMBL SMILES (2M) from HuggingFace
# ═══════════════════════════════════════════════════════════════════════════════
def download_chembl_smiles(target: int = CHEMBL_TARGET) -> list[str]:
    """
    Download SMILES from the antoinebcx/smiles-molecules-chembl dataset on
    HuggingFace. The dataset has ~1.94M total SMILES across train/val/test
    splits, all with a single 'smiles' column.
    
    Since the full dataset is only ~113MB, we download all splits and
    concatenate them to get close to our 2M target.
    """
    print("=" * 70)
    print(f"STEP 1b: Downloading {target:,} SMILES from HuggingFace ({CHEMBL_HF_DATASET})")
    print("=" * 70)
    
    cache_file = DATA_DIR / "chembl_raw_smiles.txt"
    if cache_file.exists():
        print(f"  → Found cached file: {cache_file}")
        with open(cache_file) as f:
            smiles = [line.strip() for line in f if line.strip()]
        if len(smiles) >= min(target, 1_900_000):  # ChEMBL has ~1.94M total
            print(f"  → Loaded {len(smiles):,} cached ChEMBL SMILES")
            return smiles[:target]
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    smiles = []
    
    # Load all splits to maximize coverage (~1.94M total)
    for split_name in ["train", "validation", "test"]:
        print(f"  → Loading {split_name} split...")
        ds = load_dataset(CHEMBL_HF_DATASET, split=split_name)
        for example in tqdm(ds, desc=f"  Reading {split_name}"):
            smi = example.get("smiles", "")
            if smi and isinstance(smi, str) and len(smi) > 1:
                smiles.append(smi.strip())
        print(f"  → {split_name}: {len(smiles):,} SMILES so far")
        if len(smiles) >= target:
            break
    
    print(f"  → Downloaded {len(smiles):,} ChEMBL SMILES")
    
    # Cache
    with open(cache_file, "w") as f:
        for smi in smiles:
            f.write(smi + "\n")
    
    return smiles[:target]


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Filter invalid SMILES & Canonicalize
# ═══════════════════════════════════════════════════════════════════════════════
def filter_and_canonicalize(smiles_list: list[str], source_name: str = "") -> list[str]:
    """
    Filter out invalid SMILES and canonicalize valid ones using RDKit.
    Also removes duplicates.
    """
    print(f"\n{'=' * 70}")
    print(f"STEP 2: Filtering & Canonicalizing {source_name} ({len(smiles_list):,} input)")
    print(f"{'=' * 70}")
    
    canonical = set()
    invalid_count = 0
    
    for smi in tqdm(smiles_list, desc=f"  Validating {source_name}"):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                can_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                if can_smi and len(can_smi) > 0:
                    canonical.add(can_smi)
            else:
                invalid_count += 1
        except Exception:
            invalid_count += 1
    
    result = list(canonical)
    print(f"  → Valid & unique: {len(result):,}")
    print(f"  → Invalid/filtered: {invalid_count:,}")
    print(f"  → Duplicates removed: {len(smiles_list) - len(result) - invalid_count:,}")
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Train SPE Tokenizer
# ═══════════════════════════════════════════════════════════════════════════════
def train_spe_tokenizer(corpus_file: Path, output_file: Path):
    """
    Train SMILES Pair Encoding tokenizer using the SmilesPE library.
    
    Args:
        corpus_file: Path to file with one canonical SMILES per line
        output_file: Path to save the learned SPE codes
    """
    print(f"\n{'=' * 70}")
    print(f"STEP 3: Training SPE Tokenizer")
    print(f"  Vocab size: {SPE_VOCAB_SIZE}")
    print(f"  Min frequency: {SPE_MIN_FREQ}")
    print(f"  Augmentation: {SPE_AUGMENTATION} (disabled)")
    print(f"  Corpus: {corpus_file}")
    print(f"{'=' * 70}")
    
    from SmilesPE.learner import learn_SPE
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # learn_SPE expects infile as an iterable of SMILES strings (no newlines)
    # and outfile as a writable file handle
    with open(corpus_file, "r") as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    
    print(f"  → Loaded {len(smiles_list):,} SMILES from corpus")


    
    with open(output_file, "w") as outfile:
        learn_SPE(
            infile=smiles_list,
            outfile=outfile,
            num_symbols=SPE_VOCAB_SIZE,
            min_frequency=SPE_MIN_FREQ,
            augmentation=SPE_AUGMENTATION,
            verbose=False,
        )
    


    
    # Verify training output
    with open(output_file) as f:
        num_codes = sum(1 for _ in f)
    print(f"  → Trained {num_codes} SPE merge operations")
    print(f"  → Saved to: {output_file}")
    
    return output_file


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Extract SPE tokens & Extend OLMo Tokenizer
# ═══════════════════════════════════════════════════════════════════════════════
def extract_spe_tokens(spe_codes_file: Path, corpus_file: Path) -> list[str]:
    """
    Extract all unique SPE subword tokens by tokenizing the corpus
    with the trained SPE model, then collecting all unique tokens
    that are NOT already single characters (atom-level tokens).
    """
    print(f"\n  Extracting SPE tokens from trained model...")
    
    from SmilesPE.tokenizer import SPE_Tokenizer
    
    spe_voc = codecs.open(spe_codes_file)
    spe_tokenizer = SPE_Tokenizer(spe_voc)
    
    # Collect all unique SPE tokens from the corpus
    all_tokens = set()
    
    with open(corpus_file) as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    
    # Sample a subset for token extraction (full corpus takes too long)
    sample_size = min(500_000, len(smiles_list))
    sample = random.sample(smiles_list, sample_size)
    
    for smi in tqdm(sample, desc="  Extracting SPE tokens"):
        try:
            tokenized = spe_tokenizer.tokenize(smi)
            tokens = tokenized.split(" ")
            for tok in tokens:
                if len(tok) > 1:  # skip single-char atom tokens
                    all_tokens.add(tok)
        except Exception:
            continue
    
    spe_voc.close()
    
    # Also add the merge-pair combined tokens directly from the codes file
    with open(spe_codes_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                merged = parts[0] + parts[1]
                all_tokens.add(merged)
    
    token_list = sorted(all_tokens)
    print(f"  → Extracted {len(token_list)} unique SPE subword tokens")
    return token_list


def extend_olmo_tokenizer(spe_tokens: list[str]):
    """
    Extend OLMo-7B-hf tokenizer with:
    1. SPE subword tokens (as regular tokens)
    2. <|start_of_smiles|> and <|end_of_smiles|> (as special tokens)
    
    Then push the extended tokenizer to the Hugging Face Hub.
    """
    print(f"\n{'=' * 70}")
    print(f"STEP 4: Extending OLMo-7B-hf Tokenizer")
    print(f"{'=' * 70}")
    
    from transformers import AutoTokenizer
    from huggingface_hub import login
    
    # Login to Hugging Face (will prompt for token if not cached)
    print("  → Logging into Hugging Face Hub...")
    login()
    
    # Load base tokenizer
    print(f"  → Loading base tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    original_vocab_size = len(tokenizer)
    print(f"  → Original vocab size: {original_vocab_size:,}")
    
    # Filter out tokens that already exist in the vocabulary
    existing_vocab = tokenizer.get_vocab()
    new_spe_tokens = [tok for tok in spe_tokens if tok not in existing_vocab]
    print(f"  → SPE tokens to add: {len(new_spe_tokens):,} "
          f"(filtered {len(spe_tokens) - len(new_spe_tokens)} existing)")
    
    # Add SPE tokens as regular tokens
    if new_spe_tokens:
        num_added = tokenizer.add_tokens(new_spe_tokens)
        print(f"  → Added {num_added} SPE tokens")
    
    # Add SMILES delimiter special tokens
    special_tokens = {
        "additional_special_tokens": ["<|start_of_smiles|>", "<|end_of_smiles|>"]
    }
    num_special = tokenizer.add_special_tokens(special_tokens)
    print(f"  → Added {num_special} special tokens: <|start_of_smiles|>, <|end_of_smiles|>")
    
    final_vocab_size = len(tokenizer)
    print(f"  → Final vocab size: {final_vocab_size:,} (+{final_vocab_size - original_vocab_size})")
    
    # Verify the new tokens work
    print("\n  → Verification:")
    test_smi = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    wrapped = f"<|start_of_smiles|>{test_smi}<|end_of_smiles|>"
    encoded = tokenizer.encode(wrapped)
    decoded = tokenizer.decode(encoded)
    print(f"    Input:   {wrapped}")
    print(f"    Tokens:  {encoded[:20]}{'...' if len(encoded) > 20 else ''}")
    print(f"    Decoded: {decoded}")
    
    start_id = tokenizer.convert_tokens_to_ids("<|start_of_smiles|>")
    end_id   = tokenizer.convert_tokens_to_ids("<|end_of_smiles|>")
    print(f"    <|start_of_smiles|> → ID {start_id}")
    print(f"    <|end_of_smiles|>   → ID {end_id}")
    
    # Save locally first
    local_save_dir = OUTPUT_DIR / "extended_tokenizer"
    tokenizer.save_pretrained(local_save_dir)
    print(f"\n  → Saved locally to: {local_save_dir}")
    
    # Push to Hub
    print(f"\n  → Pushing to Hub: {HF_REPO}")
    tokenizer.push_to_hub(
        HF_REPO,
        commit_message=(
            f"Add SPE tokenizer ({len(new_spe_tokens)} SMILES subword tokens) "
            f"+ <|start_of_smiles|>/<|end_of_smiles|> special tokens. "
            f"Trained on 2M ZINC20 + 2M ChEMBL canonical SMILES. "
            f"SPE vocab_size={SPE_VOCAB_SIZE}, min_freq={SPE_MIN_FREQ}."
        ),
    )
    print(f"  ✓ Successfully pushed to https://huggingface.co/{HF_REPO}")
    
    return tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "█" * 70)
    print("  SPE TOKENIZER TRAINING PIPELINE")
    print("  ZINC20 (2M) + ChEMBL (2M) → SPE → Extend OLMo-7B-hf → Push")
    print("█" * 70 + "\n")
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ── Download ──────────────────────────────────────────────────────────
    zinc_smiles   = download_zinc20_smiles(ZINC_TARGET)
    chembl_smiles = download_chembl_smiles(CHEMBL_TARGET)
    
    # ── Filter & Canonicalize ─────────────────────────────────────────────
    zinc_canonical   = filter_and_canonicalize(zinc_smiles, "ZINC20")
    chembl_canonical = filter_and_canonicalize(chembl_smiles, "ChEMBL")
    
    # ── Combine & Deduplicate ─────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("Combining & deduplicating ZINC20 + ChEMBL")
    print(f"{'=' * 70}")
    
    combined = list(set(zinc_canonical + chembl_canonical))
    random.shuffle(combined)
    
    print(f"  → ZINC20 canonical:  {len(zinc_canonical):,}")
    print(f"  → ChEMBL canonical:  {len(chembl_canonical):,}")
    print(f"  → Combined (unique): {len(combined):,}")
    
    # Write corpus file (one SMILES per line)
    with open(SMILES_CORPUS, "w") as f:
        for smi in combined:
            f.write(smi + "\n")
    print(f"  → Corpus saved to: {SMILES_CORPUS}")
    
    # ── Train SPE ─────────────────────────────────────────────────────────
    train_spe_tokenizer(SMILES_CORPUS, SPE_VOC_FILE)
    
    # ── Extract SPE tokens ────────────────────────────────────────────────
    spe_tokens = extract_spe_tokens(SPE_VOC_FILE, SMILES_CORPUS)
    
    # ── Extend OLMo tokenizer & push ─────────────────────────────────────
    tokenizer = extend_olmo_tokenizer(spe_tokens)
    
    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'█' * 70}")
    print("  PIPELINE COMPLETE!")
    print(f"{'█' * 70}")
    print(f"  Corpus:          {SMILES_CORPUS} ({len(combined):,} SMILES)")
    print(f"  SPE codes:       {SPE_VOC_FILE}")
    print(f"  Tokenizer:       {OUTPUT_DIR / 'extended_tokenizer'}")
    print(f"  Hub:             https://huggingface.co/{HF_REPO}")
    print(f"  Final vocab:     {len(tokenizer):,}")
    print()


if __name__ == "__main__":
    main()
