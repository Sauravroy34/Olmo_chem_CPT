"""
Chemistry Causal LM Dataset Preparation
Prepares and pushes to Codemaster67/Causal_lm_chemistry

Sources:
  - 200k SMILES from bisectgroup/UniChem
  - 30k title+abstract from BASF-AI/ChemRxiv-Papers
  - 10k SMILES-description pairs from liupf/ChEBI-20-MM
"""

import random
import logging
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# ── special tokens ────────────────────────────────────────────────────────────
SOM = "<|start_of_smiles|>"
EOM = "<|end_of_smiles|>"

# ── templates for ChEBI-20 SMILES-description pairs ──────────────────────────


TEMPLATES = [
    # 1. SMILES first (classic pre-training / raw association)
    "{som}{smiles}{eom}\n{description}",

    # 2. Middle wrapping (MolXPT-inspired inline context)
    "The chemical structure {som}{smiles}{eom} can be described as follows: {description}",

    # 3. Text first, SMILES last (generation style)
    "{description} This compound is represented by the structure {som}{smiles}{eom}.",

    # 4. SMILES first with connector (property prediction style)
    "Regarding {som}{smiles}{eom}, {description}",

    # 5. Conversational / integrative phrasing
    "Based on its properties, {description} Structurally, we map this to {som}{smiles}{eom}.",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  UniChem  →  200 k SMILES wrapped in special tokens
# ─────────────────────────────────────────────────────────────────────────────
def prepare_unichem(n: int = 300_000) -> Dataset:

    # The dataset is large; stream to avoid OOM
    ds = load_dataset(
        "bisectgroup/UniChem",
        split="train",
        streaming=True,
    )

    records = []
    seen = set()

    for row in ds:
        # Column may be "smiles" or "standardised_smiles" depending on the subset
        smiles = row.get("smiles") or row.get("standardised_smiles") or row.get("SMILES")
        if not smiles:
            continue
        smiles = smiles.strip()
        if not smiles or smiles in seen:
            continue
        seen.add(smiles)
        records.append({"text": f"{SOM}{smiles}{EOM}", "source": "unichem"})
        if len(records) >= n:
            break

    return Dataset.from_list(records)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ChemRxiv  →  30 k title + abstract
# ─────────────────────────────────────────────────────────────────────────────
def prepare_chemrxiv(n: int = 30_000) -> Dataset:

    ds = load_dataset(
        "BASF-AI/ChemRxiv-Papers",
        split="train",
        streaming=True,
    )

    records = []
    for row in ds:
        title    = (row.get("title")    or "").strip()
        abstract = (row.get("abstract") or "").strip()

        if not abstract:          # skip rows with no abstract
            continue

        text = f"{title}\n\n{abstract}" if title else abstract
        records.append({"text": text, "source": "chemrxiv"})

        if len(records) >= n:
            break

    return Dataset.from_list(records)



def prepare_chebi20(n: int = 30_000) -> Dataset:
    log.info("Loading ChEBI-20-MM …")

    # ── ChEBI-20-MM: columns → SMILES, description ───────────────────────────
    raw_1 = load_dataset("liupf/ChEBI-20-MM")
    splits_1 = [raw_1[s] for s in raw_1 if len(raw_1[s]) > 0]

    # ── moltextnet: columns → canonical_smiles, description ──────────────────
    raw_2 = load_dataset("liuganghuggingface/moltextnet")
    splits_2 = [
        raw_2[s].rename_column("canonical_smiles", "SMILES")
        for s in raw_2 if len(raw_2[s]) > 0
    ]

    # ── pool & shuffle ────────────────────────────────────────────────────────
    pooled = concatenate_datasets(splits_1 + splits_2).shuffle(seed=42)

    records = []
    for row in pooled:
        smiles      = (row.get("SMILES") or "").strip()
        description = (row.get("description") or "").strip()

        if not smiles or not description:
            continue

        template = random.choice(TEMPLATES)
        text = template.format(
            som=SOM, eom=EOM,
            smiles=smiles,
            description=description,
        )
        records.append({"text": text, "source": "chebi20"})

        if len(records) >= n:
            break

    return Dataset.from_list(records)


def prepare_pubmed(n: int = 30_000) -> Dataset:
    log.info("Loading PubMed title+abstracts (streaming) …")

    ds = load_dataset(
        "casinca/PUBMED_title_abstracts_2019_baseline",
        split="train"
    )

    records = []
    for row in ds:
        Text    = (row.get("text") or "").strip()
        if not Text:
            continue
        records.append({"text": Text, "source": "pubmed"})

        if len(records) >= n:
            break

    return Dataset.from_list(records)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Combine, shuffle, split, push
# ─────────────────────────────────────────────────────────────────────────────
def build_and_push(hub_repo: str = "Codemaster67/Causal_lm_chemistry"):
    random.seed(42)

    unichem_ds  = prepare_unichem(1_000_000)
    chemrxiv_ds = prepare_chemrxiv(30_000)
    chebi_ds    = prepare_chebi20(500_000)
    pubmed_ds   = prepare_pubmed(20_000)

    log.info("Combining datasets …")
    combined = concatenate_datasets([unichem_ds, chemrxiv_ds, chebi_ds, pubmed_ds])
    combined = combined.shuffle(seed=42)


    # 95 / 5  train-test split
    split = combined.train_test_split(test_size=0.05, seed=42)
    dataset_dict = DatasetDict({
        "train": split["train"],
        "test":  split["test"],
    })


    # ── push ──────────────────────────────────────────────────────────────────
    dataset_dict.push_to_hub(
        hub_repo,
        private=False,          # set True if you want it private
        commit_message="Added dataset",
    )

    return dataset_dict


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ds = build_and_push()

    # quick sanity-check: print one example from each source
    for source in ("unichem", "chemrxiv", "chebi20", "pubmed"):
        sample = ds["train"].filter(lambda x: x["source"] == source).select(range(1))[0]
        print(f"\n{'─'*60}")
        print(f"[{source}]  {sample['text'][:300]}")
