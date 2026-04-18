
import os
import json
import time
import argparse
import requests
import pandas as pd
from tqdm import tqdm

UNIPROT_API = "https://rest.uniprot.org/uniprotkb/{uid}.fasta"
MAX_RETRIES  = 3
RETRY_DELAY  = 2   
BATCH_DELAY  = 0.3 


def fetch_sequence(uid: str) -> str | None:
    url = UNIPROT_API.format(uid=uid)
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                lines = resp.text.strip().split("\n")
                seq = "".join(line for line in lines if not line.startswith(">"))
                return seq if seq else None
            elif resp.status_code == 404:
                return None 
        except requests.exceptions.RequestException:
            pass
        time.sleep(RETRY_DELAY)
    return None


def load_pairs_csv(path: str) -> pd.DataFrame:

    df = pd.read_csv(path, header=None)
    if df.shape[1] == 2:
        df.columns = ["protein_a", "protein_b"]
        df["label"] = 1  
    elif df.shape[1] == 3:
        df.columns = ["protein_a", "protein_b", "label"]
    else:
        raise ValueError(f"Unexpected CSV shape: {df.shape} in {path}")
    return df


def save_cache_atomic(seq_map: dict, cache_path: str):
    temp_path = cache_path + ".tmp"
    with open(temp_path, "w") as f:
        json.dump(seq_map, f)
    os.replace(temp_path, cache_path)


def build_sequence_lookup(protein_ids: list[str], cache_path: str | None = None) -> dict[str, str]:
    seq_map: dict[str, str] = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            seq_map = json.load(f)
        print(f"Loaded {len(seq_map)} cached sequences from {cache_path}")

    unique_ids = list(set(protein_ids))
    to_fetch   = [uid for uid in unique_ids if uid not in seq_map]

    if to_fetch:
              f"({len(unique_ids) - len(to_fetch)} already cached)...")
        for uid in tqdm(to_fetch, desc="UniProt API"):
            seq = fetch_sequence(uid)
            if seq:
                seq_map[uid] = seq
                if cache_path:
                    save_cache_atomic(seq_map, cache_path)
            else:
                print(f" Could not fetch: {uid}")
            time.sleep(BATCH_DELAY)

    fetched = sum(1 for uid in unique_ids if uid in seq_map)
    print(f"{fetched}/{len(unique_ids)} sequences available.")
    return seq_map


def pairs_to_seq_df(pairs_df: pd.DataFrame, seq_map: dict) -> pd.DataFrame:
    rows = []
    skipped = 0
    for _, row in pairs_df.iterrows():
        a, b, lbl = row["protein_a"], row["protein_b"], row["label"]
        seq_a = seq_map.get(a)
        seq_b = seq_map.get(b)
        if seq_a and seq_b:
            rows.append({"seq_a": seq_a, "seq_b": seq_b, "label": int(lbl)})
        else:
            skipped += 1
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Fetch protein sequences and build ESM-2 dataset CSVs.")
    parser.add_argument("--data_dir", type=str, default=".",
                        help="Root directory of the DPPI project (where AlphaYeastResults.csv lives).")
    parser.add_argument("--train_csv", type=str, default="myTrain/train_pairs.csv",
                        help="Path to the DPPI train pairs CSV (relative to data_dir).")
    parser.add_argument("--val_csv", type=str, default="myTrain/val_pairs.csv",
                        help="Path to the DPPI val pairs CSV (relative to data_dir).")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output folder for ESM-2 CSVs. Defaults to <data_dir>/transformer_data/")
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir  = args.out_dir or os.path.join(data_dir, "transformer_data")
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "sequences_cache.json")
    print(f"Output folder : {out_dir}")
    print(f "Sequence cache : {cache_path}")


    candidate_train = [
        os.path.join(data_dir, args.train_csv),
        os.path.join(data_dir, "myTrain", "train_pairs.csv"),
        os.path.join(data_dir, "train_pairs.csv"),
        os.path.join(data_dir, "AlphaYeastResults.csv"),
    ]
    candidate_val = [
        os.path.join(data_dir, args.val_csv),
        os.path.join(data_dir, "myTrain", "val_pairs.csv"),
        os.path.join(data_dir, "val_pairs.csv"),
    ]

    train_path = next((p for p in candidate_train if os.path.exists(p)), None)
    val_path   = next((p for p in candidate_val   if os.path.exists(p)), None)

    if not train_path:
        return

    print(f" Using train CSV : {train_path}")
    print(f" Using val CSV   : {val_path or 'NOT FOUND — will split train 90/10'}")

    train_df = load_pairs_csv(train_path)
    val_df   = load_pairs_csv(val_path) if val_path else None

    if val_df is None:
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df["label"])

    all_ids = (
        list(train_df["protein_a"]) + list(train_df["protein_b"]) +
        list(val_df["protein_a"])   + list(val_df["protein_b"])
    )
    seq_map = build_sequence_lookup(all_ids, cache_path=cache_path)

    print("\n Building transformer_train.csv ...")
    train_seq_df = pairs_to_seq_df(train_df, seq_map)
    print(f"  Label distribution:\n{train_seq_df['label'].value_counts().to_string()}")

    print("\n Building transformer_val.csv ...")
    val_seq_df = pairs_to_seq_df(val_df, seq_map)
    print(f"  Label distribution:\n{val_seq_df['label'].value_counts().to_string()}")

    train_out = os.path.join(out_dir, "transformer_train.csv")
    val_out   = os.path.join(out_dir, "transformer_val.csv")
    train_seq_df.to_csv(train_out, index=False)
    val_seq_df.to_csv(val_out, index=False)


if __name__ == "__main__":
    main()
