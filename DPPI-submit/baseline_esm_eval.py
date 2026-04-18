
import os
import json
import argparse
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, EsmModel
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--val_csv", type=str, required=True, help="Path to transformer_val.csv")
    p.add_argument("--cache",   type=str, required=True, help="Path to sequences_cache.json")
    p.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--max_len", type=int, default=512)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    print(f"  ZERO-SHOT BASELINE EVALUATION (Cosine Similarity)")
    print(f"  Model: {args.model_name}")


    # 1. Load Data
    df = pd.read_csv(args.val_csv)
    with open(args.cache, 'r') as f:
        seq_map = json.load(f)
    
    print(f"  Loaded {len(df)} pairs for evaluation.")

    # 2. Load Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    esm = EsmModel.from_pretrained(args.model_name).to(device).eval()
    
    similarities = []
    labels = []

    # 3. Predict via Cosine Similarity
    print("\n  Calculating similarities...")
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            sa, sb = str(row['seq_a']).upper(), str(row['seq_b']).upper()
            lbl = int(row['label'])
            
            # Get embeddings 
            def get_emb(s):
                inputs = tokenizer(s[:args.max_len], return_tensors="pt", padding=True, truncation=True).to(device)
                out = esm(**inputs)
                # Mean pool
                return out.last_hidden_state.mean(dim=1)

            v_a = get_emb(sa)
            v_b = get_emb(sb)
            
            # Cosine similarity (ranges from -1 to 1)
            sim = torch.sum(v_a * v_b, dim=1).item()
            similarities.append(sim)
            labels.append(lbl)

    # 4. Metrics
    auroc = roc_auc_score(labels, similarities)
    auprc = average_precision_score(labels, similarities)
    
    best_acc = 0
    for t in np.linspace(-1, 1, 100):
        preds = (np.array(similarities) >= t).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc

    print(f"\n{'='*60}")
    print(f"  BASELINE RESULTS")
    print(f"   AUROC : {auroc:.4f}")
    print(f"   AUPRC : {auprc:.4f}")
    print(f"  Best Possible Acc: {best_acc:.4f} (at optimal threshold)")


if __name__ == "__main__":
    main()
