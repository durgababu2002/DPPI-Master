import os
import argparse
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
)

try:
    from transformers import AutoTokenizer, EsmModel
except ImportError:
    import subprocess, sys
    print("Installing transformers …")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers"])
    from transformers import AutoTokenizer, EsmModel

# argument parsing
def parse_args():
    p = argparse.ArgumentParser(description="ESM-2 Siamese Transformer for PPI Prediction (FAST)")
    p.add_argument("--train_csv",   type=str, required=True,  help="Path to transformer_train.csv")
    p.add_argument("--val_csv",     type=str, required=True,  help="Path to transformer_val.csv")
    p.add_argument("--model_name",  type=str, default="facebook/esm2_t33_650M_UR50D",
                   help="HuggingFace ESM-2 model name. Options:\n"
                        "  facebook/esm2_t6_8M_UR50D    (8M params)\n"
                        "  facebook/esm2_t12_35M_UR50D  (35M params)\n"
                        "  facebook/esm2_t30_150M_UR50D (150M params, default)\n"
                        "  facebook/esm2_t33_650M_UR50D (650M params)")
    p.add_argument("--max_len",     type=int, default=512,    help="Max sequence length (tokens).")
    p.add_argument("--emb_batch",   type=int, default=8,      help="Batch size for the one-time embedding pass")
    p.add_argument("--batch_size",  type=int, default=256,    help="Training batch size (fast now! can go big)")
    p.add_argument("--epochs",      type=int, default=50,     help="Number of training epochs")
    p.add_argument("--lr",          type=float, default=1e-3, help="Learning rate for classifier head")
    p.add_argument("--hidden_dim",  type=int, default=512,    help="Hidden dim in classifier MLP")
    p.add_argument("--dropout",     type=float, default=0.3,  help="Dropout rate in classifier MLP")
    p.add_argument("--patience",    type=int, default=10,     help="Early stopping patience")
    p.add_argument("--out",         type=str, default="best_transformer_model.pth")
    p.add_argument("--results",     type=str, default="transformer_results.txt")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()



def precompute_embeddings(unique_seqs: list[str], model_name: str, max_len: int,
                          batch_size: int, device: torch.device) -> dict[str, np.ndarray]:

    print(f"\n{'─'*65}")
    print(f"RE-COMPUTING EMBEDDINGS ({len(unique_seqs)} unique proteins)")
    print(f"  Model: {model_name} | Max len: {max_len} | Batch: {batch_size}")
    

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    esm = EsmModel.from_pretrained(model_name).to(device).eval()

    for param in esm.parameters():
        param.requires_grad = False

    esm_dim = esm.config.hidden_size
    embeddings: dict[str, np.ndarray] = {}

    # Process in batches
    t0 = time.time()
    for i in range(0, len(unique_seqs), batch_size):
        batch_seqs = unique_seqs[i : i + batch_size]
        enc = tokenizer(
            batch_seqs, return_tensors="pt",
            padding=True, truncation=True, max_length=max_len,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad(), autocast(dtype=torch.float16):
            out = esm(input_ids=input_ids, attention_mask=attention_mask)

        # Mean-pool over non-padding tokens
        hidden = out.last_hidden_state.float()  # (B, L, D)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # (B, D)

        pooled_np = pooled.cpu().numpy()
        for j, seq in enumerate(batch_seqs):
            embeddings[seq] = pooled_np[j]

        done = min(i + batch_size, len(unique_seqs))
        elapsed = time.time() - t0
        eta = (elapsed / done) * (len(unique_seqs) - done) if done > 0 else 0


    elapsed = time.time() - t0

    
    del esm, tokenizer
    torch.cuda.empty_cache()

    return embeddings


# dataset
class PPICachedDataset(Dataset):
    

    def __init__(self, csv_path: str, embeddings: dict[str, np.ndarray]):
        df = pd.read_csv(csv_path)
        assert {"seq_a", "seq_b", "label"}.issubset(df.columns)

        self.emb_a  = []
        self.emb_b  = []
        self.labels = []

        skipped = 0
        for _, row in df.iterrows():
            sa, sb = str(row["seq_a"]).upper(), str(row["seq_b"]).upper()
            if sa in embeddings and sb in embeddings:
                self.emb_a.append(embeddings[sa])
                self.emb_b.append(embeddings[sb])
                self.labels.append(float(row["label"]))
            else:
                skipped += 1

        # Stack into tensors for maximum speed
        self.emb_a  = torch.tensor(np.stack(self.emb_a), dtype=torch.float32)
        self.emb_b  = torch.tensor(np.stack(self.emb_b), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

        if skipped:
            print(f"Skipped {skipped} pairs (missing embeddings)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.emb_a[idx], self.emb_b[idx], self.labels[idx]


# classifier model
class PPIClassifier(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),

            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, emb_a, emb_b):
        combined = torch.cat([
            emb_a, emb_b,
            torch.abs(emb_a - emb_b),
            emb_a * emb_b,
        ], dim=1)
        return self.net(combined).squeeze(-1)


#metrics
def compute_metrics(labels, probs, threshold=0.5):
    preds  = (np.array(probs) >= threshold).astype(int)
    labels = np.array(labels)
    return dict(
        acc       = accuracy_score(labels, preds),
        precision = precision_score(labels, preds, zero_division=0),
        recall    = recall_score(labels, preds, zero_division=0),
        f1        = f1_score(labels, preds, zero_division=0),
        auroc     = roc_auc_score(labels, probs)  if len(set(labels)) > 1 else float("nan"),
        auprc     = average_precision_score(labels, probs) if len(set(labels)) > 1 else float("nan"),
    )

def metrics_str(m):
    return (f"Acc={m['acc']:.4f}  Prec={m['precision']:.4f}  Rec={m['recall']:.4f}  "
            f"F1={m['f1']:.4f}  AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}")


#train &evaluate
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []

    ctx = torch.no_grad() if not train else torch.enable_grad()
    with ctx:
        for emb_a, emb_b, labels in loader:
            emb_a, emb_b, labels = emb_a.to(device), emb_b.to(device), labels.to(device)

            logits = model(emb_a, emb_b)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    return total_loss / len(loader.dataset), all_labels, all_probs


#main
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*65}")
    print(f"  ESM-2 Siamese Transformer — PPI Prediction (FAST MODE)")
    print(f"{'='*65}")
    print(f"  Device     : {device}")
    print(f"  ESM model  : {args.model_name}")
    print(f"  Max seqlen : {args.max_len}")
    print(f"  Emb batch  : {args.emb_batch}")
    print(f"  Train batch: {args.batch_size}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  LR         : {args.lr}")
    print(f"{'='*65}")

    
    train_df = pd.read_csv(args.train_csv)
    val_df   = pd.read_csv(args.val_csv)

    all_seqs = set()
    for df in [train_df, val_df]:
        all_seqs.update(str(s).upper() for s in df["seq_a"])
        all_seqs.update(str(s).upper() for s in df["seq_b"])
    unique_seqs = sorted(all_seqs)

    print(f"\n  Total train pairs  : {len(train_df):,}")
    print(f"  Total val pairs    : {len(val_df):,}")
    print(f"  Unique proteins    : {len(unique_seqs):,}")

    
    embeddings = precompute_embeddings(
        unique_seqs, args.model_name, args.max_len, args.emb_batch, device
    )
    emb_dim = len(next(iter(embeddings.values())))

    # Build fast datasets
    print("Building cached datasets …")
    train_ds = PPICachedDataset(args.train_csv, embeddings)
    val_ds   = PPICachedDataset(args.val_csv,   embeddings)
    print(f"  Train: {len(train_ds):,} pairs  |  Val: {len(val_ds):,} pairs")
    print(f"  Train labels: pos={int(train_ds.labels.sum())}, neg={len(train_ds)-int(train_ds.labels.sum())}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, pin_memory=True)

    #  Train the lightweight MLP
    print(f"\n Building classifier (emb_dim={emb_dim}) …")
    model = PPIClassifier(emb_dim, args.hidden_dim, args.dropout).to(device)
    trainable = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Pos weight for class imbalance
    n_pos = int(train_ds.labels.sum())
    n_neg = len(train_ds) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # training loop
    print(f"\n{'─'*90}")
    print(f"{'Ep':>4} │ {'TrLoss':>8} │ {'VaLoss':>8} │ {'Acc':>6} {'Prec':>6} {'Rec':>6} "
          f"{'F1':>6} {'AUROC':>6} {'AUPRC':>6} │ {'Time':>5}")
    print(f"{'─'*90}")

    best_val_auroc = 0.0
    best_val_loss  = float("inf")
    patience_cnt   = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_labels, val_probs = run_epoch(model, val_loader, criterion, None, device, train=False)

        val_m = compute_metrics(val_labels, val_probs)
        elapsed = time.time() - t0

        print(f"{epoch:>4d} │ {train_loss:>8.4f} │ {val_loss:>8.4f} │ "
              f"{val_m['acc']:>6.4f} {val_m['precision']:>6.4f} {val_m['recall']:>6.4f} "
              f"{val_m['f1']:>6.4f} {val_m['auroc']:>6.4f} {val_m['auprc']:>6.4f} │ "
              f"{elapsed:>4.1f}s", end="")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_m})

        if val_m["auroc"] > best_val_auroc or (
            math.isclose(val_m["auroc"], best_val_auroc, abs_tol=1e-4) and val_loss < best_val_loss
        ):
            best_val_auroc = val_m["auroc"]
            best_val_loss  = val_loss
            patience_cnt   = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "emb_dim": emb_dim, "val_metrics": val_m, "args": vars(args),
            }, args.out)
            print(f"saved")
        else:
            patience_cnt += 1
            print()
            if patience_cnt >= args.patience:
                print(f"\early stopping at epoch {epoch}.")
                break

        scheduler.step()

    
    

    ckpt = torch.load(args.out, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, val_labels, val_probs = run_epoch(model, val_loader, criterion, None, device, train=False)
    final_m = compute_metrics(val_labels, val_probs)

    lines = [
        "=" * 60,
        "  ESM-2 Siamese Transformer — Final Evaluation",
        "=" * 60,
        f"  Model          : {args.model_name}",
        f"  Embedding dim  : {emb_dim}",
        f"  Best Epoch     : {ckpt['epoch']}",
        "",
        "  ── Metrics (Validation Set) ──",
        f"  Accuracy       : {final_m['acc']:.4f}",
        f"  Precision      : {final_m['precision']:.4f}",
        f"  Recall         : {final_m['recall']:.4f}",
        f"  F1 Score       : {final_m['f1']:.4f}",
        f"  AUROC          : {final_m['auroc']:.4f}",
        f"  AUPRC          : {final_m['auprc']:.4f}",
        "",
        "  ── Comparison Guide ──",
        "  Compare these against old CNN (DPPI) results.",
        "=" * 60,
    ]
    print("\n".join(lines))

    with open(args.results, "w") as f:
        f.write("\n".join(lines) + "\n\n")
        f.write("Epoch-level history:\n")
        f.write(f"{'Epoch':>5}  {'TrainLoss':>10}  {'ValLoss':>8}  "
                f"{'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'AUROC':>6}  {'AUPRC':>6}\n")
        for h in history:
            f.write(f"{h['epoch']:>5}  {h['train_loss']:>10.4f}  {h['val_loss']:>8.4f}  "
                    f"{h['acc']:>6.4f}  {h['precision']:>6.4f}  {h['recall']:>6.4f}  "
                    f"{h['f1']:>6.4f}  {h['auroc']:>6.4f}  {h['auprc']:>6.4f}\n")

    print(f"\nResults saved to: {args.results}")


if __name__ == "__main__":
    main()
