import os
import argparse
import math
import warnings
import logging
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

# ---------------- Silent mode ----------------
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from transformers import AutoTokenizer, EsmModel


# ---------------- Args ----------------
def parse_args():
    p = argparse.ArgumentParser(description="ESM-2 Siamese Transformer for PPI Prediction")
    p.add_argument("--train_csv", type=str, required=True, help="Path to transformer_train.csv")
    p.add_argument("--val_csv", type=str, required=True, help="Path to transformer_val.csv")
    p.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--emb_batch", type=int, default=8, help="Batch size for embedding pass")
    p.add_argument("--batch_size", type=int, default=256, help="Training batch size")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--out", type=str, default="best_transformer_model.pth")
    p.add_argument("--results", type=str, default="transformer_results.txt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--cache_path",
        type=str,
        default=None,
        help="Embedding cache file. Default: based on model name and max_len",
    )
    return p.parse_args()


# ---------------- Helpers ----------------
def load_pairs_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {"seq_a", "seq_b", "label"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{csv_path} must contain columns {sorted(required)}. "
            f"Found: {list(df.columns)}"
        )

    df = df.copy()
    df["seq_a"] = df["seq_a"].astype(str).str.upper()
    df["seq_b"] = df["seq_b"].astype(str).str.upper()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(float)
    return df


def cache_file_for(args) -> str:
    if args.cache_path:
        return args.cache_path
    safe_model = args.model_name.replace("/", "_")
    out_dir = os.path.dirname(args.out) or "."
    return os.path.join(out_dir, f"embeddings_cache_{safe_model}_len{args.max_len}.npz")


def save_embedding_cache(cache_path: str, embeddings: dict[str, np.ndarray]) -> None:
    seqs = np.array(list(embeddings.keys()), dtype=object)
    embs = np.stack([embeddings[s] for s in seqs]).astype(np.float32)
    np.savez_compressed(cache_path, sequences=seqs, embeddings=embs)


def load_embedding_cache(cache_path: str) -> dict[str, np.ndarray]:
    if not os.path.exists(cache_path):
        return {}

    data = np.load(cache_path, allow_pickle=True)
    seqs = data["sequences"].tolist()
    embs = data["embeddings"].astype(np.float32)

    return {seqs[i]: embs[i] for i in range(len(seqs))}


def precompute_embeddings(
    unique_seqs: list[str],
    model_name: str,
    max_len: int,
    batch_size: int,
    device: torch.device,
    cache_path: str,
) -> dict[str, np.ndarray]:
    embeddings = load_embedding_cache(cache_path)

    missing = [s for s in unique_seqs if s not in embeddings]
    if not missing:
        return embeddings

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    esm = EsmModel.from_pretrained(model_name).to(device).eval()

    for p in esm.parameters():
        p.requires_grad = False

    use_amp = device.type == "cuda"
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp
        else nullcontext()
    )

    for i in range(0, len(missing), batch_size):
        batch = missing[i:i + batch_size]

        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad(), amp_ctx:
            out = esm(input_ids=input_ids, attention_mask=attention_mask)

        hidden = out.last_hidden_state.float()
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        pooled_np = pooled.cpu().numpy().astype(np.float32)
        for j, seq in enumerate(batch):
            embeddings[seq] = pooled_np[j]

    save_embedding_cache(cache_path, embeddings)

    del esm, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return embeddings


# ---------------- Dataset ----------------
class PPICachedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, embeddings: dict[str, np.ndarray]):
        emb_a, emb_b, labels = [], [], []

        skipped = 0
        for _, row in df.iterrows():
            sa = row["seq_a"]
            sb = row["seq_b"]
            lbl = float(row["label"])

            ea = embeddings.get(sa)
            eb = embeddings.get(sb)

            if ea is None or eb is None:
                skipped += 1
                continue

            emb_a.append(ea)
            emb_b.append(eb)
            labels.append(lbl)

        if len(labels) == 0:
            raise ValueError("No valid pairs found after matching embeddings.")

        self.emb_a = torch.from_numpy(np.asarray(emb_a, dtype=np.float32))
        self.emb_b = torch.from_numpy(np.asarray(emb_b, dtype=np.float32))
        self.labels = torch.tensor(labels, dtype=torch.float32)

        if skipped:
            print(f"Skipped {skipped} pairs (missing embeddings)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.emb_a[idx], self.emb_b[idx], self.labels[idx]


# ---------------- Model ----------------
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
        x = torch.cat(
            [emb_a, emb_b, torch.abs(emb_a - emb_b), emb_a * emb_b],
            dim=1
        )
        return self.net(x).squeeze(-1)


# ---------------- Metrics ----------------
def compute_metrics(labels, probs):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs >= 0.5).astype(int)

    out = {
        "acc": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }

    if len(np.unique(labels)) > 1:
        out["auroc"] = roc_auc_score(labels, probs)
        out["auprc"] = average_precision_score(labels, probs)
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")

    return out


# ---------------- Train / Eval ----------------
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for emb_a, emb_b, labels in loader:
            emb_a = emb_a.to(device, non_blocking=True)
            emb_b = emb_b.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(emb_a, emb_b)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            all_probs.extend(torch.sigmoid(logits).detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, all_labels, all_probs


# ---------------- Main ----------------
def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = load_pairs_df(args.train_csv)
    val_df = load_pairs_df(args.val_csv)

    unique_seqs = sorted(set(
        train_df["seq_a"].tolist()
        + train_df["seq_b"].tolist()
        + val_df["seq_a"].tolist()
        + val_df["seq_b"].tolist()
    ))

    cache_path = cache_file_for(args)

    print(f"Device: {device}")
    print(f"Train pairs: {len(train_df):,}")
    print(f"Val pairs: {len(val_df):,}")
    print(f"Unique proteins: {len(unique_seqs):,}")

    embeddings = precompute_embeddings(
        unique_seqs=unique_seqs,
        model_name=args.model_name,
        max_len=args.max_len,
        batch_size=args.emb_batch,
        device=device,
        cache_path=cache_path,
    )

    emb_dim = len(next(iter(embeddings.values())))

    train_ds = PPICachedDataset(train_df, embeddings)
    val_ds = PPICachedDataset(val_df, embeddings)

    print(f"Train usable pairs: {len(train_ds):,}")
    print(f"Val usable pairs: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    model = PPIClassifier(emb_dim, args.hidden_dim, args.dropout).to(device)

    n_pos = int(train_ds.labels.sum().item())
    n_neg = len(train_ds) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=1e-6,
    )

    best_val_auprc = -float("inf")
    best_val_loss = float("inf")
    best_epoch = 0
    patience_cnt = 0
    history = []

    print()
    print(f"{'Ep':>4} │ {'TrLoss':>8} │ {'VaLoss':>8} │ {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUROC':>6} {'AUPRC':>6}")
    print(f"{'─'*88}")

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_labels, val_probs = run_epoch(model, val_loader, criterion, None, device, train=False)

        val_m = compute_metrics(val_labels, val_probs)

        print(
            f"{epoch:>4d} │ {train_loss:>8.4f} │ {val_loss:>8.4f} │ "
            f"{val_m['acc']:>6.4f} {val_m['precision']:>6.4f} {val_m['recall']:>6.4f} "
            f"{val_m['f1']:>6.4f} {val_m['auroc']:>6.4f} {val_m['auprc']:>6.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_m,
        })

        is_better = (
            val_m["auprc"] > best_val_auprc or
            (math.isclose(val_m["auprc"], best_val_auprc, abs_tol=1e-4) and val_loss < best_val_loss)
        )

        if is_better:
            best_val_auprc = val_m["auprc"]
            best_val_loss = val_loss
            best_epoch = epoch
            patience_cnt = 0
            torch.save(model.state_dict(), args.out)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                break

        scheduler.step()

    try:
        state_dict = torch.load(args.out, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(args.out, map_location=device)

    model.load_state_dict(state_dict)

    _, val_labels, val_probs = run_epoch(model, val_loader, criterion, None, device, train=False)
    final_m = compute_metrics(val_labels, val_probs)

    lines = [
        "=" * 60,
        "ESM-2 Siamese Transformer — Final Evaluation",
        "=" * 60,
        f"Model          : {args.model_name}",
        f"Embedding dim  : {emb_dim}",
        f"Best Epoch     : {best_epoch}",
        "",
        "Metrics (Validation Set)",
        f"Accuracy       : {final_m['acc']:.4f}",
        f"Precision      : {final_m['precision']:.4f}",
        f"Recall         : {final_m['recall']:.4f}",
        f"F1 Score       : {final_m['f1']:.4f}",
        f"AUROC          : {final_m['auroc']:.4f}",
        f"AUPRC          : {final_m['auprc']:.4f}",
        "",
        f"Cache file     : {cache_path}",
        "=" * 60,
    ]

    print("\n".join(lines))

    with open(args.results, "w") as f:
        f.write("\n".join(lines) + "\n\n")
        f.write("Epoch-level history:\n")
        f.write(
            f"{'Epoch':>5}  {'TrainLoss':>10}  {'ValLoss':>8}  "
            f"{'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'AUROC':>6}  {'AUPRC':>6}\n"
        )
        for h in history:
            f.write(
                f"{h['epoch']:>5}  {h['train_loss']:>10.4f}  {h['val_loss']:>8.4f}  "
                f"{h['acc']:>6.4f}  {h['precision']:>6.4f}  {h['recall']:>6.4f}  "
                f"{h['f1']:>6.4f}  {h['auroc']:>6.4f}  {h['auprc']:>6.4f}\n"
            )

    print(f"\nResults saved to: {args.results}")


if __name__ == "__main__":
    main()
