import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#evaluation metric 
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, average_precision_score, roc_auc_score
)
from model import DPPI_Model
from data_loader import PPIDataset
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', type=str, default=None)
    parser.add_argument('-train_csv', type=str, default=None)
    parser.add_argument('-val_csv', type=str, default=None)

    parser.add_argument('-learningRate', type=float, default=0.01)
    parser.add_argument('-batchSize', type=int, default=512)
    parser.add_argument('-epochs', type=int, default=500)
    parser.add_argument('-momentum', type=float, default=0.9)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # input handling
    if args.dataset is not None:
        train_csv = f"{args.dataset}.csv"
        val_csv = f"{args.dataset}_valid.csv"
    else:
        if args.train_csv is None or args.val_csv is None:
            raise ValueError("Provide either -dataset OR both -train_csv and -val_csv")
        train_csv = args.train_csv
        val_csv = args.val_csv

    feature_path = "AlphaYeastResults_features.pt" if args.dataset else "AlphaYeastResults_features.pt"
    counts_path = "AlphaYeastResults_counts.pt" if args.dataset else "AlphaYeastResults_counts.pt"

    # data 
    train_dataset = PPIDataset(train_csv, feature_path, counts_path)
    val_dataset = PPIDataset(val_csv, feature_path, counts_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batchSize,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batchSize,
        shuffle=False,
        pin_memory=True
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    # model 
    model = DPPI_Model(num_features=20, crop_length=512).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learningRate,
        momentum=args.momentum,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[300, 400], gamma=0.1
    )

    # training loop
    best_auprc = 0.0
    for epoch in range(args.epochs):

        
        model.train()
        running_train_loss = 0.0

        
        for p1, p2, labels, _, _ in train_loader:
            p1 = p1.to(device, non_blocking=True)
            p2 = p2.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(p1, p2)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        scheduler.step()

        # validation 
        model.eval()
        running_val_loss = 0.0

        pair_to_max_score = {}
        pair_to_label = {}

        with torch.no_grad():
            for p1, p2, labels, parent_p1_batch, parent_p2_batch in val_loader:
                p1 = p1.to(device, non_blocking=True)
                p2 = p2.to(device, non_blocking=True)
                labels_dev = labels.to(device, non_blocking=True)

                outputs = model(p1, p2)

                val_loss = criterion(outputs, labels_dev)
                running_val_loss += val_loss.item()

                cpu_outputs = outputs.detach().cpu().numpy()
                cpu_labels = labels.numpy()

                for i in range(len(cpu_labels)):
                    pair_tuple = tuple(sorted([parent_p1_batch[i], parent_p2_batch[i]]))
                    label_val = cpu_labels[i][0]
                    score_val = cpu_outputs[i][0] # Raw logit
                    
                    if pair_tuple not in pair_to_max_score:
                        pair_to_max_score[pair_tuple] = score_val
                        pair_to_label[pair_tuple] = label_val
                    else:
                        pair_to_max_score[pair_tuple] = max(pair_to_max_score[pair_tuple], score_val)

        avg_val_loss = running_val_loss / len(val_loader)

        #  metrics 
        all_targets = np.array(list(pair_to_label.values()))
        all_scores = np.array(list(pair_to_max_score.values()))

        # Convert logits to probabilities 
        probs = 1 / (1 + np.exp(-all_scores))
        preds = (probs > 0.5).astype(int)

        accuracy  = accuracy_score(all_targets, preds)
        precision = precision_score(all_targets, preds, zero_division=0)
        recall    = recall_score(all_targets, preds, zero_division=0)
        f1        = f1_score(all_targets, preds, zero_division=0)
        cm        = confusion_matrix(all_targets, preds)

        try:
            auprc = average_precision_score(all_targets, probs)
        except:
            auprc = 0.0

        try:
            auroc = roc_auc_score(all_targets, probs)
        except:
            auroc = 0.0

        print(
            f"Epoch {epoch+1:02d}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"AUPRC: {auprc:.4f} | AUROC: {auroc:.4f} | "
            f"Acc: {accuracy:.4f} | Prec: {precision:.4f} | "
            f"Recall: {recall:.4f} | F1: {f1:.4f}"
        )

        # save best model 
        if auprc > best_auprc:
            best_auprc = auprc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auprc': best_auprc,
                'auroc': auroc,
            }, 'best_dppi_model.pth')
            print(f" Saved best model (AUPRC: {best_auprc:.4f})")

        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_auprc': best_auprc,
        }, 'latest_model.pt')

if __name__ == "__main__":
    main()

