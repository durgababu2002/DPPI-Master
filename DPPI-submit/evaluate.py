import torch
from sklearn.metrics import precision_recall_curve, auc

def evaluate_all(model, data_loader):
    device = next(model.parameters()).device

    model.eval()
    all_scores = []
    all_targets = []

    with torch.no_grad():
        for p1_crops, p2_crops, target in data_loader:

            p1_crops = p1_crops.to(device)
            p2_crops = p2_crops.to(device)
            target = target.to(device)

            scores = model(p1_crops, p2_crops).squeeze()
            all_scores.extend(scores.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    precision, recall, _ = precision_recall_curve(all_targets, all_scores)
    auPR = auc(recall, precision)

    return auPR
