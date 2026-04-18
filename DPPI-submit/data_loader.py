import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class PPIDataset(Dataset):
    def __init__(self, csv_path, feature_path, counts_path):
        #format is: protein1, protein2, interaction label
        self.df = pd.read_csv(csv_path, header=None)
        self.features = torch.load(feature_path)
        self.counts = torch.load(counts_path)

        self.pairs = []
        for _, row in self.df.iterrows():
            if len(row) >= 3:
                p1, p2, label = row[0], row[1], float(row[2])

                if p1 in self.counts and p2 in self.counts:
                    for i in range(1, self.counts[p1] + 1):
                        for j in range(1, self.counts[p2] + 1):

                            p1_sub = f"{p1}-sub{i}"
                            p2_sub = f"{p2}-sub{j}"

                            if p1_sub in self.features and p2_sub in self.features:
                                self.pairs.append((p1_sub, p2_sub, label))
        print("CSV rows:", len(self.df))
        print("Final pairs:", len(self.pairs))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1_name, p2_name, label = self.pairs[idx]

        # Random swap
        if np.random.rand() > 0.5:
            p1_name, p2_name = p2_name, p1_name

        feat1 = self.features[p1_name].squeeze(0).unsqueeze(1)  # (20,1,512)
        feat2 = self.features[p2_name].squeeze(0).unsqueeze(1)

        parent_p1 = p1_name.split("-sub")[0]
        parent_p2 = p2_name.split("-sub")[0]

        return feat1, feat2, torch.tensor([label], dtype=torch.float32), parent_p1, parent_p2
