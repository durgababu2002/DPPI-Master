import pandas as pd
import torch

def build_dataset_with_split(
    hint_path,
    dataset_name,
    train_csv="train.csv",
    val_csv="val.csv",
    val_frac=0.2,
    seed=42
):
    df = pd.read_csv(hint_path, header=None, names=["Uniprot_A", "Uniprot_B", "label"])

    df = df[df["label"].isin([0, 1])].copy()

    counts = torch.load(f"{dataset_name}_counts.pt")
    valid_proteins = set(counts.keys())

    print(f"Proteins with features: {len(valid_proteins)}")

    df = df[
        df["Uniprot_A"].isin(valid_proteins) &
        df["Uniprot_B"].isin(valid_proteins)
    ].copy()

    df["pair"] = df.apply(
        lambda x: tuple(sorted([x["Uniprot_A"], x["Uniprot_B"]])),
        axis=1
    )

    conflict_mask = df.groupby("pair")["label"].transform("nunique") > 1
    if conflict_mask.any():
        bad_pairs = df.loc[conflict_mask, "pair"].drop_duplicates()
        print(f"Warning: dropping {len(bad_pairs)} conflicting pairs with both labels.")
        df = df[~df["pair"].isin(bad_pairs)].copy()

    df = df.drop_duplicates("pair", keep="first").copy()

    pos_df = df[df["label"] == 1][["Uniprot_A", "Uniprot_B", "label"]].copy()
    neg_df = df[df["label"] == 0][["Uniprot_A", "Uniprot_B", "label"]].copy()

    print(f"Remaining positives: {len(pos_df)}")
    print(f"Remaining negatives: {len(neg_df)}")

    pos_df = pos_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    neg_df = neg_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    pos_cut = int((1 - val_frac) * len(pos_df))
    neg_cut = int((1 - val_frac) * len(neg_df))

    train_pos = pos_df.iloc[:pos_cut]
    val_pos = pos_df.iloc[pos_cut:]

    train_neg = neg_df.iloc[:neg_cut]
    val_neg = neg_df.iloc[neg_cut:]

    train_df = pd.concat([train_pos, train_neg], ignore_index=True)
    val_df = pd.concat([val_pos, val_neg], ignore_index=True)

    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    train_df.to_csv(train_csv, index=False, header=False)
    val_df.to_csv(val_csv, index=False, header=False)

    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print("\nTrain label distribution:")
    print(train_df["label"].value_counts())
    print("\nVal label distribution:")
    print(val_df["label"].value_counts())


if __name__ == "__main__":
    build_dataset_with_split(
        hint_path="/content/drive/MyDrive/DPPI-master/AlphaYeastResults.csv",
        dataset_name="AlphaYeastResults",
        train_csv="myTrain.csv",
        val_csv="myTrain_valid.csv",
        val_frac=0.2,
        seed=42
    )