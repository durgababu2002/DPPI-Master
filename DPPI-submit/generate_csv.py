import pandas as pd

pos_file = "/content/drive/MyDrive/DPPI-master/human_pos.txt"
neg_file = "/content/drive/MyDrive/DPPI-master/human_neg.txt"
output_csv = "/content/drive/MyDrive/DPPI-master/AlphaYeastResults.csv"

try:
    df_pos = pd.read_csv(pos_file, sep=r'\s+', header=None, names=['Protein1', 'Protein2'])
    df_pos['Label'] = 1  

    df_neg = pd.read_csv(neg_file, sep=r'\s+', header=None, names=['Protein1', 'Protein2'])
    df_neg['Label'] = 0  

    df_combined = pd.concat([df_pos, df_neg], ignore_index=True)
    
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    df_combined.to_csv(output_csv, index=False, header=False)

    print(f"Total Training Pairs: {len(df_combined)}")
    print(f" - Positives (1s): {len(df_pos)}")
    print(f" - Negatives (0s): {len(df_neg)}")

except FileNotFoundError as e:
    print(f"ERROR: Could not find the text files.")
    print(f"Details: {e}")