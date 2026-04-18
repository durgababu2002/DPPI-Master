import os
import torch
import numpy as np
import argparse

BACKGROUND = np.array([
    0.07999, 0.04844, 0.04429, 0.04578, 0.01718,
    0.03805, 0.06381, 0.07606, 0.02234, 0.05509,
    0.08668, 0.06045, 0.02153, 0.03963, 0.04657,
    0.06300, 0.05803, 0.01449, 0.03635, 0.07002
], dtype=np.float32)

C_CONST = 0.8
CROP_SIZE = 512

BG_TERM = (1.0 - C_CONST) * BACKGROUND


def fast_pssm_read(file_path):
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i < 3:
                continue
            parts = line.split()
            if len(parts) < 42 or not parts[0].isdigit():
                continue
            data.append(parts[22:42])
    return np.asarray(data, dtype=np.float32)


def process_protein(file_path):
    try:
        profile_data = fast_pssm_read(file_path)
        length = len(profile_data)

        if length == 0:
            return {}, 0

        num_crops = max(1, int(np.ceil(2 * length / CROP_SIZE - 1)))

        starts = [(i * CROP_SIZE) // 2 for i in range(num_crops)]
        starts = [min(s, length - CROP_SIZE) for s in starts]
        starts = [max(0, s) for s in starts]

        crops = {}
        base_crop = np.zeros((1, 20, CROP_SIZE), dtype=np.float32)

        for idx, start in enumerate(starts):
            chunk = profile_data[start:start + CROP_SIZE]
            actual_len = len(chunk)

            transformed = np.log(C_CONST * (chunk * 0.01) + BG_TERM)

            crop_arr = base_crop.copy()
            crop_arr[0, :, :actual_len] = transformed.T

            crops[f"sub{idx+1}"] = torch.from_numpy(crop_arr)

        return crops, num_crops

    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        return {}, 0


def run_preprocessing(dataset_name, data_dir):
    node_file = os.path.join(data_dir, f"{dataset_name}.node")

    with open(node_file, 'r') as f:
        proteins = [line.strip() for line in f if line.strip()]

    from pathlib import Path
    search_path = os.path.join(data_dir, dataset_name)

    
    all_pssms = {p.name[:-5]: str(p) for p in Path(search_path).glob('*.pssm')}

    feature_dict = {}
    p_numbers = {}

    total = len(proteins)

    for i, p_id in enumerate(proteins):
        if p_id in all_pssms:
            path = all_pssms[p_id]

            crops, n = process_protein(path)

            for sub_id, tensor in crops.items():
                feature_dict[f"{p_id}-{sub_id}"] = tensor

            p_numbers[p_id] = n

        if (i + 1) % 500 == 0:
            print(f"Processed {i+1}/{total} proteins...")

    torch.save(feature_dict, os.path.join(data_dir, f"{dataset_name}_features.pt"))
    torch.save(p_numbers, os.path.join(data_dir, f"{dataset_name}_counts.pt"))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, required=True)
    parser.add_argument('-data_dir', type=str, default='./')

    args = parser.parse_args()

    run_preprocessing(args.dataset, args.data_dir)