import os
import argparse
from pathlib import Path

def generate_node_file(folder_path):
    folder_path = os.path.normpath(folder_path)
    
    if not os.path.exists(folder_path):
        print(f"Error: The directory '{folder_path}' does not exist.")
        return

    
    parent_dir = os.path.dirname(folder_path)
    dataset_name = os.path.basename(folder_path)
    output_file = os.path.join(parent_dir, f"{dataset_name}.node")
    
    
    try:
        pssm_files = list(Path(folder_path).rglob('*.pssm'))
    except Exception as e:
        print(f"error reading directory: {e}")
        return
    
    if not pssm_files:
        print(f" Warning: Found 0 files ending with '.pssm'.")
        return

    with open(output_file, 'w') as f:
        for file_path in sorted(pssm_files):
            protein_id = file_path.name[:-5]
            f.write(f"{protein_id}\n")
            
    print(f"generated {output_file} containing {len(pssm_files)} protein entries.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-generate a DPPI .node file from a folder of PSSMs.")
    parser.add_argument("-folder", type=str, required=True, help="Absolute path to the folder containing .pssm files.")
    
    args = parser.parse_args()
    generate_node_file(args.folder)