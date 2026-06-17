import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from rinalmo.config import model_config
from rinalmo.data.alphabet import Alphabet
from rinalmo.model.model import RiNALMo
from dependency_map import DependencyMap, DependencyMapOptions

def read_fasta(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        seq = "".join([line.strip() for line in lines if not line.startswith(">")])
    return seq

def main():
    parser = argparse.ArgumentParser(description="Compute RiNALMo reconstruction accuracy on annotated elements")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input directory containing FASTA and CSV files")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory for saving results")
    parser.add_argument("--test_name", type=str, default="test", help="Test name to append to the output filename")
    parser.add_argument("--model_name", type=str, default="mega", help="RiNALMo model config name (e.g. giga, mega, micro)")
    parser.add_argument("--weights_path", type=str, default=None, help="Path to model weights")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--context_window", type=int, default=1000, help="Max context window size")
    parser.add_argument("--stride", type=int, default=100, help="Sliding window stride (number of bases to mask per window)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Model
    weights_path = args.weights_path
    if weights_path is None:
        weights_path = Path(__file__).resolve().parents[1] / "models" / "rinalmo" / f"rinalmo_{args.model_name}_pretrained.pt"
    else:
        weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"Missing RiNALMo weights: {weights_path}")

    print(f"Loading {args.model_name} from {weights_path}...")
    config = model_config(args.model_name)
    alphabet = Alphabet(**config["alphabet"])
    model = RiNALMo(config)

    state_dict = torch.load(weights_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()

    # Fast tokenization setup
    mask_idx = alphabet.mask_idx
    sequence_tokens = torch.tensor([alphabet.get_idx(c) for c in "ACGT"])
    char_map = np.full(256, alphabet.unk_idx, dtype=np.int64)
    for char, idx in alphabet.tkn_to_idx.items():
        if len(char) == 1:
            char_map[ord(char)] = idx

    def tokenize_func(sequence: str, mask: int | None) -> list[int]:
        sequence = sequence.replace("U", "T")
        arr = np.frombuffer(sequence.encode('ascii'), dtype=np.uint8)
        encoded = char_map[arr].tolist()
        encoded.insert(0, alphabet.cls_idx)
        encoded.append(alphabet.eos_idx)
        if mask is not None:
            encoded[mask + 1] = mask_idx
        return encoded

    def forward_func(batch: list[list[int]]) -> np.ndarray:
        max_len = max(len(seq) for seq in batch)
        padded_batch = [seq + [alphabet.pad_idx] * (max_len - len(seq)) for seq in batch]
        tokens = torch.tensor(padded_batch, dtype=torch.int64, device=device)
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type if device.type == "cuda" else "cpu"):
                outputs = model(tokens)
                logits = outputs["logits"]
        return logits[:, 1:-1, sequence_tokens].cpu().numpy()

    # 2. Process Files
    all_results = []
    char_to_idx = {c: i for i, c in enumerate("ACGT")}

    csv_files = list(input_dir.glob("*_cleaned_matches.csv"))
    print(f"Found {len(csv_files)} annotation CSV files in {input_dir}.")

    for csv_path in csv_files:
        fasta_path = str(csv_path).replace("_cleaned_matches.csv", ".fasta")
        if not Path(fasta_path).exists():
            print(f"Warning: FASTA not found for {csv_path.name}, skipping.")
            continue

        print(f"\nProcessing {csv_path.name}...")
        df = pd.read_csv(csv_path)
        sequence = read_fasta(fasta_path).upper().replace("U", "T")
        seq_len = len(sequence)
        
        true_indices = np.array([char_to_idx.get(nt, -1) for nt in sequence])

        # Find which bases need to be computed
        to_compute = np.zeros(seq_len, dtype=bool)
        for _, row in df.iterrows():
            start, end = int(row['start']), int(row['end'])
            to_compute[start : end + 1] = True

        global_reconstruction_sum = np.zeros((seq_len, 4))
        global_reconstruction_count = np.zeros(seq_len)

        # 3. Slide Window (Overlapping)
        windows_to_run = []
        for window_start in range(0, seq_len, args.stride):
            window_end = min(seq_len, window_start + args.context_window)
            annotated_in_window = to_compute[window_start:window_end]
            if np.any(annotated_in_window):
                windows_to_run.append((window_start, window_end, annotated_in_window))
        
        print(f"Identified {len(windows_to_run)} overlapping windows to compute.")

        for window_start, window_end, annotated_in_window in tqdm(windows_to_run, desc="Running overlapping windows"):
            window_seq = sequence[window_start:window_end]

            # To save compute, we only mask from the first annotated base to the last annotated base IN THIS WINDOW
            first_annotated = np.argmax(annotated_in_window)
            last_annotated = len(annotated_in_window) - 1 - np.argmax(annotated_in_window[::-1])
            
            subset_start_rel = first_annotated
            subset_end_rel = last_annotated + 1

            options = DependencyMapOptions(
                subset=(subset_start_rel, subset_end_rel),
                with_reconstruction=True,
                dependency_by_masking=True
            )

            dep_map = DependencyMap.compute_batched(
                window_seq,
                tokenize_func,
                forward_func,
                batch_size=args.batch_size,
                enable_progress_bar=False,
                options=options
            )
            
            abs_start = window_start + subset_start_rel
            abs_end = window_start + subset_end_rel
            
            global_reconstruction_sum[abs_start:abs_end] += dep_map.reconstruction
            global_reconstruction_count[abs_start:abs_end] += 1

        # Calculate the final average probabilities across all overlapping contexts
        # Ignore division by zero for bases we never computed
        with np.errstate(invalid='ignore'):
            global_reconstruction = global_reconstruction_sum / global_reconstruction_count[:, None]

        # 4. Evaluate Elements
        species_name = csv_path.name.replace("_cleaned_matches.csv", "")
        
        for _, row in df.iterrows():
            label = row['label']
            el_type = row['type']
            start, end = int(row['start']), int(row['end'])
            
            recon_slice = global_reconstruction[start : end + 1]
            true_slice = true_indices[start : end + 1]
            
            # Filter out invalid bases (like N) from accuracy
            valid_mask = (true_slice != -1) & ~np.isnan(recon_slice[:, 0])
            if not np.any(valid_mask):
                continue
                
            recon_valid = recon_slice[valid_mask]
            true_valid = true_slice[valid_mask]
            
            pred_indices = np.argmax(recon_valid, axis=1)
            accuracy = np.mean(pred_indices == true_valid)
            
            correct_base_probs = recon_valid[np.arange(len(true_valid)), true_valid]
            avg_confidence = np.mean(correct_base_probs)
            cross_entropy = -np.mean(np.log(correct_base_probs + 1e-12))
            
            # Baseline calculation
            freqs = np.bincount(true_valid, minlength=4) / len(true_valid)
            baseline_acc = np.sum(freqs ** 2)
            baseline_ce = -np.sum(freqs * np.log(freqs + 1e-12))
            
            all_results.append({
                "Species": species_name,
                "Label": label,
                "Type": el_type,
                "Length": len(true_valid),
                "Accuracy": accuracy,
                "Baseline Accuracy": baseline_acc,
                "Avg Confidence": avg_confidence,
                "Cross-Entropy": cross_entropy,
                "Baseline CE": baseline_ce
            })
            
        # Save individual file results
        species_results = [r for r in all_results if r["Species"] == species_name]
        if species_results:
            species_df = pd.DataFrame(species_results)
            species_output_file = output_dir / f"{species_name}_{args.test_name}_reconstruction_accuracy.csv"
            species_df.to_csv(species_output_file, index=False)
            print(f"Saved {species_name} results to {species_output_file}")

    # 5. Summarize Results
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_file = output_dir / f"all_species_{args.test_name}_reconstruction_accuracy.csv"
        
        # Calculate summary statistics
        total_len = results_df['Length'].sum()
        global_weighted = (results_df['Length'] * results_df['Accuracy']).sum() / total_len
        global_weighted_base_acc = (results_df['Length'] * results_df['Baseline Accuracy']).sum() / total_len
        global_weighted_ce = (results_df['Length'] * results_df['Cross-Entropy']).sum() / total_len
        global_weighted_base_ce = (results_df['Length'] * results_df['Baseline CE']).sum() / total_len
        global_weighted_conf = (results_df['Length'] * results_df['Avg Confidence']).sum() / total_len
        
        global_mean = results_df['Accuracy'].mean()
        global_median = results_df['Accuracy'].median()
        global_ce_mean = results_df['Cross-Entropy'].mean()
        
        global_base_acc_mean = results_df['Baseline Accuracy'].mean()
        global_base_ce_mean = results_df['Baseline CE'].mean()

        # Append summary rows
        summary_rows = [
            {"Species": "SUMMARY", "Label": "Overall Mean", "Type": "ALL", "Length": "", 
             "Accuracy": global_mean, "Baseline Accuracy": global_base_acc_mean,
             "Avg Confidence": "", "Cross-Entropy": global_ce_mean, "Baseline CE": global_base_ce_mean},
            {"Species": "SUMMARY", "Label": "Overall Median", "Type": "ALL", "Length": "", 
             "Accuracy": global_median, "Baseline Accuracy": results_df['Baseline Accuracy'].median(),
             "Avg Confidence": "", "Cross-Entropy": results_df['Cross-Entropy'].median(), "Baseline CE": results_df['Baseline CE'].median()},
            {"Species": "SUMMARY", "Label": "Overall Weighted Mean", "Type": "ALL", "Length": total_len, 
             "Accuracy": global_weighted, "Baseline Accuracy": global_weighted_base_acc,
             "Avg Confidence": global_weighted_conf, "Cross-Entropy": global_weighted_ce, "Baseline CE": global_weighted_base_ce}
        ]
        results_df = pd.concat([results_df, pd.DataFrame(summary_rows)], ignore_index=True)
        
        # Save to CSV
        results_df.to_csv(output_file, index=False)
        print(f"\nSaved results to {output_file}")
        
        print("\n=== Global Averages by Type ===")
        def weighted_avg(group):
            # Ignore summary rows when grouping
            group = group[group['Species'] != 'SUMMARY']
            if len(group) == 0: return np.nan
            d = pd.to_numeric(group['Length'])
            w = pd.to_numeric(group['Accuracy'])
            return (d * w).sum() / d.sum()
            
        summary = results_df[results_df['Species'] != 'SUMMARY'].groupby("Type").apply(weighted_avg).reset_index(name="Weighted Accuracy")
        summary['Weighted Accuracy'] = (summary['Weighted Accuracy'] * 100).round(2).astype(str) + "%"
        print(summary.to_string(index=False))
        
        print(f"\nTotal Global Weighted Accuracy: {global_weighted * 100:.2f}%\n")
    else:
        print("No valid elements found to evaluate.")

if __name__ == "__main__":
    main()
