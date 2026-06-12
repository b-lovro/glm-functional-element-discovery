import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from rinalmo.config import model_config
from rinalmo.data.alphabet import Alphabet
from rinalmo.model.model import RiNALMo
from dependency_map import DependencyMap, DependencyMapOptions

def read_fasta(file_path, max_sequences=None):
    sequences = []
    with open(file_path, "r") as f:
        seq_id = ""
        seq = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id:
                    sequences.append((seq_id, "".join(seq)))
                    if max_sequences is not None and len(sequences) >= max_sequences:
                        return sequences
                seq_id = line[1:].split()[0] # take the first word as id
                seq = []
            else:
                seq.append(line)
        if seq_id and (max_sequences is None or len(sequences) < max_sequences):
            sequences.append((seq_id, "".join(seq)))
    return sequences

def main():
    parser = argparse.ArgumentParser(description="Run RiNALMo inference and Dependency Map")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input FASTA file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save dependency maps")
    parser.add_argument("--max_sequences", type=int, default=None, help="Maximum number of sequences to process")
    parser.add_argument("--model_name", type=str, default="mega", help="RiNALMo model config name to use (e.g. giga, mega, micro)")
    parser.add_argument("--weights_path", type=str, default=None, help="Path to the model weights. If not provided, it will look in models/rinalmo/rinalmo_<model_name>_pretrained.pt")
    parser.add_argument("--subset_start", type=int, default=None, help="Subset start index for dependency map")
    parser.add_argument("--subset_end", type=int, default=None, help="Subset end index for dependency map")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--test_name", type=str, default="test", help="Name of the test to append to the output filename")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"Reading {input_path}...")
    sequences = read_fasta(input_path, max_sequences=args.max_sequences)
    print(f"Found {len(sequences)} sequences to process.")

    mask_idx = alphabet.mask_idx
    sequence_tokens = torch.tensor([alphabet.get_idx(c) for c in "ACGT"])

    # Create a fast lookup array for tokenization to avoid python loop overhead
    char_map = np.full(256, alphabet.unk_idx, dtype=np.int64)
    for char, idx in alphabet.tkn_to_idx.items():
        if len(char) == 1:
            char_map[ord(char)] = idx

    def tokenize_func(sequence: str, mask: int | None) -> list[int]:
        sequence = sequence.replace("U", "T")
        # Extremely fast tokenization in C (numpy)
        arr = np.frombuffer(sequence.encode('ascii'), dtype=np.uint8)
        encoded = char_map[arr].tolist()
        encoded.insert(0, alphabet.cls_idx)
        encoded.append(alphabet.eos_idx)
        if mask is not None:
            encoded[mask + 1] = mask_idx # +1 for CLS
        return encoded

    def forward_func(batch: list[list[int]]) -> np.ndarray:
        # Pad batch
        max_len = max(len(seq) for seq in batch)
        padded_batch = []
        for seq in batch:
            padded = seq + [alphabet.pad_idx] * (max_len - len(seq))
            padded_batch.append(padded)
        
        tokens = torch.tensor(padded_batch, dtype=torch.int64, device=device)
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type if device.type == "cuda" else "cpu"):
                outputs = model(tokens)
                logits = outputs["logits"]
        
        # Return logits for the 4 nucleotides
        # shape: (batch, seq_len, 4)
        return logits[:, 1:-1, sequence_tokens].cpu().numpy()

    for seq_id, seq in sequences:
        print(f"Processing {seq_id} (length {len(seq)})...")
        # Ensure sequence only contains ACGT (or U which is replaced)
        seq_clean = seq.upper().replace("U", "T")
        
        subset = None
        if args.subset_start is not None and args.subset_end is not None:
            subset = (args.subset_start, args.subset_end)
        
        options = DependencyMapOptions(subset=subset, with_reconstruction=True)
        try:
            dep_map = DependencyMap.compute_batched(
                seq_clean,
                tokenize_func,
                forward_func,
                batch_size=args.batch_size,
                options=options
            )
            
            safe_seq_id = "".join([c if c.isalnum() else "_" for c in seq_id])
            
            # Construct a descriptive filename suffix
            suffix_parts = [args.test_name]
            if subset is not None:
                suffix_parts.append(f"{subset[0]}_{subset[1]}")
            suffix = "_".join(suffix_parts)
            
            # Save raw numpy arrays just in case plotting fails or is too slow
            np.save(output_dir / f"{safe_seq_id}_{suffix}_dependency_map.npy", dep_map.dependency_map)
            if dep_map.reconstruction is not None:
                np.save(output_dir / f"{safe_seq_id}_{suffix}_reconstruction.npy", dep_map.reconstruction)
            
            fig = dep_map.plot()
            output_file = output_dir / f"{safe_seq_id}_{suffix}_dependency_map.svg"
            fig.write_image(str(output_file))
            print(f"Saved {output_file}")
            
        except Exception as e:
            print(f"Error processing {seq_id}: {e}")

if __name__ == "__main__":
    main()
