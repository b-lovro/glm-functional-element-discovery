import torch
import pandas as pd
from torch.utils.data import Dataset


class GenomicSequenceDataset(Dataset):
    """PyTorch dataset for complete genomic sequences."""

    def __init__(
        self, 
        records_df: pd.DataFrame, 
        max_length: int = 1022,
        deterministic: bool = False,
    ):
        if "sequence" not in records_df.columns:
            raise ValueError("records dataframe must contain a 'sequence' column")
        self.sequences = records_df["sequence"].tolist()
        self.max_length = max_length
        self.deterministic = deterministic

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[str, bool, bool]:
        seq = self.sequences[idx]
        if len(seq) <= self.max_length:
            return seq, True, True
            
        if self.deterministic:
            g = torch.Generator()
            g.manual_seed(42 + idx)
            start = torch.randint(0, len(seq) - self.max_length + 1, (1,), generator=g).item()
        else:
            start = torch.randint(0, len(seq) - self.max_length + 1, (1,)).item()
            
        seq_crop = seq[start:start + self.max_length].upper()
            
        is_start = (start == 0)
        is_end = (start + self.max_length == len(seq))
        return seq_crop, is_start, is_end
