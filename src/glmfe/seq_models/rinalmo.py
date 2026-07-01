from pathlib import Path

import numpy as np
import torch
from rinalmo.config import model_config
from rinalmo.data.alphabet import Alphabet
from rinalmo.model.model import RiNALMo

from glmfe.seq_models.base import BaseSequenceModel


class RiNALMoSequenceModel(BaseSequenceModel):
    def __init__(
        self,
        model: RiNALMo,
        alphabet: Alphabet,
        model_size: str,
        device: torch.device,
    ):
        self.model = model
        self.alphabet = alphabet
        self.device = device
        self.model_id = f"rinalmo-{model_size}"
        self.reconstruction_protocol = "masked_single_base"
        self.max_context_length = 1022
        valid_bases = "ACGTNRYKMSWBDHVI-"
        self.nucleotide_token_indices = torch.tensor(
            [alphabet.tkn_to_idx[base] for base in valid_bases if base in alphabet.tkn_to_idx],
            dtype=torch.int64,
            device=device,
        )
        self.random_replace_indices = torch.tensor(
            [alphabet.tkn_to_idx[base] for base in "ACGTN" if base in alphabet.tkn_to_idx],
            dtype=torch.int64,
            device=device,
        )

    def predict_masked_base_probabilities(
        self,
        sequences: list[str],
        target_positions: list[int],
        batch_size: int,
    ) -> np.ndarray:
        probabilities = []
        for batch_start in range(0, len(sequences), batch_size):
            batch_sequences = sequences[batch_start : batch_start + batch_size]
            batch_positions = target_positions[batch_start : batch_start + batch_size]
            for sequence, target_position in zip(
                batch_sequences,
                batch_positions,
                strict=True,
            ):
                if len(sequence) > self.max_context_length:
                    raise ValueError(
                        f"Context length {len(sequence)} exceeds "
                        f"{self.max_context_length}"
                    )
                if set(sequence) - set("ACGTNRYKMSWBDHVI-"):
                    raise ValueError("RiNALMo contexts contain unsupported characters")
                if not 0 <= target_position < len(sequence):
                    raise ValueError(
                        f"Invalid context target position {target_position} "
                        f"for context length {len(sequence)}"
                    )

            tokens = torch.tensor(
                self.alphabet.batch_tokenize(batch_sequences),
                dtype=torch.int64,
                device=self.device,
            )
            token_positions = torch.tensor(
                batch_positions,
                dtype=torch.int64,
                device=self.device,
            ) + 1
            batch_indices = torch.arange(
                len(batch_sequences),
                dtype=torch.int64,
                device=self.device,
            )
            tokens[batch_indices, token_positions] = self.alphabet.mask_idx

            with torch.no_grad():
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    logits = self.model(tokens)["logits"]
            masked_logits = logits[
                batch_indices,
                token_positions,
            ][:, self.nucleotide_token_indices]
            batch_probabilities = torch.softmax(
                masked_logits.float(),
                dim=-1,
            )
            probabilities.append(batch_probabilities.cpu().numpy())

        return np.concatenate(probabilities, axis=0)

    def dependency_tokenize(
        self,
        sequence: str,
        mask_position: int | None,
    ) -> object:
        sequence = sequence.replace("U", "T")
        if len(sequence) > self.max_context_length:
            raise ValueError(
                f"Context length {len(sequence)} exceeds "
                f"{self.max_context_length}"
            )
        if set(sequence) - set("ACGTNRYKMSWBDHVI-"):
            raise ValueError("RiNALMo contexts contain unsupported characters")
        if mask_position is not None and not 0 <= mask_position < len(sequence):
            raise ValueError(
                f"Invalid mask position {mask_position} "
                f"for sequence length {len(sequence)}"
            )

        tokens = self.alphabet.batch_tokenize([sequence])[0]
        if mask_position is not None:
            tokens[mask_position + 1] = self.alphabet.mask_idx
        return tokens

    def dependency_forward(
        self,
        tokenized_sequences: list[object],
        batch_size: int,
    ) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not tokenized_sequences:
            return np.empty((0, 0, 4), dtype=np.float32)
        if not all(
            isinstance(tokens, list)
            for tokens in tokenized_sequences
        ):
            raise TypeError("RiNALMo dependency inputs must be token lists")

        token_lengths = {
            len(tokens)
            for tokens in tokenized_sequences
            if isinstance(tokens, list)
        }
        if len(token_lengths) != 1:
            raise ValueError(
                "RiNALMo dependency inputs must have equal token lengths"
            )
        sequence_length = token_lengths.pop() - 2

        output_logits = []
        for batch_start in range(0, len(tokenized_sequences), batch_size):
            batch_tokens = tokenized_sequences[
                batch_start : batch_start + batch_size
            ]
            tokens = torch.tensor(
                batch_tokens,
                dtype=torch.int64,
                device=self.device,
            )
            with torch.inference_mode():
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    logits = self.model(tokens)["logits"]
            nucleotide_logits = logits[:, 1:-1, :][
                :, :, self.nucleotide_token_indices
            ]
            output_logits.append(
                nucleotide_logits.float().cpu().numpy()
            )

        result = np.concatenate(output_logits, axis=0)
        expected_shape = (
            len(tokenized_sequences),
            sequence_length,
            4,
        )
        if result.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected RiNALMo dependency logits shape "
                f"{result.shape}; expected {expected_shape}"
            )
        return result

    def prepare_for_training(self, lora_config: dict) -> None:
        import torch.nn as nn
        from peft import LoraConfig, get_peft_model
        
        target_modules = lora_config["target_modules"]
        if target_modules == "all-linear" or target_modules == ["all-linear"]:
            target_modules = set()
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    target_modules.add(name)
            target_modules = list(target_modules)
            
        peft_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["alpha"],
            target_modules=target_modules,
            lora_dropout=lora_config["dropout"],
            bias="none",
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.train()
        
        trainable_params = sum(p.numel() for p in self.get_trainable_parameters())
        print(f"LoRA injected: {trainable_params / 1e6:.2f} M trainable parameters")

    def get_trainable_parameters(self) -> filter:
        return filter(lambda p: p.requires_grad, self.model.parameters())

    def compute_pretraining_loss(
        self, 
        sequences: list[str],
        is_start: list[bool] | None = None,
        is_end: list[bool] | None = None,
        deterministic_mask: bool = False,
    ) -> object:
        import torch.nn.functional as F

        sanitized_sequences = []
        for sequence in sequences:
            if len(sequence) > self.max_context_length:
                raise ValueError(
                    f"Context length {len(sequence)} exceeds "
                    f"{self.max_context_length}"
                )
            seq = sequence.upper().replace('U', 'T')
            sanitized_sequences.append(seq)
        sequences = sanitized_sequences
                
        if is_start is not None and hasattr(is_start, "tolist"):
            is_start = is_start.tolist()
        if is_end is not None and hasattr(is_end, "tolist"):
            is_end = is_end.tolist()
            
        if is_start is None:
            is_start = [True] * len(sequences)
        if is_end is None:
            is_end = [True] * len(sequences)

        batch_tokens = []
        max_len = 0
        for seq, start_flag, end_flag in zip(sequences, is_start, is_end, strict=True):
            tokens = self.alphabet.batch_tokenize([seq])[0]
            if not start_flag:
                tokens = tokens[1:]
            if not end_flag:
                tokens = tokens[:-1]
            batch_tokens.append(tokens)
            if len(tokens) > max_len:
                max_len = len(tokens)
                
        pad_idx = self.alphabet.pad_idx
        padded_tokens = []
        for t in batch_tokens:
            padded_tokens.append(t + [pad_idx] * (max_len - len(t)))

        # tokens shape: (Batch, SeqLen)
        tokens = torch.tensor(
            padded_tokens,
            dtype=torch.int64,
            device=self.device,
        )
        
        labels = tokens.clone()
        
        # Identify valid nucleotide positions
        is_nucleotide = torch.isin(tokens, self.nucleotide_token_indices)
        
        prob_matrix = torch.full(tokens.shape, 0.0, device=self.device)
        prob_matrix[is_nucleotide] = 0.15
        
        if deterministic_mask:
            g = torch.Generator(device=self.device)
            g.manual_seed(42)
            masked_indices = torch.bernoulli(prob_matrix, generator=g).bool()
            rand = torch.rand(tokens.shape, generator=g, device=self.device)
        else:
            masked_indices = torch.bernoulli(prob_matrix).bool()
            rand = torch.rand(tokens.shape, device=self.device)
        
        # 80% of 15% -> [MASK] token
        replace_mask = masked_indices & (rand < 0.8)
        tokens[replace_mask] = self.alphabet.mask_idx
        
        # 10% of 15% -> Random nucleotide
        replace_random = masked_indices & (rand >= 0.8) & (rand < 0.9)
        if replace_random.any():
            random_nucleotides = self.random_replace_indices[
                torch.randint(
                    0, len(self.random_replace_indices), 
                    (replace_random.sum().item(),), 
                    device=self.device
                )
            ]
            tokens[replace_random] = random_nucleotides
            
        # Remaining 10% is left intact
        
        # Only compute loss on masked_indices
        labels[~masked_indices] = -100
        
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            # logits shape: (Batch, SeqLen, Vocab)
            logits = self.model(tokens)["logits"]
            
        # loss is scalar
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            labels.view(-1), 
            ignore_index=-100
        )
        
        return loss



def load_rinalmo_model(
    model_size: str,
    weights_path: Path,
    device: str,
    lora_weights_path: Path | None = None,
) -> RiNALMoSequenceModel:
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing RiNALMo weights: {weights_path}")

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for configured device {device}")

    config = model_config(model_size)
    alphabet = Alphabet(**config["alphabet"])
    model = RiNALMo(config)
    state_dict = torch.load(weights_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError("RiNALMo checkpoint must contain a direct state dictionary")
    model.load_state_dict(state_dict)
    
    if lora_weights_path is not None:
        if not lora_weights_path.is_file():
            raise FileNotFoundError(f"Missing LoRA weights: {lora_weights_path}")
            
        import json
        run_dir = lora_weights_path.parent
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing manifest.json for LoRA weights in {run_dir}")
            
        with manifest_path.open() as f:
            manifest = json.load(f)
            
        if "training" not in manifest or "lora" not in manifest["training"]:
            raise ValueError(f"LoRA config not found in manifest at {manifest_path}")
            
        lora_config = manifest["training"]["lora"]
        
        import torch.nn as nn
        from peft import LoraConfig, get_peft_model
        
        target_modules = lora_config["target_modules"]
        if target_modules == "all-linear" or target_modules == ["all-linear"]:
            target_modules = set()
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    target_modules.add(name)
            target_modules = list(target_modules)
            
        peft_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["alpha"],
            target_modules=target_modules,
            lora_dropout=lora_config["dropout"],
            bias="none",
        )
        
        model = get_peft_model(model, peft_config)
        
        lora_checkpoint = torch.load(lora_weights_path, map_location="cpu")
        if "model_state_dict" not in lora_checkpoint:
            raise ValueError(f"LoRA checkpoint {lora_weights_path} missing 'model_state_dict'")
            
        model.load_state_dict(lora_checkpoint["model_state_dict"])
        model = model.merge_and_unload()

    model = model.to(torch_device)
    model.eval()
    return RiNALMoSequenceModel(
        model,
        alphabet,
        model_size,
        torch_device,
    )
