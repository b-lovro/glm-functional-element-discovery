import json
import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader, Dataset

from glmfe.seq_models.base import BaseSequenceModel
from glmfe.datasets.sequence_dataset import GenomicSequenceDataset
from glmfe.datasets.splitting import split_records

logger = logging.getLogger(__name__)

def collate_sequences(batch: list[str]) -> list[str]:
    """Identity collate function to pass lists of strings to the model."""
    return batch


def run_pretraining(
    records: pd.DataFrame,
    model: BaseSequenceModel,
    config: dict,
    run_dir: Path,
    prepared_dir: Path,
) -> None:
    """Main continued pretraining loop with LoRA and wandb logging."""
    
    train_config = config["training"]
    run_split = train_config["run_split"]
    
    # Prepare data
    splits = split_records(
        records=records,
        split_config=config["dataset"]["splits"],
        seed=train_config["seed"],
        prepared_dir=prepared_dir,
    )
    
    if run_split not in splits:
        raise ValueError(f"Unknown run_split: {run_split}. Available: {list(splits.keys())}")
        
    train_df = splits[run_split]
    
    if run_split == "train":
        val_df = splits.get("val")
    elif run_split == "overfit":
        val_df = splits.get("overfit")
    else:
        val_df = None
    
    def custom_collate(batch):
        if isinstance(batch[0], tuple):
            sequences = [item[0] for item in batch]
            is_start = [item[1] for item in batch]
            is_end = [item[2] for item in batch]
            return sequences, is_start, is_end
        return batch, None, None

    is_overfit = (run_split == "overfit")
    
    dataset = GenomicSequenceDataset(
        train_df, 
        max_length=model.max_context_length,
        deterministic=is_overfit,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_config["batch_size"],
        shuffle=not is_overfit, # Do not shuffle dataloader batches if overfitting
        drop_last=False,
        num_workers=train_config["num_workers"],
        pin_memory=True,
        collate_fn=custom_collate,
    )
    
    if val_df is not None:
        val_dataset = GenomicSequenceDataset(
            val_df, 
            max_length=model.max_context_length,
            deterministic=is_overfit,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=train_config["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=train_config["num_workers"],
            pin_memory=True,
            collate_fn=custom_collate,
        )
    else:
        val_dataloader = None
    
    # Prepare model for training
    model.prepare_for_training(train_config["lora"])
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    
    # GradScaler for Mixed Precision Stability
    scaler = torch.cuda.amp.GradScaler(enabled=(model.device.type == "cuda"))
    
    # Scheduler
    total_steps = len(dataloader) * train_config["epochs"]
    warmup_fraction = float(train_config["warmup_fraction"])
    num_warmup_steps = int(total_steps * warmup_fraction)
    scheduler_type = train_config["lr_scheduler"]
    
    if scheduler_type == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
        )
    elif scheduler_type == "linear":
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
        )
    else:
        raise ValueError(f"Unsupported lr_scheduler: {scheduler_type}. Use 'cosine' or 'linear'.")
    
    # Initialize wandb
    wandb_config = train_config["logging"]
    wandb.init(
        project=wandb_config["wandb_project"],
        name=config["run_id"],
        config=config,
        dir=str(run_dir),
    )
    
    epochs = train_config["epochs"]
    log_interval = wandb_config["log_interval"]
    
    global_step = 0
    best_loss = float('inf')
    start_epoch = 1
    
    resume = train_config["resume"]
    checkpoint_path = run_dir / "checkpoint_latest.pt"
    
    if resume and checkpoint_path.is_file():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=model.device)
        model.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["epoch"] * len(dataloader)
        
        best_loss = float('inf')
        if "best_loss" in checkpoint:
            best_loss = checkpoint["best_loss"]
        
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            print("Fast-forwarding scheduler to match global step...")
            for _ in range(global_step):
                scheduler.step()
                
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            
    for epoch in range(start_epoch, epochs + 1):
        epoch_loss = 0.0
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                batch_sequences, batch_is_start, batch_is_end = batch
            else:
                batch_sequences = batch
                batch_is_start, batch_is_end = None, None

            optimizer.zero_grad()
            
            # Forward pass and loss calculation
            loss = model.compute_pretraining_loss(
                batch_sequences, 
                is_start=batch_is_start, 
                is_end=batch_is_end,
                deterministic_mask=is_overfit,
            )
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Unscale gradients before clipping so the norm is correct
            scaler.unscale_(optimizer)
            
            # Gradient clipping and norm calculation
            max_grad_norm = train_config["max_grad_norm"]
            trainable_params = list(model.get_trainable_parameters())
            if max_grad_norm > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, max_grad_norm
                ).item()
            else:
                total_norm = 0.0
                for p in trainable_params:
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
            
            # Update weights safely with scaler
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            if global_step % log_interval == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/grad_norm": total_norm,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                })
                print(
                    f"Epoch {epoch}/{epochs} | Step {global_step} | "
                    f"Loss: {loss.item():.4f} | Grad Norm: {total_norm:.4f}"
                )
                
        avg_epoch_loss = epoch_loss / len(dataloader)
        wandb.log({"train/epoch_loss": avg_epoch_loss, "epoch": epoch}, step=global_step)
        print(f"Epoch {epoch} Complete. Average Training Loss: {avg_epoch_loss:.4f}")
        
        # Validation Loop
        avg_val_loss = None
        if val_dataloader is not None:
            model.model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    if isinstance(batch, (list, tuple)) and len(batch) == 3:
                        batch_sequences, batch_is_start, batch_is_end = batch
                    else:
                        batch_sequences = batch
                        batch_is_start, batch_is_end = None, None
                    
                    with torch.amp.autocast(device_type=model.device.type, dtype=torch.bfloat16):
                        loss = model.compute_pretraining_loss(
                            batch_sequences, 
                            is_start=batch_is_start, 
                            is_end=batch_is_end,
                            deterministic_mask=is_overfit,
                        )
                    val_loss_sum += loss.item()
            
            avg_val_loss = val_loss_sum / len(val_dataloader)
            val_perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
            wandb.log({
                "val/epoch_loss": avg_val_loss, 
                "val/perplexity": val_perplexity,
                "epoch": epoch
            }, step=global_step)
            print(f"Epoch {epoch} Validation Loss: {avg_val_loss:.4f} | Validation Perplexity: {val_perplexity:.4f}")
            model.model.train()
        
        # Track and save the best model
        eval_loss = avg_val_loss if avg_val_loss is not None else avg_epoch_loss
        if eval_loss < best_loss:
            best_loss = eval_loss
            is_best = True
        else:
            is_best = False
            
        save_interval_epochs = train_config["save_interval_epochs"]
        
        # Proper Checkpointing Strategy
        if epoch % save_interval_epochs == 0 or epoch == epochs or is_best:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "train_loss": avg_epoch_loss,
                "val_loss": avg_val_loss,
                "best_loss": best_loss,
            }
            
            if epoch % save_interval_epochs == 0 or epoch == epochs:
                torch.save(checkpoint, run_dir / "checkpoint_latest.pt")
                
            if is_best:
                torch.save(checkpoint, run_dir / "checkpoint_best.pt")
                print(f"New best checkpoint saved with loss: {best_loss:.4f}")
        
    wandb.finish()
    print("Pretraining completed successfully.")
