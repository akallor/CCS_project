#!/usr/bin/env python3
"""
Unified ESM-2 + BiLSTM CCS Training and Evaluation (per-residue / sequence-aware)
================================================================================

Updated from the pooled version to feed the BiLSTM model.

Changes vs the pooled trainer
------------------------------
- Dataset now holds a LIST of per-residue ESM tensors (L_i, D) + a lengths vector,
  not a single stacked (N, D) pooled tensor.
- New pad_collate() pads each batch to its max length and returns lengths.
- Model is called as model(esm_tokens=..., lengths=..., engineered=..., charge=...)
  everywhere (train_epoch, evaluate, test loop) - matching UnifiedESMCCSPredictor.
- load_features_and_tsv() keeps ESM as a list, reads "lengths" from the .pt
  (falling back to per-tensor shape), and does list-aware index/length filtering.
- esm_dim is inferred from the per-residue feature dimension (last axis).

Everything else (AdamW + warmup-cosine, weight decay, grad clip, early stopping,
train-only z-normalization, charge-stratified 80/10/10 split, metrics/plots) is
unchanged.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import torch.multiprocessing as _torch_mp
    _torch_mp.set_sharing_strategy("file_system")
except Exception:
    pass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from unified_feature_embedding import stratified_split_indices_by_charge
from unified_esm_rnn_model import UnifiedESMCCSPredictor, ENGINEERED_FEAT_DIM


def get_lr_with_warmup_cosine(step, warmup_steps, total_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress))


class UnifiedESMTrainer:
    def __init__(self, model, learning_rate=1e-3, weight_decay=1e-5, warmup_ratio=0.1,
                 gradient_clip=1.0, target_mean=0.0, target_std=1.0, device="auto", use_amp=False):
        self.model = model
        self.device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
                       if device == "auto" else torch.device(device))
        self.model.to(self.device)
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.base_lr = learning_rate
        self.warmup_ratio = warmup_ratio
        self.gradient_clip = gradient_clip
        self.target_mean = target_mean
        self.target_std = target_std
        self.history = {"train_loss": [], "val_loss": [], "train_r2": [], "val_r2": [], "lr": []}

    def train_epoch(self, train_loader, criterion, epoch, total_epochs):
        self.model.train()
        total_steps = total_epochs * len(train_loader)
        warmup_steps = int(total_steps * self.warmup_ratio)
        non_blocking = self.device.type == "cuda"

        for batch_idx, (esm, lengths, eng, charge, target) in enumerate(train_loader):
            esm = esm.to(self.device, non_blocking=non_blocking)
            lengths = lengths.to(self.device, non_blocking=non_blocking)
            eng = eng.to(self.device, non_blocking=non_blocking)
            charge = charge.to(self.device, non_blocking=non_blocking)
            target = target.to(self.device, non_blocking=non_blocking)

            step = epoch * len(train_loader) + batch_idx
            lr = get_lr_with_warmup_cosine(step, warmup_steps, total_steps, self.base_lr)
            for g in self.optimizer.param_groups:
                g["lr"] = lr

            self.optimizer.zero_grad()
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    pred = self.model(esm_tokens=esm, lengths=lengths, engineered=eng, charge=charge).squeeze(-1)
                    loss = criterion(pred, target)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(esm_tokens=esm, lengths=lengths, engineered=eng, charge=charge).squeeze(-1)
                loss = criterion(pred, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, data_loader, criterion):
        self.model.eval()
        total_loss = total_r2 = 0.0
        n_batches = 0
        non_blocking = self.device.type == "cuda"
        for esm, lengths, eng, charge, target in data_loader:
            esm = esm.to(self.device, non_blocking=non_blocking)
            lengths = lengths.to(self.device, non_blocking=non_blocking)
            eng = eng.to(self.device, non_blocking=non_blocking)
            charge = charge.to(self.device, non_blocking=non_blocking)
            target = target.to(self.device, non_blocking=non_blocking)
            pred = self.model(esm_tokens=esm, lengths=lengths, engineered=eng, charge=charge).squeeze(-1)
            loss = criterion(pred, target)
            pred_raw = pred * self.target_std + self.target_mean
            target_raw = target * self.target_std + self.target_mean
            total_loss += loss.item()
            total_r2 += _r2_torch(pred_raw, target_raw)
            n_batches += 1
        return total_loss / max(n_batches, 1), total_r2 / max(n_batches, 1)

    def train(self, train_loader, val_loader, num_epochs=100, early_stopping_patience=15,
              save_path=None, loss_type="mse", val_every=1):
        criterion = nn.MSELoss() if loss_type == "mse" else nn.HuberLoss()
        best_val_loss = float("inf")
        patience = 0
        for epoch in range(num_epochs):
            self.train_epoch(train_loader, criterion, epoch, num_epochs)
            train_loss, train_r2 = self.evaluate(train_loader, criterion)
            do_val = (epoch % val_every == 0) or (epoch == num_epochs - 1)
            if do_val:
                val_loss, val_r2 = self.evaluate(val_loader, criterion)
            else:
                val_loss = self.history["val_loss"][-1] if self.history["val_loss"] else float("inf")
                val_r2 = self.history["val_r2"][-1] if self.history["val_r2"] else 0.0
            lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_r2"].append(train_r2)
            self.history["val_r2"].append(val_r2)
            self.history["lr"].append(lr)
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                      f"train_r2={train_r2:.4f}  val_r2={val_r2:.4f}  lr={lr:.2e}")
            if do_val:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
                else:
                    patience += 1
                    if patience >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
        return self.history


def _r2_torch(pred, target):
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return (1 - ss_res / (ss_tot + 1e-8)).item()


def charge_stratified_metrics(preds, targets, charges):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    out = {}
    for ch in np.unique(charges):
        mask = charges == ch
        if mask.sum() < 2:
            continue
        p, t = preds[mask], targets[mask]
        out[int(ch)] = {"rmse": float(np.sqrt(mean_squared_error(t, p))),
                        "mae": float(mean_absolute_error(t, p)),
                        "r2": float(r2_score(t, p)), "n": int(mask.sum())}
    return out


def _save_fig_multiformat(fig, basepath, dpi=300):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{basepath}.{ext}", dpi=dpi if ext == "png" else None, bbox_inches="tight")


# (plot_charge_stratified is unchanged from the pooled trainer - omitted here for brevity;
#  paste your existing function back in, it needs no edits.)


# ---------------------------------------------------------------------------
# Dataset, collate and data loading (per-residue)
# ---------------------------------------------------------------------------


class UnifiedESMDataset(torch.utils.data.Dataset):
    """Dataset of (per_residue_esm (L_i, D), length_i, engineered, charge_0based, ccs_normalized)."""

    def __init__(self, esm_list, engineered_tensor, lengths, charges, targets, target_mean, target_std):
        assert len(esm_list) == len(engineered_tensor) == len(lengths) == len(charges) == len(targets)
        self.esm = esm_list  # list of (L_i, D) tensors
        self.engineered = engineered_tensor
        self.lengths = torch.as_tensor(lengths, dtype=torch.long)
        self.charges = torch.as_tensor(charges, dtype=torch.long)
        self.targets_raw = torch.as_tensor(targets, dtype=torch.float32)
        self.target_mean = target_mean
        self.target_std = target_std
        self.targets = (self.targets_raw - target_mean) / target_std

    def __len__(self):
        return len(self.esm)

    def __getitem__(self, idx):
        return (self.esm[idx].float(),        # cast (handles float16-on-disk)
                self.lengths[idx],
                self.engineered[idx],
                self.charges[idx],
                self.targets[idx])


def pad_collate(batch):
    """Pad variable-length per-residue ESM tensors to the batch max length."""
    esm_list, lengths, eng, charge, target = zip(*batch)
    lengths = torch.stack([torch.as_tensor(l, dtype=torch.long) for l in lengths])
    max_len = int(lengths.max().item())
    dim = esm_list[0].shape[-1]
    padded = esm_list[0].new_zeros((len(esm_list), max_len, dim))
    for i, t in enumerate(esm_list):
        padded[i, : t.shape[0]] = t
    eng = torch.stack([torch.as_tensor(e, dtype=torch.float32) for e in eng])
    charge = torch.stack([torch.as_tensor(c, dtype=torch.long) for c in charge])
    target = torch.stack([torch.as_tensor(y, dtype=torch.float32) for y in target])
    return padded, lengths, eng, charge, target


def load_features_and_tsv(features_path, data_path, sequence_column="Peptide",
                          charge_column="Charge", target_column="CCS_Experimental",
                          min_length=0, max_length=0):
    data = torch.load(features_path, weights_only=False)
    esm_features = data["esm_features"]           # list of (L_i, D)
    engineered_features = data["engineered_features"]
    saved_lengths = data.get("lengths", None)

    if not isinstance(esm_features, list):
        raise ValueError(
            "Expected per-residue features (a list of (L_i, D) tensors). "
            "Re-extract with aggregation_strategy='none' for the BiLSTM model."
        )
    esm_list = [f.cpu() if hasattr(f, "is_cuda") and f.is_cuda else f for f in esm_features]

    if isinstance(engineered_features, list):
        eng_tensor = torch.tensor(np.array(engineered_features), dtype=torch.float32)
    else:
        eng_tensor = torch.tensor(np.asarray(engineered_features), dtype=torch.float32)

    df = pd.read_csv(data_path, sep="\t")
    if charge_column not in df.columns or target_column not in df.columns:
        raise ValueError(f"TSV must contain {charge_column} and {target_column}. Got {list(df.columns)}")

    selected_indices = data.get("selected_indices", None)
    if selected_indices is not None:
        selected_indices = np.asarray(selected_indices, dtype=np.int64)
        if selected_indices.size > 0:
            df = df.iloc[selected_indices].reset_index(drop=True)

    charges_raw = np.asarray(df[charge_column].values, dtype=np.int64)
    targets_raw = np.asarray(df[target_column].values, dtype=np.float64)

    n_feat, n_tsv = len(esm_list), len(df)
    if n_feat != n_tsv:
        print(f"Aligning by index: features n={n_feat}, TSV n={n_tsv}; using min={min(n_feat, n_tsv)}")
    n = min(n_feat, n_tsv)
    esm_list = esm_list[:n]
    eng_tensor = eng_tensor[:n]
    charges = np.clip(charges_raw[:n].astype(np.int64) - 1, 0, None)
    targets = targets_raw[:n]
    if saved_lengths is not None:
        lengths = np.asarray(saved_lengths, dtype=np.int64)[:n]
    else:
        lengths = np.array([int(t.shape[0]) for t in esm_list], dtype=np.int64)

    if min_length > 0 or max_length > 0:
        if sequence_column not in df.columns:
            raise ValueError(f"Length filtering requires '{sequence_column}' in TSV. Got {list(df.columns)}")
        seqs = df[sequence_column].astype(str).str.strip().to_numpy()[:n]
        seq_lengths = np.array([len(s) for s in seqs], dtype=np.int64)
        keep = np.ones(n, dtype=bool)
        if min_length > 0:
            keep &= seq_lengths >= min_length
        if max_length > 0:
            keep &= seq_lengths <= max_length
        kept = int(keep.sum())
        if kept == 0:
            raise RuntimeError(f"No samples remain after length filtering min={min_length}, max={max_length}")
        keep_idx = np.where(keep)[0]
        esm_list = [esm_list[i] for i in keep_idx]       # list-aware filtering
        eng_tensor = eng_tensor[torch.from_numpy(keep_idx)]
        charges = charges[keep_idx]
        targets = targets[keep_idx]
        lengths = lengths[keep_idx]
        print(f"Length filter applied: min={min_length}, max={max_length}; kept {kept}/{n} aligned samples")

    return esm_list, eng_tensor, lengths, charges, targets


def main():
    parser = argparse.ArgumentParser(description="Unified ESM-2 + BiLSTM CCS training (per-residue)")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--features_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./unified_esm_results_bilstm")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "huber"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--rnn_type", type=str, default="lstm", choices=["lstm", "gru", "rnn"])
    parser.add_argument("--esm_dim", type=int, default=None)
    parser.add_argument("--min_length", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.data_path):
        print(f"Data not found: {args.data_path}"); return
    if not os.path.exists(args.features_path):
        print(f"Features not found: {args.features_path}. Run ESM feature extraction (aggregation='none') first."); return

    esm_list, eng_tensor, lengths, charges, targets = load_features_and_tsv(
        args.features_path, args.data_path, min_length=args.min_length, max_length=args.max_length)
    n_samples = len(esm_list)
    esm_dim = esm_list[0].shape[-1]        # per-residue feature dim
    eng_dim = eng_tensor.shape[1]
    if args.esm_dim is not None and args.esm_dim != esm_dim:
        print(f"Warning: --esm_dim {args.esm_dim} != inferred {esm_dim}; using {esm_dim}")

    train_idx, val_idx, test_idx = stratified_split_indices_by_charge(
        charges, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        ch_arr = charges[np.asarray(idx)]
        uniq, cnt = np.unique(ch_arr, return_counts=True)
        print(f"  {split_name}: " + " ".join([f"z={int(u)+1}:{c}" for u, c in zip(uniq, cnt)]))

    train_targets = targets[train_idx]
    target_mean = float(np.mean(train_targets))
    target_std = float(np.std(train_targets)) or 1.0
    if target_std < 1e-8:
        target_std = 1.0

    full_dataset = UnifiedESMDataset(esm_list, eng_tensor, lengths, charges, targets,
                                     target_mean=target_mean, target_std=target_std)
    train_set = torch.utils.data.Subset(full_dataset, train_idx)
    val_set = torch.utils.data.Subset(full_dataset, val_idx)
    test_set = torch.utils.data.Subset(full_dataset, test_idx)

    use_cuda = torch.cuda.is_available()
    loader_kw = {"batch_size": args.batch_size, "num_workers": args.num_workers,
                 "pin_memory": use_cuda and args.num_workers > 0,
                 "persistent_workers": bool(args.persistent_workers and args.num_workers > 0),
                 "collate_fn": pad_collate}
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_kw)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_kw)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, **loader_kw)

    num_charges = max(10, int(charges.max()) + 1)
    model = UnifiedESMCCSPredictor(esm_dim=esm_dim, engineered_feat_dim=eng_dim,
                                   num_charges=num_charges, charge_embed_dim=32,
                                   rnn_input_dim=256, rnn_hidden_dim=128, num_layers=2,
                                   rnn_type=args.rnn_type, dropout=0.3, output_dim=1)

    trainer = UnifiedESMTrainer(model, learning_rate=args.lr, weight_decay=1e-5,
                                warmup_ratio=0.1, gradient_clip=1.0,
                                target_mean=target_mean, target_std=target_std, use_amp=args.amp)

    save_path = os.path.join(args.output_dir, "best_unified_esm_model.pt")
    history = trainer.train(train_loader, val_loader, num_epochs=args.epochs,
                            early_stopping_patience=args.patience, save_path=save_path,
                            loss_type=args.loss, val_every=args.val_every)

    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump({**history, "min_length": args.min_length, "max_length": args.max_length}, f, indent=2)

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location="cpu"))
    device = trainer.device
    model.to(device)

    model.eval()
    preds_list, targets_list, charges_list = [], [], []
    non_blocking = device.type == "cuda"
    with torch.no_grad():
        for esm, lengths_b, eng, charge, target in test_loader:
            esm = esm.to(device, non_blocking=non_blocking)
            lengths_b = lengths_b.to(device, non_blocking=non_blocking)
            eng = eng.to(device, non_blocking=non_blocking)
            charge = charge.to(device, non_blocking=non_blocking)
            out = model(esm_tokens=esm, lengths=lengths_b, engineered=eng, charge=charge).squeeze(-1)
            preds_list.append(out.cpu().numpy() * target_std + target_mean)
            targets_list.append(target.cpu().numpy() * target_std + target_mean)
            charges_list.append(charge.cpu().numpy())

    preds = np.concatenate(preds_list)
    targets_test = np.concatenate(targets_list)
    charges_test = np.concatenate(charges_list)

    per_ch = charge_stratified_metrics(preds, targets_test, charges_test)
    per_ch_display = {str(int(k) + 1): v for k, v in per_ch.items()}
    with open(os.path.join(args.output_dir, "charge_stratified_metrics.json"), "w") as f:
        json.dump(per_ch_display, f, indent=2)

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    print("\nUnified ESM BiLSTM - test set:")
    print(f"  RMSE = {np.sqrt(mean_squared_error(targets_test, preds)):.4f}")
    print(f"  MAE  = {mean_absolute_error(targets_test, preds):.4f}")
    print(f"  R2   = {r2_score(targets_test, preds):.4f}")
    print("Per charge:", per_ch_display)
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
