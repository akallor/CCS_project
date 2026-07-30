"""
ESM-2 Feature Extraction with Engineered Residue Features
==========================================================

UPDATED for the sequence-aware BiLSTM model.

Change vs the previous version
------------------------------
Adds an aggregation strategy "none" that returns the FULL per-residue ESM-2
embedding (L_i x D) for each peptide instead of collapsing it to a single
pooled vector. This is the path required by UnifiedESMCCSPredictor, whose
forward() reads per-residue embeddings with a genuine bidirectional LSTM.

- aggregation_strategy="none"  -> per-residue (list of (L_i, D) tensors) + lengths
- aggregation_strategy="global_mean"/"global_max"/"attention_weighted"
                               -> single pooled (D,) vector per peptide (old path)

The saved .pt now always includes a "lengths" array. For "none" it is the true
residue count per peptide (the first dim of each per-residue tensor); for the
pooled strategies it is still recorded for reference.

Set SAVE_HALF=True to store the per-residue embeddings as float16 (halves disk
footprint; the loader/collate casts back to float32 for the CPU model).
"""

import torch
import torch.nn as nn
import csv
import numpy as np
import os
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import warnings
from collections import defaultdict

# Store per-residue embeddings as float16 on disk (cast to float32 at load time).
SAVE_HALF = False


class ESMFeatureExtractor:
    """ESM-2 feature extractor for peptide sequences."""

    def __init__(self,
                 model_type: str = "esm2_t6_8M_UR50D",
                 aggregation_strategy: str = "none",
                 batch_size: int = 20000):
        self.model_variant = model_type
        self.aggregation_method = aggregation_strategy
        self.processing_batch_size = batch_size

        self.available_models = {
            "esm2_t6_8M_UR50D": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t6_8M_UR50D.pt",
            "esm2_t12_35M_UR50D": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t12_35M_UR50D.pt",
            "esm2_t30_150M_UR50D": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt",
            "esm2_t33_650M_UR50D": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt",
            "esm2_t36_3B_UR50D": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt",
            "esm2_t48_15B_UR50D": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t48_15B_UR50D.pt"
        }

        # "none" added: keep per-residue embeddings for the BiLSTM.
        self.aggregation_strategies = [
            "none", "global_mean", "global_max", "attention_weighted"
        ]

        self.esm_model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extraction_stats = {"total_sequences": 0, "sequence_lengths": []}

    def load_esm_model(self) -> None:
        print(f"Loading ESM-2 model: {self.model_variant}")
        try:
            self.esm_model, self.tokenizer = torch.hub.load(
                "facebookresearch/esm:main", self.model_variant)
            self.esm_model.eval()
            self.esm_model.to(self.device)
            for param in self.esm_model.parameters():
                param.requires_grad = False
            print(f"Successfully loaded {self.model_variant} (frozen)")
            test_sequence = ("test_seq", "MKFLVNVALVFMVVYISYIY")
            batch_converter = self.tokenizer.get_batch_converter()
            _, _, batch_tokens = batch_converter([test_sequence])
            batch_tokens = batch_tokens.to(self.device)
            with torch.no_grad():
                test_output = self.esm_model(batch_tokens, repr_layers=[6], return_contacts=False)
                print(f"Model test successful - output shape: {test_output['representations'][6].shape}")
        except Exception as e:
            print(f"Error loading model {self.model_variant}: {e}")
            raise e

    def compute_engineered_features(self, sequence: str) -> np.ndarray:
        """Return [nK, nR, nH, nD, nE, length, net_basicity]."""
        seq_upper = sequence.upper()
        nK = seq_upper.count('K'); nR = seq_upper.count('R'); nH = seq_upper.count('H')
        nD = seq_upper.count('D'); nE = seq_upper.count('E')
        length = len(sequence)
        w = 0.5  # histidine weight
        net_basicity = nK + nR + w * nH - (nD + nE)
        return np.array([nK, nR, nH, nD, nE, length, net_basicity], dtype=np.float32)

    def read_sequence_data(self, file_path: str, sequence_column: int = 1,
                           delimiter: str = '\t', skip_header: bool = True) -> List[List[str]]:
        print(f"Reading sequence data from: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as input_file:
            csv_reader = csv.reader(input_file, delimiter=delimiter)
            raw_data = [row for row in csv_reader]
        if skip_header and raw_data:
            header = raw_data[0]
            print(f"Header row: {header}")
            if sequence_column < len(header):
                print(f"Sequence column name: '{header[sequence_column]}'")
            raw_data = raw_data[1:]
        print(f"Loaded {len(raw_data)} sequences")
        return raw_data

    def process_batch(self, batch_data: List[Tuple[str, str]]):
        if not batch_data:
            raise ValueError("Empty batch data provided")
        sequences = []
        for seq_id, sequence in batch_data:
            if not sequence or not sequence.strip():
                print(f"Warning: Empty sequence found for ID: {seq_id}")
                continue
            if not all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in sequence.upper()):
                print(f"Warning: Invalid amino acid characters in sequence {seq_id}: {sequence}")
                continue
            sequences.append((seq_id, sequence.strip().upper()))
            self.extraction_stats["sequence_lengths"].append(len(sequence.strip()))
        if not sequences:
            raise ValueError("No valid sequences found in batch")

        batch_converter = self.tokenizer.get_batch_converter()
        _, _, batch_tokens = batch_converter(sequences)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            layer_num = 6
            for tag, ln in (("t6", 6), ("t12", 12), ("t30", 30), ("t33", 33), ("t36", 36), ("t48", 48)):
                if tag in self.model_variant:
                    layer_num = ln
                    break
            model_output = self.esm_model(batch_tokens, repr_layers=[layer_num], return_contacts=False)
            if layer_num in model_output['representations']:
                token_embeddings = model_output["representations"][layer_num]
            else:
                token_embeddings = model_output["representations"][max(model_output['representations'].keys())]

        sequence_lengths = [len(seq) for _, seq in sequences]
        aggregated_features = self._aggregate_sequence_features(token_embeddings, batch_tokens, sequence_lengths)
        engineered_features = [self.compute_engineered_features(seq) for _, seq in sequences]
        return aggregated_features, engineered_features

    def _aggregate_sequence_features(self, token_embeddings, batch_tokens, sequence_lengths):
        if self.aggregation_method == "none":
            return self._no_pooling(token_embeddings, batch_tokens, sequence_lengths)
        elif self.aggregation_method == "global_mean":
            return self._global_mean_pooling(token_embeddings, batch_tokens, sequence_lengths)
        elif self.aggregation_method == "global_max":
            return self._global_max_pooling(token_embeddings, batch_tokens, sequence_lengths)
        elif self.aggregation_method == "attention_weighted":
            return self._attention_weighted_pooling(token_embeddings, batch_tokens, sequence_lengths)
        else:
            return self._global_mean_pooling(token_embeddings, batch_tokens, sequence_lengths)

    def _no_pooling(self, token_embeddings, batch_tokens, sequence_lengths) -> List[torch.Tensor]:
        """Per-residue embeddings (L_i, D) per peptide, real residues only (CLS/EOS excluded)."""
        per_residue = []
        for i, seq_len in enumerate(sequence_lengths):
            seq_tokens = token_embeddings[i, 1:seq_len + 1]  # (L_i, D)
            seq_tokens = seq_tokens.detach().cpu()
            if SAVE_HALF:
                seq_tokens = seq_tokens.half()
            per_residue.append(seq_tokens)
        return per_residue

    def _global_mean_pooling(self, token_embeddings, batch_tokens, sequence_lengths):
        pooled_features = []
        for i, seq_len in enumerate(sequence_lengths):
            attention_mask = (batch_tokens[i] != self.tokenizer.padding_idx).float()
            seq_tokens = token_embeddings[i, 1:seq_len + 1]
            seq_mask = attention_mask[1:seq_len + 1]
            if seq_mask.sum() > 0:
                masked = seq_tokens * seq_mask.unsqueeze(1)
                pooled = masked.sum(dim=0) / seq_mask.sum()
            else:
                pooled = seq_tokens.mean(dim=0)
            pooled_features.append(pooled.detach().cpu())
        return pooled_features

    def _global_max_pooling(self, token_embeddings, batch_tokens, sequence_lengths):
        pooled_features = []
        for i, seq_len in enumerate(sequence_lengths):
            seq_tokens = token_embeddings[i, 1:seq_len + 1]
            pooled_features.append(seq_tokens.max(dim=0)[0].detach().cpu())
        return pooled_features

    def _attention_weighted_pooling(self, token_embeddings, batch_tokens, sequence_lengths):
        pooled_features = []
        for i, seq_len in enumerate(sequence_lengths):
            seq_tokens = token_embeddings[i, 1:seq_len + 1]
            attn = torch.softmax(torch.sum(seq_tokens * seq_tokens, dim=1), dim=0)
            pooled = (seq_tokens * attn.unsqueeze(1)).sum(dim=0)
            pooled_features.append(pooled.detach().cpu())
        return pooled_features

    def extract_features_from_file(self, input_file_path: str, output_file_path: str,
                                   sequence_column: int = 1, delimiter: str = '\t',
                                   skip_header: bool = True) -> None:
        self.load_esm_model()
        raw_sequence_data = self.read_sequence_data(input_file_path, sequence_column, delimiter, skip_header)
        total_sequences = len(raw_sequence_data)
        num_batches = (total_sequences // self.processing_batch_size) + 1
        print(f"Processing {total_sequences} sequences in {num_batches} batches")
        print(f"Using aggregation strategy: {self.aggregation_method}")

        all_esm_features, all_engineered_features = [], []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.processing_batch_size
            end_idx = min((batch_idx + 1) * self.processing_batch_size, total_sequences)
            if start_idx >= total_sequences:
                break
            batch_data = raw_sequence_data[start_idx:end_idx]
            batch_sequences = [(f"peptide_{start_idx + i}", row[sequence_column])
                               for i, row in enumerate(batch_data)]
            print(f"Processing batch {batch_idx + 1}/{num_batches} (sequences {start_idx + 1}-{end_idx})")
            b_esm, b_eng = self.process_batch(batch_sequences)
            all_esm_features.extend(b_esm)
            all_engineered_features.extend(b_eng)

        # lengths: true residue count per peptide.
        if self.aggregation_method == "none":
            lengths = np.array([int(t.shape[0]) for t in all_esm_features], dtype=np.int64)
        else:
            lengths = np.array(self.extraction_stats["sequence_lengths"], dtype=np.int64)

        # Normalize engineered features across the dataset.
        normalization_params = None
        if all_engineered_features:
            engineered_array = np.array(all_engineered_features)
            mean = engineered_array.mean(axis=0)
            std = engineered_array.std(axis=0) + 1e-8
            normalized = (engineered_array - mean) / std
            normalization_params = {'mean': mean.tolist(), 'std': std.tolist()}
            all_engineered_features = [normalized[i] for i in range(len(normalized))]

        results = {
            'esm_features': all_esm_features,          # list of (L_i, D) if "none", else (D,) each
            'engineered_features': all_engineered_features,
            'lengths': lengths,                        # NEW: per-peptide residue count
            'aggregation': self.aggregation_method,    # NEW: records how features were produced
            'normalization_params': normalization_params,
        }
        torch.save(results, output_file_path)
        print(f"Saved feature vectors to: {output_file_path}")
        self._print_extraction_statistics()
        print("Feature extraction completed successfully!")
        print(f"Total sequences processed: {len(all_esm_features)}")
        if all_esm_features:
            print(f"ESM feature shape (first peptide): {tuple(all_esm_features[0].size())}")
            print(f"Engineered feature dimension: {len(all_engineered_features[0])}")

    def _print_extraction_statistics(self) -> None:
        print("\n" + "=" * 60)
        print("FEATURE EXTRACTION STATISTICS")
        print("=" * 60)
        lengths = self.extraction_stats['sequence_lengths']
        print(f"Total sequences processed: {len(lengths)}")
        if lengths:
            print(f"Sequence length - Min: {min(lengths)}, Max: {max(lengths)}, Mean: {np.mean(lengths):.2f}")
        print("=" * 60)


def main():
    input_data_path = '/hpc/shared/uu_immunopeptidomics/CCS_project_part1/revised_ccs_codes/ccs_mz_correlation_data2.tsv'
    output_features_path = '/hpc/shared/uu_immunopeptidomics/features_hla1.pt'
    selected_model = "esm2_t6_8M_UR50D"
    selected_strategy = "none"   # per-residue for the BiLSTM
    batch_processing_size = 20000

    print("=" * 80)
    print("ESM-2 FEATURE EXTRACTION WITH ENGINEERED FEATURES")
    print(f"Model: {selected_model} | Aggregation: {selected_strategy} | Batch: {batch_processing_size}")
    print("=" * 80)

    extractor = ESMFeatureExtractor(model_type=selected_model,
                                    aggregation_strategy=selected_strategy,
                                    batch_size=batch_processing_size)
    extractor.extract_features_from_file(
        input_file_path=input_data_path, output_file_path=output_features_path,
        sequence_column=1, delimiter='\t', skip_header=True)


if __name__ == "__main__":
    main()
