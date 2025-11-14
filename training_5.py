"""
training_6.py
Step 6: Training (CPU-friendly test mode + production knobs)

Changes from previous:
 - Adds quick-debug subsampling (MAX_SAMPLES) for fast CPU runs (so tqdm shows progress).
 - Ensures graphs and targets are moved to `device` per sample.
 - Uses batch_size=1 by default on CPU to avoid batching complexities with graph objects.
 - Clear commented "accuracy knobs" with approximate effect estimates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime

# -------------------- Import model --------------------
from model_architecture_4 import WildfirePredictionModel, create_model

# -------------------- Dataset --------------------
class WildfireDataset(Dataset):
    """Dataset class for wildfire graph sequences.
       Each item is a tuple: (sequence_of_graphs, target_dict)
       NOTE: sequence_of_graphs is a list of torch_geometric.data.Data objects
    """
    def __init__(self, graph_sequences, targets):
        self.graph_sequences = graph_sequences
        self.targets = targets

    def __len__(self):
        return len(self.graph_sequences)

    def __getitem__(self, idx):
        return self.graph_sequences[idx], self.targets[idx]


# -------------------- Focal Loss --------------------
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# -------------------- Trainer --------------------
class WildfireTrainer:
    """Trainer class for wildfire prediction model."""

    def __init__(self, model, device=None):
        # Select device (user override allowed)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = model.to(self.device)
        print("Using device:", self.device)
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    # ---------- Data Preparation ----------
    def prepare_data(self, processed_data_path='processed_data.pkl',
                     temporal_graphs_path='temporal_graphs.pkl',
                     sequence_length=7, test_size=0.2, val_size=0.1,
                     debug_max_samples=None):
        """
        Loads preprocessed dataframe and temporal_graphs (list of graphs).

        debug_max_samples: if int, slice dataset to this many samples for fast CPU testing.
                           Recommended: 500..5000 (lower -> faster)
        """
        print("Preparing training data...")

        if not os.path.exists(processed_data_path) or not os.path.exists(temporal_graphs_path):
            raise FileNotFoundError("Missing required files. Run preprocessing and graph steps first.")

        with open(processed_data_path, 'rb') as f:
            processed_df = pickle.load(f)

        with open(temporal_graphs_path, 'rb') as f:
            temporal_graphs = pickle.load(f)

        sequences, targets = self._create_sequences(processed_df, temporal_graphs, sequence_length)

        # ---------------- DEBUG: quick CPU subset ----------------
        if debug_max_samples is not None and len(sequences) > debug_max_samples:
            print(f"⚠ Using only {debug_max_samples} samples for quick CPU testing (set debug_max_samples=None to disable)")
            sequences = sequences[:debug_max_samples]
            targets = targets[:debug_max_samples]
        # --------------------------------------------------------

        train_val_seq, test_seq, train_val_tgt, test_tgt = train_test_split(
            sequences, targets, test_size=test_size, random_state=42)

        train_seq, val_seq, train_tgt, val_tgt = train_test_split(
            train_val_seq, train_val_tgt, test_size=val_size / (1 - test_size), random_state=42)

        print(f"   Train: {len(train_seq)} | Val: {len(val_seq)} | Test: {len(test_seq)}")

        return train_seq, train_tgt, val_seq, val_tgt, test_seq, test_tgt

    def _create_sequences(self, processed_df, temporal_graphs, sequence_length):
        """
        Build sequences by grouping processed_df by location (lat,lon).
        For each location we take sliding windows of length `sequence_length` across time.
        Each sequence references the same temporal_graphs slices.
        """
        sequences, targets = [], []

        for (lat, lon), group in processed_df.groupby(['latitude', 'longitude']):
            group = group.sort_values('time_window').reset_index(drop=True)
            for i in range(len(group) - sequence_length):
                # use the same temporal_graph slices for all locations (as in graph_construction)
                seq_graphs = temporal_graphs[i:i + sequence_length]
                target_row = group.iloc[i + sequence_length]

                target = {
                    'fire_occurrence': torch.tensor(float(target_row.get('has_fire', 0)), dtype=torch.float32),
                    'fire_intensity': torch.tensor(float(target_row.get('frp_mean', 0)), dtype=torch.float32)
                }

                sequences.append(seq_graphs)
                targets.append(target)

        return sequences, targets

    # ---------- Training Loop ----------
    def _move_sequence_to_device(self, sequence):
        """
        Move a list of torch_geometric Data graphs to device.
        Returns a new list of graphs already on device.
        """
        moved = []
        for g in sequence:
            # Each g is expected to be a torch_geometric.data.Data
            # .to(device) moves tensors inside Data object (x, edge_index, edge_attr, etc.)
            moved.append(g.to(self.device))
        return moved

    def train_epoch(self, loader, optimizer, criterion_dict, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Train Epoch {epoch}", unit="sample")
        for batch in pbar:
            # Each batch is (sequence, target) because we keep batch_size=1 for safety on CPU
            sequence, target = batch  # sequence: list(graphs), target: dict of tensors

            # Move sequence graphs + target tensors to device
            sequence = self._move_sequence_to_device(sequence)
            target = {k: v.to(self.device) for k, v in target.items()}

            optimizer.zero_grad()
            preds = self.model(sequence)

            loss = 0.0
            # Fire occurrence loss
            if 'fire_occurrence' in preds:
                loss_occ = criterion_dict['occurrence'](
                    preds['fire_occurrence'].squeeze(), target['fire_occurrence'])
                loss = loss + loss_occ

            # Fire intensity loss (only where fire occurred)
            if 'fire_intensity' in preds:
                mask = target['fire_occurrence'] > 0.5
                if mask.sum() > 0:
                    loss_int = criterion_dict['intensity'](
                        preds['fire_intensity'].squeeze()[mask],
                        target['fire_intensity'][mask])
                    loss = loss + loss_int

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate(self, loader, criterion_dict):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        pbar = tqdm(loader, desc="Validation", unit="sample")
        with torch.no_grad():
            for batch in pbar:
                sequence, target = batch
                sequence = self._move_sequence_to_device(sequence)
                target = {k: v.to(self.device) for k, v in target.items()}

                preds = self.model(sequence)
                loss = 0.0

                if 'fire_occurrence' in preds:
                    loss += criterion_dict['occurrence'](
                        preds['fire_occurrence'].squeeze(),
                        target['fire_occurrence']
                    )

                if 'fire_intensity' in preds:
                    mask = target['fire_occurrence'] > 0.5
                    if mask.sum() > 0:
                        loss += criterion_dict['intensity'](
                            preds['fire_intensity'].squeeze()[mask],
                            target['fire_intensity'][mask]
                        )

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / num_batches if num_batches > 0 else 0.0

    def train(self, train_seq, train_tgt, val_seq, val_tgt,
              num_epochs=5, batch_size=1, lr=0.001, save_dir='checkpoints'):
        """
        Training entrypoint.

        Defaults are CPU-test friendly:
          - num_epochs=5 (faster)
          - batch_size=1 (safe with graph objects and node-wise batching)
        Change these for production (see "Accuracy knobs" comments further below).
        """
        print("\n" + "=" * 60)
        print("TRAINING MODEL")
        print("=" * 60)

        os.makedirs(save_dir, exist_ok=True)

        # Use batch_size=1 to keep batching simple for graph sequences on CPU.
        # If/when on GPU and you know graph shapes are identical per sequence, you can
        # increase batch_size for throughput (requires extra collation logic).
        train_loader = DataLoader(WildfireDataset(train_seq, train_tgt), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(WildfireDataset(val_seq, val_tgt), batch_size=batch_size, shuffle=False)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        criterion_dict = {'occurrence': FocalLoss(alpha=2, gamma=2), 'intensity': nn.MSELoss()}

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(train_loader, optimizer, criterion_dict, epoch)
            val_loss = self.validate(val_loader, criterion_dict)

            scheduler.step(val_loss)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
                print(f"   ✓ Saved best model (val_loss: {val_loss:.4f})")

            # save latest after each epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'train_losses': self.train_losses,
                'val_losses': self.val_losses
            }, os.path.join(save_dir, 'latest.pt'))

        print("\nTraining Complete!")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")

        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump({'train': self.train_losses, 'val': self.val_losses}, f, indent=2)


# =======================
# Accuracy knobs (what to change for final runs)
# =======================
# The comments below describe *what to change in code/config* for production runs and
# a rough estimate of the expected impact on accuracy / runtime. These are guidelines
# — exact numbers depend on your data and task.
#
# 1) sequence_length (in prepare_data call)
#    - Increase from 7 -> 14 or 21 to capture longer temporal context.
#    - Accuracy effect: +3% to +8% on temporal forecasting metrics (if signal exists).
#    - Runtime effect: linear increase in forward pass time with sequence_length.
#
# 2) modality_embed_dim (in model config when creating model)
#    - Increase from 64 -> 128 or 256 to let encoders learn richer embeddings.
#    - Accuracy effect: +1% to +5%, best with more training and regularization.
#    - Runtime effect: model size grows ~O(d^2) in some layers; memory and time increase.
#
# 3) graph_hidden_dim / num_graph_layers / GAT heads
#    - graph_hidden_dim: 128 -> 256 increases model capacity (+2% to +6%).
#    - num_graph_layers: 3 -> 4/5 may help but risks over-smoothing; tune carefully.
#    - GAT heads: 1 -> 4 adds expressivity; if on CPU prefer fewer heads.
#
# 4) batch_size & epochs
#    - Larger batch_size accelerates throughput on GPU; on CPU keep small (1-4).
#    - More epochs: 30 -> 100 may give +2% to +7% but watch for overfitting.
#
# 5) data augmentation / more features
#    - Add additional weather/topo features (if available) -> often best single gain (+5%).
#
# 6) training loss weights
#    - Tune weights between occurrence vs intensity heads if your main goal is one.
#
# These are approximate expected changes; always validate on a hold-out set.

# -------------------- Main --------------------
def main():
    print("=" * 60)
    print("STEP 6: TRAINING")
    print("=" * 60)

    # Production config (you can edit these before run)
    model_config = {
        'fire_feature_dim': 3,
        'weather_feature_dim': 3,
        'topo_feature_dim': 7,
        'modality_embed_dim': 64,   # accuracy knob: raise to 128 for +1–5% (slower)
        'graph_hidden_dim': 128,    # accuracy knob: raise to 256 for capacity
        'num_graph_layers': 3,
        'sequence_length': 7,
        'conv_type': 'GAT',
        'fusion_type': 'attention'
    }

    # Create model with config
    model = WildfirePredictionModel(
        fire_feature_dim=model_config['fire_feature_dim'],
        weather_feature_dim=model_config['weather_feature_dim'],
        topo_feature_dim=model_config['topo_feature_dim'],
        modality_embed_dim=model_config['modality_embed_dim'],
        graph_hidden_dim=model_config['graph_hidden_dim'],
        num_graph_layers=model_config['num_graph_layers'],
        sequence_length=model_config['sequence_length'],
        conv_type=model_config['conv_type'],
        fusion_type=model_config['fusion_type']
    )

    # Trainer: explicitly use CPU for now (or 'cuda' if available)
    trainer = WildfireTrainer(model, device='cpu')

    try:
        # DEBUG: set debug_max_samples to a small number (e.g., 1000 or 3000) for fast CPU runs.
        # For final training set debug_max_samples=None to use the whole dataset.
        train_seq, train_tgt, val_seq, val_tgt, test_seq, test_tgt = trainer.prepare_data(
            processed_data_path='processed_data.pkl',
            temporal_graphs_path='temporal_graphs.pkl',
            sequence_length=model_config['sequence_length'],
            debug_max_samples=2000  # << change to None for full run (remove this for final GPU runs)
        )

        # Train: defaults are conservative for CPU. On GPU increase num_epochs and batch_size.
        trainer.train(
            train_seq, train_tgt,
            val_seq, val_tgt,
            num_epochs=5,     # increase to 30-100 for production training (longer -> usually better)
            batch_size=1,     # increase to 16-64 on GPU (requires careful collation)
            lr=0.001,
            save_dir='checkpoints'
        )

    except FileNotFoundError as e:
        print(f"❌ Missing file: {e}")
        print("Please ensure you've run Steps 1–4 first (data_loading, preprocessing, graph_construction, model_architecture).")

    return trainer


if __name__ == "__main__":
    trainer = main()
