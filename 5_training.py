"""
Step 5: Training
================
This module handles model training with validation, checkpointing, and logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import pickle
import json
from datetime import datetime

# Import model
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("model_architecture", "4_model_architecture.py")
model_arch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_arch)
WildfirePredictionModel = model_arch.WildfirePredictionModel
create_model = model_arch.create_model

class WildfireDataset(Dataset):
    """Dataset class for wildfire graph sequences."""
    
    def __init__(self, graph_sequences, targets, edge_index, edge_attr=None):
        """
        Initialize dataset.
        
        Args:
            graph_sequences: List of graph sequences (each sequence is a list of graphs)
            targets: List of target dictionaries
            edge_index: Edge indices (same for all graphs)
            edge_attr: Edge attributes (optional)
        """
        self.graph_sequences = graph_sequences
        self.targets = targets
        self.edge_index = edge_index
        self.edge_attr = edge_attr
    
    def __len__(self):
        return len(self.graph_sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': self.graph_sequences[idx],
            'target': self.targets[idx],
            'edge_index': self.edge_index,
            'edge_attr': self.edge_attr
        }


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Initialize focal loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """Forward pass."""
        ce_loss = nn.functional.binary_cross_entropy(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WildfireTrainer:
    """Trainer class for wildfire prediction model."""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            device: Device to use for training
        """
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def prepare_data(self, processed_data_path='processed_data.pkl', 
                    graph_data_path='graph_data.pkl',
                    sequence_length=7, test_size=0.2, val_size=0.1):
        """
        Prepare training data from processed files.
        
        Args:
            processed_data_path: Path to processed data
            graph_data_path: Path to graph data
            sequence_length: Length of input sequences
            test_size: Proportion of test set
            val_size: Proportion of validation set
        """
        print("Preparing training data...")
        
        # Load processed data
        with open(processed_data_path, 'rb') as f:
            processed_df = pickle.load(f)
        
        # Load graph data
        with open(graph_data_path, 'rb') as f:
            graph_data = pickle.load(f)
        
        # Create sequences (simplified - in practice, use preprocessor's prepare_sequences)
        sequences, targets = self._create_sequences(
            processed_df, graph_data, sequence_length
        )
        
        # Split data
        train_val_sequences, test_sequences, train_val_targets, test_targets = \
            train_test_split(sequences, targets, test_size=test_size, random_state=42)
        
        train_sequences, val_sequences, train_targets, val_targets = \
            train_test_split(train_val_sequences, train_val_targets, 
                           test_size=val_size/(1-test_size), random_state=42)
        
        print(f"   Train: {len(train_sequences)} sequences")
        print(f"   Validation: {len(val_sequences)} sequences")
        print(f"   Test: {len(test_sequences)} sequences")
        
        return (train_sequences, train_targets, 
                val_sequences, val_targets,
                test_sequences, test_targets, graph_data)
    
    def _create_sequences(self, processed_df, graph_data, sequence_length):
        """Create sequences from processed data."""
        sequences = []
        targets = []
        
        # Group by location
        for (lat, lon), group in processed_df.groupby(['latitude', 'longitude']):
            group = group.sort_values('time_window').reset_index(drop=True)
            
            # Create sequences
            for i in range(len(group) - sequence_length):
                seq = group.iloc[i:i+sequence_length]
                target_idx = i + sequence_length
                
                if target_idx < len(group):
                    target = group.iloc[target_idx]
                    
                    # Extract features for sequence
                    seq_features = []
                    for _, row in seq.iterrows():
                        # Exclude spatial and temporal columns
                        exclude = ['latitude', 'longitude', 'time_window', 
                                  'acq_date_parsed', 'date_parsed', 'datetime', 'acq_datetime']
                        features = row.drop(exclude).values.astype(np.float32)
                        seq_features.append(torch.tensor(features))
                    
                    sequences.append(seq_features)
                    
                    # Create target
                    target_dict = {
                        'fire_occurrence': torch.tensor(
                            float(target.get('has_fire', 0)), dtype=torch.float32
                        ),
                        'fire_intensity': torch.tensor(
                            float(target.get('frp_mean', 0)), dtype=torch.float32
                        )
                    }
                    targets.append(target_dict)
        
        return sequences, targets
    
    def train_epoch(self, train_loader, optimizer, criterion_dict, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch in pbar:
            # Move data to device
            sequences = [seq.to(self.device) for seq in batch['sequence']]
            edge_index = batch['edge_index'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['target'].items()}
            
            # Forward pass
            optimizer.zero_grad()
            predictions = self.model(sequences, edge_index)
            
            # Calculate losses
            loss_dict = {}
            total_batch_loss = 0
            
            # Fire occurrence loss (binary classification)
            if 'fire_occurrence' in predictions:
                loss_dict['occurrence'] = criterion_dict['occurrence'](
                    predictions['fire_occurrence'].squeeze(),
                    targets['fire_occurrence']
                )
                total_batch_loss += loss_dict['occurrence']
            
            # Fire intensity loss (regression)
            if 'fire_intensity' in predictions:
                # Only compute loss where fire occurred
                fire_mask = targets['fire_occurrence'] > 0.5
                if fire_mask.sum() > 0:
                    loss_dict['intensity'] = criterion_dict['intensity'](
                        predictions['fire_intensity'].squeeze()[fire_mask],
                        targets['fire_intensity'][fire_mask]
                    )
                    total_batch_loss += loss_dict['intensity']
            
            # Backward pass
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += total_batch_loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_batch_loss.item():.4f}',
                'occ': f'{loss_dict.get("occurrence", 0):.4f}',
                'int': f'{loss_dict.get("intensity", 0):.4f}'
            })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def validate(self, val_loader, criterion_dict):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                sequences = [seq.to(self.device) for seq in batch['sequence']]
                edge_index = batch['edge_index'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch['target'].items()}
                
                predictions = self.model(sequences, edge_index)
                
                # Calculate losses
                batch_loss = 0
                
                if 'fire_occurrence' in predictions:
                    batch_loss += criterion_dict['occurrence'](
                        predictions['fire_occurrence'].squeeze(),
                        targets['fire_occurrence']
                    )
                
                if 'fire_intensity' in predictions:
                    fire_mask = targets['fire_occurrence'] > 0.5
                    if fire_mask.sum() > 0:
                        batch_loss += criterion_dict['intensity'](
                            predictions['fire_intensity'].squeeze()[fire_mask],
                            targets['fire_intensity'][fire_mask]
                        )
                
                total_loss += batch_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def train(self, train_sequences, train_targets, val_sequences, val_targets,
              graph_data, num_epochs=50, batch_size=32, learning_rate=0.001,
              weight_decay=1e-5, save_dir='checkpoints'):
        """
        Train the model.
        
        Args:
            train_sequences: Training sequences
            train_targets: Training targets
            val_sequences: Validation sequences
            val_targets: Validation targets
            graph_data: Graph structure
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay
            save_dir: Directory to save checkpoints
        """
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Create datasets
        edge_index = graph_data.edge_index
        edge_attr = graph_data.edge_attr if hasattr(graph_data, 'edge_attr') else None
        
        train_dataset = WildfireDataset(train_sequences, train_targets, edge_index, edge_attr)
        val_dataset = WildfireDataset(val_sequences, val_targets, edge_index, edge_attr)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer and loss functions
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        criterion_dict = {
            'occurrence': FocalLoss(alpha=2, gamma=2),
            'intensity': nn.MSELoss()
        }
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, criterion_dict, epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader, criterion_dict)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            print(f"   Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(save_dir, epoch, val_loss)
                print(f"   ✓ Saved best model (val_loss: {val_loss:.4f})")
            
            # Save training history
            self.save_training_history(save_dir)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, save_dir, epoch, val_loss):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # Save best model
        torch.save(checkpoint, os.path.join(save_dir, 'best_model.pt'))
        
        # Save latest model
        torch.save(checkpoint, os.path.join(save_dir, 'latest_model.pt'))
    
    def save_training_history(self, save_dir):
        """Save training history."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")


def main():
    """Main function to run training."""
    print("="*60)
    print("STEP 5: TRAINING")
    print("="*60)
    
    # Create model
    config = {
        'fire_feature_dim': 10,
        'weather_feature_dim': 8,
        'topo_feature_dim': 9,
        'modality_embed_dim': 64,
        'graph_hidden_dim': 128,
        'num_graph_layers': 3,
        'sequence_length': 7,
        'prediction_horizon': 1,
        'conv_type': 'GCN',
        'fusion_type': 'attention'
    }
    
    model = create_model(config)
    
    # Initialize trainer
    trainer = WildfireTrainer(model)
    
    # Prepare data
    try:
        (train_sequences, train_targets,
         val_sequences, val_targets,
         test_sequences, test_targets, graph_data) = trainer.prepare_data()
        
        # Train model
        trainer.train(
            train_sequences, train_targets,
            val_sequences, val_targets,
            graph_data,
            num_epochs=50,
            batch_size=32,
            learning_rate=0.001
        )
        
    except FileNotFoundError as e:
        print(f"Error: Required data files not found. Please run previous steps first.")
        print(f"Missing file: {e}")
        print("\nTo train the model, you need to:")
        print("1. Run 1_data_loading.py")
        print("2. Run 2_data_preprocessing.py")
        print("3. Run 3_graph_construction.py")
        print("4. Then run this training script")
    
    return trainer


if __name__ == "__main__":
    trainer = main()

