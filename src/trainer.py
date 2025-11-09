"""
Training utilities for wildfire spread prediction models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import os


class WildfireTrainer:
    """Trainer for wildfire spread prediction models."""
    
    def __init__(self,
                 model: nn.Module,
                 device: Optional[torch.device] = None,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            device: Device to train on (cuda/cpu)
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
        """
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()  # For regression (fire intensity)
        self.binary_criterion = nn.BCEWithLogitsLoss()  # For binary classification (fire presence)
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, dataloader: DataLoader, task: str = 'regression') -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader with training data
            task: 'regression' or 'classification'
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            batch = batch.to(self.device)
            
            # Forward pass
            if isinstance(batch, Batch):
                out = self.model(batch.x, batch.edge_index, batch.batch)
            else:
                out = self.model(batch.x, batch.edge_index)
            
            # Get targets
            if hasattr(batch, 'y'):
                targets = batch.y
            else:
                # If no targets, skip this batch
                continue
            
            # Compute loss
            if task == 'regression':
                loss = self.criterion(out.squeeze(), targets.float())
            else:
                loss = self.binary_criterion(out.squeeze(), targets.float())
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self, dataloader: DataLoader, task: str = 'regression') -> Tuple[float, Dict]:
        """
        Validate model.
        
        Args:
            dataloader: DataLoader with validation data
            task: 'regression' or 'classification'
            
        Returns:
            Tuple of (average loss, metrics dictionary)
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                batch = batch.to(self.device)
                
                # Forward pass
                if isinstance(batch, Batch):
                    out = self.model(batch.x, batch.edge_index, batch.batch)
                else:
                    out = self.model(batch.x, batch.edge_index)
                
                if hasattr(batch, 'y'):
                    targets = batch.y
                    
                    # Compute loss
                    if task == 'regression':
                        loss = self.criterion(out.squeeze(), targets.float())
                    else:
                        loss = self.binary_criterion(out.squeeze(), targets.float())
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Store predictions
                    all_preds.append(out.squeeze().cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Compute metrics
        if len(all_preds) > 0:
            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            
            if task == 'regression':
                mae = np.mean(np.abs(all_preds - all_targets))
                rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
                metrics = {'mae': mae, 'rmse': rmse}
            else:
                # Binary classification metrics
                pred_binary = (all_preds > 0.5).astype(int)
                accuracy = np.mean(pred_binary == all_targets)
                precision = np.sum((pred_binary == 1) & (all_targets == 1)) / (np.sum(pred_binary == 1) + 1e-8)
                recall = np.sum((pred_binary == 1) & (all_targets == 1)) / (np.sum(all_targets == 1) + 1e-8)
                metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall}
        else:
            metrics = {}
        
        return avg_loss, metrics
    
    def train(self,
             train_loader: DataLoader,
             val_loader: DataLoader,
             num_epochs: int = 50,
             task: str = 'regression',
             save_path: Optional[str] = None,
             patience: int = 10):
        """
        Full training loop.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs: Number of epochs
            task: 'regression' or 'classification'
            save_path: Path to save best model
            patience: Early stopping patience
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, task)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader, task)
            self.val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Val Metrics: {val_metrics}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_metrics': val_metrics
                    }, save_path)
                    print(f"Saved best model to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
    
    def predict(self, dataloader: DataLoader) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            dataloader: DataLoader with test data
            
        Returns:
            Predictions array
        """
        self.model.eval()
        all_preds = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                batch = batch.to(self.device)
                
                if isinstance(batch, Batch):
                    out = self.model(batch.x, batch.edge_index, batch.batch)
                else:
                    out = self.model(batch.x, batch.edge_index)
                
                all_preds.append(out.squeeze().cpu().numpy())
        
        return np.concatenate(all_preds)

