"""
Step 6: Evaluation
===================
This module handles model evaluation, metrics calculation, and visualization.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score
)
from scipy.spatial.distance import directed_hausdorff
import os
import pickle
from tqdm import tqdm

# Import model and trainer
import sys
import importlib.util

# Import model architecture
spec = importlib.util.spec_from_file_location("model_architecture", "4_model_architecture.py")
model_arch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_arch)
WildfirePredictionModel = model_arch.WildfirePredictionModel
create_model = model_arch.create_model

# Import training utilities
spec = importlib.util.spec_from_file_location("training", "5_training.py")
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)
WildfireDataset = training.WildfireDataset

class ModelEvaluator:
    """Class for evaluating wildfire prediction model."""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            device: Device to use
        """
        self.model = model.to(device)
        self.device = device
        self.results = {}
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from checkpoint: {checkpoint_path}")
    
    def evaluate_classification(self, y_true, y_pred, y_proba=None, threshold=0.5):
        """
        Evaluate binary classification performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels (or probabilities if threshold provided)
            y_proba: Predicted probabilities
            threshold: Classification threshold
        
        Returns:
            Dictionary of metrics
        """
        # Convert probabilities to predictions if needed
        if y_proba is not None:
            y_pred_binary = (y_proba >= threshold).astype(int)
        else:
            y_pred_binary = (y_pred >= threshold).astype(int) if threshold else y_pred
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'precision': precision_score(y_true, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true, y_pred_binary, zero_division=0),
            'f1_score': f1_score(y_true, y_pred_binary, zero_division=0)
        }
        
        # ROC-AUC if probabilities available
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except ValueError:
                metrics['roc_auc'] = 0.0
        
        return metrics
    
    def evaluate_regression(self, y_true, y_pred):
        """
        Evaluate regression performance.
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2_score': r2_score(y_true, y_pred)
        }
        
        # Calculate percentage errors
        non_zero_mask = y_true != 0
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
            metrics['mape'] = mape
        
        return metrics
    
    def evaluate_spatial_accuracy(self, true_locations, pred_locations, threshold_km=5.0):
        """
        Evaluate spatial accuracy of fire predictions.
        
        Args:
            true_locations: List of true fire locations (lat, lon)
            pred_locations: List of predicted fire locations (lat, lon)
            threshold_km: Distance threshold for correct prediction (km)
        
        Returns:
            Dictionary of spatial metrics
        """
        from math import radians, sin, cos, sqrt, atan2
        
        def haversine_distance(coord1, coord2):
            """Calculate Haversine distance."""
            lat1, lon1 = radians(coord1[0]), radians(coord1[1])
            lat2, lon2 = radians(coord2[0]), radians(coord2[1])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            return 6371 * c  # km
        
        if len(true_locations) == 0 and len(pred_locations) == 0:
            return {'spatial_precision': 1.0, 'spatial_recall': 1.0, 'spatial_f1': 1.0}
        
        if len(true_locations) == 0:
            return {'spatial_precision': 0.0, 'spatial_recall': 1.0, 'spatial_f1': 0.0}
        
        if len(pred_locations) == 0:
            return {'spatial_precision': 1.0, 'spatial_recall': 0.0, 'spatial_f1': 0.0}
        
        # Calculate distances
        true_matched = set()
        pred_matched = set()
        
        for i, true_loc in enumerate(true_locations):
            for j, pred_loc in enumerate(pred_locations):
                dist = haversine_distance(true_loc, pred_loc)
                if dist <= threshold_km:
                    true_matched.add(i)
                    pred_matched.add(j)
        
        precision = len(pred_matched) / len(pred_locations) if len(pred_locations) > 0 else 0
        recall = len(true_matched) / len(true_locations) if len(true_locations) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'spatial_precision': precision,
            'spatial_recall': recall,
            'spatial_f1': f1
        }
    
    def evaluate_on_test_set(self, test_loader):
        """
        Evaluate model on test set.
        
        Args:
            test_loader: DataLoader for test set
        
        Returns:
            Dictionary of evaluation results
        """
        print("\nEvaluating on test set...")
        
        self.model.eval()
        all_predictions = {
            'fire_occurrence': [],
            'fire_intensity': []
        }
        all_targets = {
            'fire_occurrence': [],
            'fire_intensity': []
        }
        all_locations = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluating'):
                sequences = [seq.to(self.device) for seq in batch['sequence']]
                edge_index = batch['edge_index'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch['target'].items()}
                
                predictions = self.model(sequences, edge_index)
                
                # Collect predictions and targets
                all_predictions['fire_occurrence'].append(
                    predictions['fire_occurrence'].cpu().numpy()
                )
                all_predictions['fire_intensity'].append(
                    predictions['fire_intensity'].cpu().numpy()
                )
                
                all_targets['fire_occurrence'].append(
                    targets['fire_occurrence'].cpu().numpy()
                )
                all_targets['fire_intensity'].append(
                    targets['fire_intensity'].cpu().numpy()
                )
        
        # Concatenate all predictions
        pred_occurrence = np.concatenate(all_predictions['fire_occurrence']).flatten()
        pred_intensity = np.concatenate(all_predictions['fire_intensity']).flatten()
        true_occurrence = np.concatenate(all_targets['fire_occurrence']).flatten()
        true_intensity = np.concatenate(all_targets['fire_intensity']).flatten()
        
        # Evaluate classification (fire occurrence)
        classification_metrics = self.evaluate_classification(
            true_occurrence, pred_occurrence, y_proba=pred_occurrence
        )
        
        # Evaluate regression (fire intensity) - only on fires
        fire_mask = true_occurrence > 0.5
        if fire_mask.sum() > 0:
            regression_metrics = self.evaluate_regression(
                true_intensity[fire_mask], pred_intensity[fire_mask]
            )
        else:
            regression_metrics = {'mae': 0, 'mse': 0, 'rmse': 0, 'r2_score': 0}
        
        results = {
            'classification': classification_metrics,
            'regression': regression_metrics,
            'predictions': {
                'occurrence': pred_occurrence,
                'intensity': pred_intensity
            },
            'targets': {
                'occurrence': true_occurrence,
                'intensity': true_intensity
            }
        }
        
        self.results = results
        return results
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='confusion_matrix.png'):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Fire', 'Fire'],
                   yticklabels=['No Fire', 'Fire'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix - Fire Occurrence')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrix to {save_path}")
    
    def plot_predictions_vs_targets(self, y_true, y_pred, title='Predictions vs Targets', 
                                   save_path='predictions_vs_targets.png'):
        """Plot predictions vs targets."""
        plt.figure(figsize=(10, 6))
        
        plt.scatter(y_true, y_pred, alpha=0.5, s=10)
        
        # Perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {save_path}")
    
    def plot_roc_curve(self, y_true, y_proba, save_path='roc_curve.png'):
        """Plot ROC curve."""
        from sklearn.metrics import roc_curve
        
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Fire Occurrence Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved ROC curve to {save_path}")
    
    def generate_evaluation_report(self, results, save_path='evaluation_report.txt'):
        """Generate comprehensive evaluation report."""
        with open(save_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            
            # Classification metrics
            f.write("FIRE OCCURRENCE PREDICTION (Classification)\n")
            f.write("-"*60 + "\n")
            for metric, value in results['classification'].items():
                f.write(f"{metric.upper()}: {value:.4f}\n")
            
            f.write("\nFIRE INTENSITY PREDICTION (Regression)\n")
            f.write("-"*60 + "\n")
            for metric, value in results['regression'].items():
                f.write(f"{metric.upper()}: {value:.4f}\n")
            
            f.write("\n" + "="*60 + "\n")
        
        print(f"Saved evaluation report to {save_path}")
    
    def visualize_results(self, results, save_dir='evaluation_plots'):
        """Generate visualization plots."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Confusion matrix
        pred_binary = (results['predictions']['occurrence'] >= 0.5).astype(int)
        self.plot_confusion_matrix(
            results['targets']['occurrence'].astype(int),
            pred_binary,
            os.path.join(save_dir, 'confusion_matrix.png')
        )
        
        # ROC curve
        self.plot_roc_curve(
            results['targets']['occurrence'],
            results['predictions']['occurrence'],
            os.path.join(save_dir, 'roc_curve.png')
        )
        
        # Fire intensity predictions vs targets
        fire_mask = results['targets']['occurrence'] > 0.5
        if fire_mask.sum() > 0:
            self.plot_predictions_vs_targets(
                results['targets']['intensity'][fire_mask],
                results['predictions']['intensity'][fire_mask],
                'Fire Intensity: Predictions vs Targets',
                os.path.join(save_dir, 'intensity_predictions.png')
            )
        
        print(f"\nAll visualizations saved to {save_dir}/")


def main():
    """Main function to run evaluation."""
    print("="*60)
    print("STEP 6: EVALUATION")
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
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model)
    
    # Load checkpoint if available
    checkpoint_path = 'checkpoints/best_model.pt'
    if os.path.exists(checkpoint_path):
        evaluator.load_checkpoint(checkpoint_path)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using 5_training.py")
        return evaluator
    
    # Load test data
    try:
        # In practice, load test data from files
        # For now, this is a placeholder
        print("\nNote: Test data loading needs to be implemented based on your data structure")
        print("Please modify this script to load your test data")
        
        # Example evaluation (commented out - needs actual test data)
        # test_loader = ...  # Load test data
        # results = evaluator.evaluate_on_test_set(test_loader)
        # evaluator.generate_evaluation_report(results)
        # evaluator.visualize_results(results)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Please ensure test data is properly prepared")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    
    return evaluator


if __name__ == "__main__":
    evaluator = main()

