"""
Utility functions for wildfire spread prediction project.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader


def visualize_graph_structure(data: Data, 
                              title: str = "Graph Structure",
                              save_path: Optional[str] = None):
    """
    Visualize graph structure (nodes and edges).
    
    Args:
        data: PyTorch Geometric Data object
        title: Plot title
        save_path: Path to save figure
    """
    try:
        from torch_geometric.utils import to_networkx
        import networkx as nx
        
        G = to_networkx(data, to_undirected=True)
        
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        nx.draw(G, pos, node_size=20, node_color='red', 
               edge_color='gray', alpha=0.6, with_labels=False)
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    except ImportError:
        print("NetworkX not available for graph visualization")


def plot_training_history(train_losses: List[float],
                         val_losses: List[float],
                         save_path: Optional[str] = None):
    """
    Plot training history.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_predictions_vs_actual(predictions: np.ndarray,
                               actuals: np.ndarray,
                               title: str = "Predictions vs Actual",
                               save_path: Optional[str] = None):
    """
    Plot predictions vs actual values.
    
    Args:
        predictions: Predicted values
        actuals: Actual values
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([actuals.min(), actuals.max()], 
            [actuals.min(), actuals.max()], 
            'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_spatial_visualization(df: pd.DataFrame,
                                 predictions: Optional[np.ndarray] = None,
                                 title: str = "Fire Locations",
                                 save_path: Optional[str] = None):
    """
    Create spatial visualization of fire locations.
    
    Args:
        df: Dataframe with latitude and longitude
        predictions: Optional predictions to color-code
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(12, 10))
    
    if predictions is not None:
        scatter = plt.scatter(df['longitude'], df['latitude'], 
                           c=predictions, cmap='YlOrRd', 
                           s=20, alpha=0.6, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, label='Fire Intensity (Predicted)')
    else:
        plt.scatter(df['longitude'], df['latitude'], 
                   c='red', s=20, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def compute_spatial_metrics(predictions: np.ndarray,
                           actuals: np.ndarray,
                           coordinates: np.ndarray,
                           threshold: float = 0.05) -> Dict:
    """
    Compute spatial accuracy metrics.
    
    Args:
        predictions: Predicted fire locations/intensities
        actuals: Actual fire locations/intensities
        coordinates: Array of [latitude, longitude]
        threshold: Distance threshold for spatial matching
        
    Returns:
        Dictionary of spatial metrics
    """
    from scipy.spatial.distance import cdist
    
    # Find predicted fire locations
    pred_fire_locs = coordinates[predictions > 0.5]
    actual_fire_locs = coordinates[actuals > 0.5]
    
    if len(pred_fire_locs) == 0 or len(actual_fire_locs) == 0:
        return {'spatial_precision': 0.0, 'spatial_recall': 0.0}
    
    # Compute distances between predicted and actual locations
    distances = cdist(pred_fire_locs, actual_fire_locs)
    
    # Count matches (within threshold)
    matches = np.sum(distances.min(axis=1) <= threshold)
    
    spatial_precision = matches / len(pred_fire_locs) if len(pred_fire_locs) > 0 else 0.0
    spatial_recall = matches / len(actual_fire_locs) if len(actual_fire_locs) > 0 else 0.0
    
    return {
        'spatial_precision': spatial_precision,
        'spatial_recall': spatial_recall,
        'num_predicted_fires': len(pred_fire_locs),
        'num_actual_fires': len(actual_fire_locs)
    }


def create_data_loader_from_graphs(graphs: List[Data],
                                  batch_size: int = 32,
                                  shuffle: bool = True) -> DataLoader:
    """
    Create DataLoader from list of graphs.
    
    Args:
        graphs: List of PyTorch Geometric Data objects
        batch_size: Batch size
        shuffle: Whether to shuffle
        
    Returns:
        DataLoader
    """
    return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle)


def normalize_features(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features using z-score normalization.
    
    Args:
        features: Feature array [N, F]
        
    Returns:
        Tuple of (normalized_features, mean, std)
    """
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    normalized = (features - mean) / std
    return normalized, mean, std


def denormalize_features(normalized_features: np.ndarray,
                        mean: np.ndarray,
                        std: np.ndarray) -> np.ndarray:
    """
    Denormalize features.
    
    Args:
        normalized_features: Normalized feature array
        mean: Mean used for normalization
        std: Std used for normalization
        
    Returns:
        Denormalized features
    """
    return normalized_features * std + mean

