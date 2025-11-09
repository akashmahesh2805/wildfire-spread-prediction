"""
Diagnostic and fix script for model performance issues.
Addresses: scale mismatch, false positives, poor localization.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from data_loader import WildfireDataLoader
from grid_graph_builder import GridBasedGraphBuilder
from models import MultiModalGCN, GraphAttentionWildfire
from trainer import WildfireTrainer

print("=" * 60)
print("DIAGNOSING MODEL PERFORMANCE ISSUES")
print("=" * 60)

# Load data
loader = WildfireDataLoader()
df = loader.merge_data()
df = loader.prepare_features(include_temporal=True, include_spatial=True)
df = df.sort_values('time').ffill().fillna(0)

feature_groups = loader.get_feature_columns(df)

# Split data
graph_builder = GridBasedGraphBuilder(grid_size=0.01, temporal_window=1)
train_df, val_df, test_df = graph_builder.split_temporal(df, train_ratio=0.7, val_ratio=0.15)

print("\n" + "=" * 60)
print("ISSUE 1: Target Scale Mismatch")
print("=" * 60)

# Analyze target distribution
train_df_with_grid = graph_builder.assign_events_to_grid(train_df)
graph_builder.create_grid(train_df)  # Create grid first
train_targets = graph_builder.create_targets(train_df_with_grid, prediction_horizon=1)

print(f"Target statistics:")
print(f"  Min: {train_targets.min():.4f}")
print(f"  Max: {train_targets.max():.4f}")
print(f"  Mean: {train_targets.mean():.4f}")
print(f"  Std: {train_targets.std():.4f}")
print(f"  Zero ratio: {(train_targets == 0).sum() / len(train_targets):.2%}")
print(f"  Non-zero ratio: {(train_targets > 0).sum() / len(train_targets):.2%}")

# Check actual FRP values
actual_frp = train_df['frp'].values
print(f"\nActual FRP statistics:")
print(f"  Min: {actual_frp.min():.4f}")
print(f"  Max: {actual_frp.max():.4f}")
print(f"  Mean: {actual_frp.mean():.4f}")
print(f"  Std: {actual_frp.std():.4f}")

print("\n⚠️ PROBLEM: Target scale (0-{:.1f}) doesn't match actual FRP scale (0-{:.1f})".format(
    train_targets.max(), actual_frp.max()))

print("\n" + "=" * 60)
print("ISSUE 2: Class Imbalance")
print("=" * 60)

zero_count = (train_targets == 0).sum()
non_zero_count = (train_targets > 0).sum()
print(f"Zero targets: {zero_count} ({100*zero_count/len(train_targets):.2f}%)")
print(f"Non-zero targets: {non_zero_count} ({100*non_zero_count/len(train_targets):.2f}%)")
print("\n⚠️ PROBLEM: Severe class imbalance - model learns to predict zeros!")

print("\n" + "=" * 60)
print("SOLUTION 1: Improved Target Creation")
print("=" * 60)

def create_better_targets(df, grid_cells, cell_to_index, prediction_horizon=1, spatial_radius=0.05):
    """
    Create better targets that:
    1. Predict fire spread to nearby cells (not just continuation)
    2. Use actual FRP values (not aggregated)
    3. Handle multiple time steps
    """
    df_sorted = df.sort_values('time').reset_index(drop=True)
    num_cells = len(grid_cells)
    targets = []
    
    # Get unique times
    unique_times = sorted(df_sorted['time'].unique())
    
    for cell_idx in range(num_cells):
        cell_coords = grid_cells[cell_idx]
        cell_targets = []
        
        # For each time step, check future
        for t_idx in range(len(unique_times) - 1):
            current_time = unique_times[t_idx]
            future_time = current_time + pd.Timedelta(hours=prediction_horizon)
            
            # Find fires in this cell at current time
            current_fires = df_sorted[
                (df_sorted['time'] == current_time) &
                (np.abs(df_sorted['latitude'] - cell_coords[0]) <= 0.005) &
                (np.abs(df_sorted['longitude'] - cell_coords[1]) <= 0.005)
            ]
            
            if len(current_fires) > 0:
                # Fire exists in this cell - predict spread
                # Check future fires in nearby area (spatial radius)
                future_fires = df_sorted[
                    (df_sorted['time'] == future_time) &
                    (np.abs(df_sorted['latitude'] - cell_coords[0]) <= spatial_radius) &
                    (np.abs(df_sorted['longitude'] - cell_coords[1]) <= spatial_radius)
                ]
                
                if len(future_fires) > 0 and 'frp' in future_fires.columns:
                    # Maximum FRP in nearby area (fire spread)
                    max_frp = future_fires['frp'].max()
                    cell_targets.append(max_frp)
                else:
                    # Fire might have extinguished
                    cell_targets.append(0.0)
            else:
                # No fire in this cell - predict if fire spreads here
                future_fires = df_sorted[
                    (df_sorted['time'] == future_time) &
                    (np.abs(df_sorted['latitude'] - cell_coords[0]) <= spatial_radius) &
                    (np.abs(df_sorted['longitude'] - cell_coords[1]) <= spatial_radius)
                ]
                
                if len(future_fires) > 0 and 'frp' in future_fires.columns:
                    # Fire spreads to this cell
                    max_frp = future_fires['frp'].max()
                    cell_targets.append(max_frp)
                else:
                    cell_targets.append(0.0)
        
        # Use maximum target across all time steps
        target = max(cell_targets) if len(cell_targets) > 0 else 0.0
        targets.append(float(target))
    
    return torch.FloatTensor(targets)

print("Creating improved targets...")
graph_builder.create_grid(train_df)
train_df_with_grid = graph_builder.assign_events_to_grid(train_df)
better_targets = create_better_targets(
    train_df_with_grid, 
    graph_builder.grid_cells,
    graph_builder.cell_to_index,
    prediction_horizon=1,
    spatial_radius=0.05
)

print(f"Improved target statistics:")
print(f"  Min: {better_targets.min():.4f}")
print(f"  Max: {better_targets.max():.4f}")
print(f"  Mean: {better_targets.mean():.4f}")
print(f"  Non-zero: {(better_targets > 0).sum() / len(better_targets):.2%}")

print("\n" + "=" * 60)
print("SOLUTION 2: Weighted Loss for Class Imbalance")
print("=" * 60)

# Calculate class weights
zero_weight = len(better_targets) / (2 * (better_targets == 0).sum())
non_zero_weight = len(better_targets) / (2 * (better_targets > 0).sum())

print(f"Class weights:")
print(f"  Zero class weight: {zero_weight:.4f}")
print(f"  Non-zero class weight: {non_zero_weight:.4f}")

print("\n" + "=" * 60)
print("SOLUTION 3: Try Different Model")
print("=" * 60)

print("Recommendations:")
print("1. Use Graph Attention Network (GAT) - better for spatial patterns")
print("2. Add weighted loss")
print("3. Normalize targets properly (match actual FRP scale)")
print("4. Use focal loss or weighted MSE loss")

print("\n" + "=" * 60)
print("NEXT STEPS")
print("=" * 60)
print("""
1. Fix target creation (use actual FRP values, predict spread)
2. Add weighted loss to handle class imbalance
3. Try GAT model (attention mechanism)
4. Normalize targets to match actual scale
5. Add threshold for fire/no-fire classification
6. Post-process predictions (remove isolated predictions)
""")

