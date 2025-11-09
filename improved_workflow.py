"""
Improved workflow with fixes for model performance issues:
1. Better target creation (predict spread, use actual FRP)
2. Weighted loss for class imbalance
3. GAT model (better spatial attention)
4. Proper target normalization
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from data_loader import WildfireDataLoader
from grid_graph_builder import GridBasedGraphBuilder
from models import GraphAttentionWildfire  # Using GAT instead of GCN
from trainer import WildfireTrainer

print("=" * 60)
print("IMPROVED WORKFLOW - FIXING MODEL PERFORMANCE")
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
print("STEP 1: Build Graphs")
print("=" * 60)

train_graph = graph_builder.build_graph(train_df, feature_groups)
val_graph = graph_builder.build_graph(val_df, feature_groups)
test_graph = graph_builder.build_graph(test_df, feature_groups)

print("\n" + "=" * 60)
print("STEP 2: Create Improved Targets")
print("=" * 60)

def create_improved_targets(df, grid_cells, cell_to_index, prediction_horizon=1):
    """Create targets that predict fire spread using actual FRP values."""
    df_sorted = df.sort_values('time').reset_index(drop=True)
    num_cells = len(grid_cells)
    targets = []
    
    unique_times = sorted(df_sorted['time'].unique())
    
    for cell_idx in range(num_cells):
        cell_coords = grid_cells[cell_idx]
        cell_targets = []
        
        for t_idx in range(len(unique_times) - 1):
            current_time = unique_times[t_idx]
            future_time = current_time + pd.Timedelta(hours=prediction_horizon)
            
            # Find fires near this cell at future time (spread prediction)
            future_fires = df_sorted[
                (df_sorted['time'] == future_time) &
                (np.abs(df_sorted['latitude'] - cell_coords[0]) <= 0.05) &
                (np.abs(df_sorted['longitude'] - cell_coords[1]) <= 0.05)
            ]
            
            if len(future_fires) > 0 and 'frp' in future_fires.columns:
                # Use actual FRP value (not aggregated)
                max_frp = future_fires['frp'].max()
                cell_targets.append(max_frp)
            else:
                cell_targets.append(0.0)
        
        target = max(cell_targets) if len(cell_targets) > 0 else 0.0
        targets.append(float(target))
    
    return torch.FloatTensor(targets)

# Create targets using improved method
graph_builder.grid_cells = train_graph.grid_cells
graph_builder.cell_to_index = train_graph.cell_to_index
train_df_with_grid = graph_builder.assign_events_to_grid(train_df)
train_targets = create_improved_targets(train_df_with_grid, train_graph.grid_cells, 
                                       train_graph.cell_to_index, prediction_horizon=1)

graph_builder.grid_cells = val_graph.grid_cells
graph_builder.cell_to_index = val_graph.cell_to_index
val_df_with_grid = graph_builder.assign_events_to_grid(val_df)
val_targets = create_improved_targets(val_df_with_grid, val_graph.grid_cells,
                                     val_graph.cell_to_index, prediction_horizon=1)

graph_builder.grid_cells = test_graph.grid_cells
graph_builder.cell_to_index = test_graph.cell_to_index
test_df_with_grid = graph_builder.assign_events_to_grid(test_df)
test_targets = create_improved_targets(test_df_with_grid, test_graph.grid_cells,
                                      test_graph.cell_to_index, prediction_horizon=1)

# Normalize targets to [0, 1] for training (but keep original for evaluation)
train_targets_original = train_targets.clone()
val_targets_original = val_targets.clone()
test_targets_original = test_targets.clone()

# Normalize
target_max = train_targets.max()
if target_max > 0:
    train_targets_norm = train_targets / target_max
    val_targets_norm = val_targets / target_max
    test_targets_norm = test_targets / target_max
else:
    train_targets_norm = train_targets
    val_targets_norm = val_targets
    test_targets_norm = test_targets

print(f"Target statistics:")
print(f"  Original scale: 0 to {target_max:.2f}")
print(f"  Normalized scale: 0 to 1.0")
print(f"  Non-zero ratio: {(train_targets > 0).sum() / len(train_targets):.2%}")

# Assign normalized targets for training
train_graph.y = train_targets_norm
val_graph.y = val_targets_norm
test_graph.y = test_targets_norm

print("\n" + "=" * 60)
print("STEP 3: Initialize Model (GAT)")
print("=" * 60)

input_dim = train_graph.x.shape[1]
model = GraphAttentionWildfire(
    input_dim=input_dim,
    hidden_dim=128,  # Increased capacity
    num_layers=4,    # More layers
    num_heads=8,     # More attention heads
    output_dim=1,
    dropout=0.3
)

num_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model initialized: {num_params:,} parameters")

print("\n" + "=" * 60)
print("STEP 4: Train with Weighted Loss")
print("=" * 60)

# Calculate class weights
zero_count = (train_targets_norm == 0).sum().item()
non_zero_count = (train_targets_norm > 0).sum().item()
total = len(train_targets_norm)

if zero_count > 0 and non_zero_count > 0:
    zero_weight = total / (2 * zero_count)
    non_zero_weight = total / (2 * non_zero_count)
    class_weights = torch.tensor([zero_weight, non_zero_weight])
    print(f"Class weights: Zero={zero_weight:.2f}, Non-zero={non_zero_weight:.2f}")
else:
    class_weights = None

train_loader = DataLoader([train_graph], batch_size=1, shuffle=False)
val_loader = DataLoader([val_graph], batch_size=1, shuffle=False)
test_loader = DataLoader([test_graph], batch_size=1, shuffle=False)

trainer = WildfireTrainer(
    model=model,
    learning_rate=0.0005,  # Lower learning rate
    weight_decay=1e-4,
    class_weights=class_weights
)

os.makedirs('models', exist_ok=True)
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    task='regression',
    save_path='models/improved_model.pt',
    patience=10
)

print("\n" + "=" * 60)
print("STEP 5: Evaluate (Denormalize Predictions)")
print("=" * 60)

checkpoint = torch.load('models/improved_model.pt', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

predictions_norm = trainer.predict(test_loader)
# Denormalize predictions
predictions = predictions_norm * target_max

actuals = test_targets_original.numpy()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))
r2 = r2_score(actuals, predictions)

print(f"\nTest Set Metrics (on original scale):")
print(f"  MAE:  {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²:   {r2:.4f}")

# Binary classification metrics
pred_binary = (predictions > 0.5).astype(int)
actual_binary = (actuals > 0.5).astype(int)

print(f"\nBinary Classification:")
print(f"  Precision: {(pred_binary & actual_binary).sum() / (pred_binary.sum() + 1e-8):.4f}")
print(f"  Recall: {(pred_binary & actual_binary).sum() / (actual_binary.sum() + 1e-8):.4f}")
print(f"  Accuracy: {(pred_binary == actual_binary).sum() / len(pred_binary):.4f}")

print("\n✓ Improved model training complete!")
print("Compare results with previous model to see improvements.")

