"""
Script to compare predicted fire spread with actual fire locations.
Shows side-by-side and overlaid visualizations.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, classification_report

# Import our modules
from data_loader import WildfireDataLoader
from grid_graph_builder import GridBasedGraphBuilder
from models import MultiModalGCN
from trainer import WildfireTrainer
from torch_geometric.loader import DataLoader
import torch

print("=" * 60)
print("COMPARING PREDICTIONS TO ACTUAL FIRES")
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

# Build test graph (already done, but for comparison we need it)
print("\nBuilding test graph...")
test_graph = graph_builder.build_graph(test_df, feature_groups, include_temporal=True, include_spatial=True)

# Assign test events to grid
test_df_with_grid = graph_builder.assign_events_to_grid(test_df)
graph_builder.grid_cells = test_graph.grid_cells
graph_builder.cell_to_index = test_graph.cell_to_index
test_targets = graph_builder.create_targets(test_df_with_grid, prediction_horizon=1)
test_graph.y = test_targets

# Load trained model
print("\nLoading trained model...")
model = MultiModalGCN(input_dim=test_graph.x.shape[1], hidden_dim=64, num_layers=3, output_dim=1, dropout=0.2)
checkpoint = torch.load('models/best_model.pt', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
print("Making predictions...")
test_loader = DataLoader([test_graph], batch_size=1, shuffle=False)
trainer = WildfireTrainer(model)
predictions = trainer.predict(test_loader)
actuals = test_graph.y.numpy()

# Get grid cell coordinates
grid_coords = test_graph.grid_cells

print("\n" + "=" * 60)
print("ANALYSIS: Predicted vs Actual")
print("=" * 60)

# Convert to binary (fire/no fire) for comparison
pred_binary = (predictions > 0.5).astype(int)
actual_binary = (actuals > 0.5).astype(int)

# Calculate metrics
print(f"\nBinary Classification Metrics:")
print(f"  Predicted fires: {pred_binary.sum()} cells")
print(f"  Actual fires: {actual_binary.sum()} cells")
print(f"  Overlap: {(pred_binary & actual_binary).sum()} cells")

# Confusion matrix
cm = confusion_matrix(actual_binary, pred_binary)
print(f"\nConfusion Matrix:")
print(f"  True Negatives (correct no-fire):  {cm[0,0]}")
print(f"  False Positives (predicted fire, no actual): {cm[0,1]}")
print(f"  False Negatives (missed fires):     {cm[1,0]}")
print(f"  True Positives (correct predictions): {cm[1,1]}")

# Calculate spatial accuracy
if cm[1,1] + cm[0,1] > 0:
    precision = cm[1,1] / (cm[1,1] + cm[0,1])
    print(f"  Precision: {precision:.4f}")

if cm[1,1] + cm[1,0] > 0:
    recall = cm[1,1] / (cm[1,1] + cm[1,0])
    print(f"  Recall: {recall:.4f}")

# Create comparison visualizations
print("\nCreating comparison visualizations...")
os.makedirs('results', exist_ok=True)

# 1. Side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Actual fires
scatter1 = axes[0].scatter(grid_coords[:, 1], grid_coords[:, 0], 
                           c=actuals, cmap='YlOrRd', s=10, alpha=0.7, edgecolors='black', linewidth=0.3)
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
axes[0].set_title('Actual Fire Intensity')
plt.colorbar(scatter1, ax=axes[0], label='Fire Intensity (Actual)')

# Predicted fires
scatter2 = axes[1].scatter(grid_coords[:, 1], grid_coords[:, 0], 
                           c=predictions, cmap='YlOrRd', s=10, alpha=0.7, edgecolors='black', linewidth=0.3)
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('Latitude')
axes[1].set_title('Predicted Fire Intensity')
plt.colorbar(scatter2, ax=axes[1], label='Fire Intensity (Predicted)')

plt.tight_layout()
plt.savefig('results/actual_vs_predicted_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/actual_vs_predicted_comparison.png")

# 2. Overlay comparison
fig, ax = plt.subplots(figsize=(12, 10))

# Plot actual fires
actual_fire_cells = grid_coords[actual_binary == 1]
pred_fire_cells = grid_coords[pred_binary == 1]
overlap_cells = grid_coords[(actual_binary == 1) & (pred_binary == 1)]

# Background: all cells
ax.scatter(grid_coords[:, 1], grid_coords[:, 0], c='lightgray', s=5, alpha=0.3, label='No Fire')

# Actual fires (red)
if len(actual_fire_cells) > 0:
    ax.scatter(actual_fire_cells[:, 1], actual_fire_cells[:, 0], 
              c='red', s=20, alpha=0.6, marker='s', label='Actual Fire', edgecolors='darkred', linewidth=0.5)

# Predicted fires (blue)
if len(pred_fire_cells) > 0:
    ax.scatter(pred_fire_cells[:, 1], pred_fire_cells[:, 0], 
              c='blue', s=20, alpha=0.6, marker='^', label='Predicted Fire', edgecolors='darkblue', linewidth=0.5)

# Overlap (purple)
if len(overlap_cells) > 0:
    ax.scatter(overlap_cells[:, 1], overlap_cells[:, 0], 
              c='purple', s=30, alpha=0.8, marker='*', label='Correct Prediction', edgecolors='black', linewidth=0.5)

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Fire Spread: Actual vs Predicted (Overlay)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/overlay_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/overlay_comparison.png")

# 3. Error analysis
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Prediction error
error = np.abs(predictions - actuals)
scatter3 = axes[0].scatter(grid_coords[:, 1], grid_coords[:, 0], 
                          c=error, cmap='Reds', s=10, alpha=0.7, edgecolors='black', linewidth=0.3)
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
axes[0].set_title('Prediction Error (|Predicted - Actual|)')
plt.colorbar(scatter3, ax=axes[0], label='Absolute Error')

# Error by actual intensity
axes[1].scatter(actuals, error, alpha=0.5, s=10)
axes[1].set_xlabel('Actual Fire Intensity')
axes[1].set_ylabel('Absolute Error')
axes[1].set_title('Error vs Actual Intensity')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/error_analysis.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/error_analysis.png")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total grid cells: {len(grid_coords)}")
print(f"Cells with actual fires: {actual_binary.sum()} ({100*actual_binary.sum()/len(grid_coords):.2f}%)")
print(f"Cells with predicted fires: {pred_binary.sum()} ({100*pred_binary.sum()/len(grid_coords):.2f}%)")
print(f"Correct predictions: {(pred_binary == actual_binary).sum()} ({100*(pred_binary == actual_binary).sum()/len(grid_coords):.2f}%)")
print(f"\nVisualizations saved to results/")
print("\nNext: Use these to identify where model succeeds/fails!")

