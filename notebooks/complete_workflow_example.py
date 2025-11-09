"""
Complete workflow example for wildfire spread prediction.
This script demonstrates the end-to-end process.
Run this in a Jupyter notebook or as a Python script.
"""

import sys
import os
sys.path.append('../src')

import pandas as pd
import numpy as np
import torch
from torch_geometric.loader import DataLoader

# Import our modules
from data_loader import WildfireDataLoader
from graph_builder import SpatialTemporalGraphBuilder
from models import MultiModalGCN
from trainer import WildfireTrainer
from utils import plot_training_history, plot_predictions_vs_actual, create_spatial_visualization

# ============================================================================
# STEP 1: Load and Preprocess Data
# ============================================================================

print("=" * 60)
print("STEP 1: Loading and Preprocessing Data")
print("=" * 60)

# Initialize data loader
loader = WildfireDataLoader(
    fire_path="../fire_data.csv",
    weather_path="../weather_data.csv",
    topo_path="../topo_data_cleaned.csv"
)

# Load and merge all data
df = loader.merge_data()
print(f"✓ Loaded combined data: {df.shape}")

# Create temporal and spatial features
df = loader.prepare_features(include_temporal=True, include_spatial=True)
print(f"✓ Created features: {df.shape}")

# Get feature groups
feature_groups = loader.get_feature_columns(df)
print(f"\nFeature Groups:")
for group, cols in feature_groups.items():
    print(f"  {group}: {len(cols)} features")

# Handle missing values
print("\nHandling missing values...")
df = df.sort_values('time')
df = df.fillna(method='ffill').fillna(0)
print(f"✓ Missing values handled")

# ============================================================================
# STEP 2: Build Spatial-Temporal Graph
# ============================================================================

print("\n" + "=" * 60)
print("STEP 2: Building Spatial-Temporal Graph")
print("=" * 60)

# Initialize graph builder
graph_builder = SpatialTemporalGraphBuilder(
    spatial_threshold=0.05,  # ~5.5 km
    temporal_window=1  # 1 hour
)

# Split data temporally
train_df, val_df, test_df = graph_builder.split_temporal(
    df,
    train_ratio=0.7,
    val_ratio=0.15
)
print(f"✓ Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# Build graphs for each split
print("\nBuilding graphs...")
train_graph = graph_builder.build_graph(train_df, feature_groups)
val_graph = graph_builder.build_graph(val_df, feature_groups)
test_graph = graph_builder.build_graph(test_df, feature_groups)

print(f"✓ Train graph: {train_graph.x.shape[0]} nodes, {train_graph.edge_index.shape[1]} edges")
print(f"✓ Val graph: {val_graph.x.shape[0]} nodes, {val_graph.edge_index.shape[1]} edges")
print(f"✓ Test graph: {test_graph.x.shape[0]} nodes, {test_graph.edge_index.shape[1]} edges")

# Create targets (future fire intensity)
print("\nCreating prediction targets...")
train_targets, _ = graph_builder.create_targets(train_df, prediction_horizon=1)
val_targets, _ = graph_builder.create_targets(val_df, prediction_horizon=1)
test_targets, _ = graph_builder.create_targets(test_df, prediction_horizon=1)

# Add targets to graphs
train_graph.y = train_targets
val_graph.y = val_targets
test_graph.y = test_targets
print(f"✓ Targets created")

# ============================================================================
# STEP 3: Initialize Model
# ============================================================================

print("\n" + "=" * 60)
print("STEP 3: Initializing Model")
print("=" * 60)

# Get input dimension
input_dim = train_graph.x.shape[1]
print(f"Input feature dimension: {input_dim}")

# Initialize model
model = MultiModalGCN(
    input_dim=input_dim,
    hidden_dim=64,
    num_layers=3,
    output_dim=1,  # Regression: predict fire intensity
    dropout=0.2
)

num_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model initialized: {num_params:,} parameters")

# ============================================================================
# STEP 4: Prepare Data Loaders
# ============================================================================

print("\n" + "=" * 60)
print("STEP 4: Preparing Data Loaders")
print("=" * 60)

# Create data loaders
train_loader = DataLoader([train_graph], batch_size=1, shuffle=False)
val_loader = DataLoader([val_graph], batch_size=1, shuffle=False)
test_loader = DataLoader([test_graph], batch_size=1, shuffle=False)
print("✓ Data loaders created")

# ============================================================================
# STEP 5: Train Model
# ============================================================================

print("\n" + "=" * 60)
print("STEP 5: Training Model")
print("=" * 60)

# Initialize trainer
trainer = WildfireTrainer(
    model=model,
    learning_rate=0.001,
    weight_decay=1e-5
)

# Create models directory
os.makedirs('../models', exist_ok=True)

# Train
print("\nStarting training...")
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    task='regression',
    save_path='../models/best_model.pt',
    patience=10
)

print("\n✓ Training complete!")

# ============================================================================
# STEP 6: Evaluate Model
# ============================================================================

print("\n" + "=" * 60)
print("STEP 6: Evaluating Model")
print("=" * 60)

# Load best model
checkpoint = torch.load('../models/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
print("✓ Loaded best model")

# Make predictions
print("\nMaking predictions...")
predictions = trainer.predict(test_loader)
actuals = test_graph.y.numpy()

# Compute metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))
r2 = r2_score(actuals, predictions)

print(f"\nTest Set Metrics:")
print(f"  MAE:  {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²:   {r2:.4f}")

# ============================================================================
# STEP 7: Visualize Results
# ============================================================================

print("\n" + "=" * 60)
print("STEP 7: Visualizing Results")
print("=" * 60)

# Create results directory
os.makedirs('../results', exist_ok=True)

# Plot training history
print("\nPlotting training history...")
plot_training_history(
    trainer.train_losses,
    trainer.val_losses,
    save_path='../results/training_history.png'
)

# Plot predictions vs actual
print("Plotting predictions vs actual...")
plot_predictions_vs_actual(
    predictions,
    actuals,
    title="Fire Intensity Predictions",
    save_path='../results/predictions_vs_actual.png'
)

# Spatial visualization
print("Creating spatial visualization...")
create_spatial_visualization(
    test_df,
    predictions=predictions,
    title="Predicted Fire Spread",
    save_path='../results/spatial_predictions.png'
)

print("\n✓ All visualizations saved to ../results/")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 60)
print("WORKFLOW COMPLETE!")
print("=" * 60)
print("\nSummary:")
print(f"  • Processed {len(df)} data points")
print(f"  • Built graphs with {train_graph.edge_index.shape[1]} edges")
print(f"  • Trained model with {num_params:,} parameters")
print(f"  • Achieved R² = {r2:.4f} on test set")
print(f"\nFiles saved:")
print(f"  • Model: ../models/best_model.pt")
print(f"  • Results: ../results/")
print("\nNext steps:")
print("  1. Experiment with different model architectures")
print("  2. Tune hyperparameters")
print("  3. Add more features")
print("  4. Try different graph construction methods")

