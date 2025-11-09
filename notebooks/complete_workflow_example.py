"""
Complete workflow example for wildfire spread prediction.
This script demonstrates the end-to-end process.
Run this in a Jupyter notebook or as a Python script.
"""

import sys
import os

# Add src directory to path (works from both root and notebooks directory)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in globals() else os.getcwd()
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import pandas as pd
import numpy as np
import torch
from torch_geometric.loader import DataLoader

# Import our modules
from data_loader import WildfireDataLoader
from grid_graph_builder import GridBasedGraphBuilder  # Using grid-based approach
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
# Get data paths (works from both root and notebooks directory)
data_dir = project_root
loader = WildfireDataLoader(
    fire_path=os.path.join(data_dir, "fire_data.csv"),
    weather_path=os.path.join(data_dir, "weather_data.csv"),
    topo_path=os.path.join(data_dir, "topo_data_cleaned.csv")
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

# -> Change to nearest neighbour (neares 3-5 neighbours)
# Handle missing values
print("\nHandling missing values...")
df = df.sort_values('time')
df = df.ffill().fillna(0)  # Use ffill() instead of deprecated fillna(method='ffill')
print(f"✓ Missing values handled")

# ============================================================================
# STEP 2: Build Spatial-Temporal Graph
# ============================================================================

print("\n" + "=" * 60)
print("STEP 2: Building Spatial-Temporal Graph")
print("=" * 60)

# Initialize grid-based graph builder
# Grid-based approach: divides area into cells, more structured
graph_builder = GridBasedGraphBuilder(
    grid_size=0.01,  # ~1.1 km grid cells
    temporal_window=1  # 1 hour
    # -> Change to 6 hrs
)

# Split data temporally
train_df, val_df, test_df = graph_builder.split_temporal(
    df,
    train_ratio=0.7,
    val_ratio=0.15
)
print(f"✓ Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# Build graphs for each split
# Grid-based: creates nodes for grid cells, not individual events
print("\nBuilding graphs...")
print("Note: Using grid-based approach (cells, not events)")

# Build grid-based graphs
train_graph = graph_builder.build_graph(
    train_df, 
    feature_groups,
    include_temporal=True,
    include_spatial=True
)

val_graph = graph_builder.build_graph(
    val_df, 
    feature_groups,
    include_temporal=True,
    include_spatial=True
)

test_graph = graph_builder.build_graph(
    test_df, 
    feature_groups,
    include_temporal=True,
    include_spatial=True
)

print(f"✓ Train graph: {train_graph.x.shape[0]} nodes, {train_graph.edge_index.shape[1]} edges")
print(f"✓ Val graph: {val_graph.x.shape[0]} nodes, {val_graph.edge_index.shape[1]} edges")
print(f"✓ Test graph: {test_graph.x.shape[0]} nodes, {test_graph.edge_index.shape[1]} edges")

# Create targets (future fire intensity for each grid cell)
print("\nCreating prediction targets...")

# IMPORTANT: Each graph already has its grid created during build_graph()
# We need to use the same grid for target creation
# The grid is stored in graph.grid_cells

# For train: use the grid from train_graph
# Set the grid first, then assign events
graph_builder.grid_cells = train_graph.grid_cells
graph_builder.cell_to_index = train_graph.cell_to_index
train_df_with_grid = graph_builder.assign_events_to_grid(train_df)
train_targets = graph_builder.create_targets(train_df_with_grid, prediction_horizon=1)

# For val: use the grid from val_graph
graph_builder.grid_cells = val_graph.grid_cells
graph_builder.cell_to_index = val_graph.cell_to_index
val_df_with_grid = graph_builder.assign_events_to_grid(val_df)
val_targets = graph_builder.create_targets(val_df_with_grid, prediction_horizon=1)

# For test: use the grid from test_graph
graph_builder.grid_cells = test_graph.grid_cells
graph_builder.cell_to_index = test_graph.cell_to_index
test_df_with_grid = graph_builder.assign_events_to_grid(test_df)
test_targets = graph_builder.create_targets(test_df_with_grid, prediction_horizon=1)

# Verify sizes match
assert len(train_targets) == train_graph.x.shape[0], f"Train: {len(train_targets)} targets != {train_graph.x.shape[0]} nodes"
assert len(val_targets) == val_graph.x.shape[0], f"Val: {len(val_targets)} targets != {val_graph.x.shape[0]} nodes"
assert len(test_targets) == test_graph.x.shape[0], f"Test: {len(test_targets)} targets != {test_graph.x.shape[0]} nodes"

# Assign targets to graphs
train_graph.y = train_targets
val_graph.y = val_targets
test_graph.y = test_targets
print(f"✓ Targets created: Train={len(train_targets)}, Val={len(val_targets)}, Test={len(test_targets)}")

# ============================================================================
# STEP 3: Initialize Model
# ============================================================================

print("\n" + "=" * 60)
print("STEP 3: Initializing Model")
print("=" * 60)

# Get input dimension
input_dim = train_graph.x.shape[1]
print(f"Input feature dimension: {input_dim}")

# -> Change to Graph Wave Net
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
models_dir = os.path.join(project_root, 'models')
os.makedirs(models_dir, exist_ok=True)

# Train
print("\nStarting training...")
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    task='regression',
    save_path=os.path.join(models_dir, 'best_model.pt'),
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
# Note: weights_only=False needed because checkpoint contains numpy scalars in metrics
checkpoint = torch.load(os.path.join(models_dir, 'best_model.pt'), weights_only=False)
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
results_dir = os.path.join(project_root, 'results')
os.makedirs(results_dir, exist_ok=True)

# Plot training history
print("\nPlotting training history...")
plot_training_history(
    trainer.train_losses,
    trainer.val_losses,
    save_path=os.path.join(results_dir, 'training_history.png')
)

# Plot predictions vs actual
print("Plotting predictions vs actual...")
plot_predictions_vs_actual(
    predictions,
    actuals,
    title="Fire Intensity Predictions",
    save_path=os.path.join(results_dir, 'predictions_vs_actual.png')
)

# Spatial visualization (using grid cell centers)
print("Creating spatial visualization...")
# Create dataframe from grid cells for visualization
grid_df = pd.DataFrame({
    'latitude': test_graph.grid_cells[:, 0],
    'longitude': test_graph.grid_cells[:, 1]
})
create_spatial_visualization(
    grid_df,
    predictions=predictions,
    title="Predicted Fire Spread (Grid-Based)",
    save_path=os.path.join(results_dir, 'spatial_predictions.png')
)

print(f"\n✓ All visualizations saved to {results_dir}/")

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
print(f"  • Model: {os.path.join(models_dir, 'best_model.pt')}")
print(f"  • Results: {results_dir}/")
print("\nNext steps:")
print("  1. Experiment with different model architectures")
print("  2. Tune hyperparameters")
print("  3. Add more features")
print("  4. Try different graph construction methods")

