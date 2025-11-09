"""
Test the model on a specific known fire event.
Predicts a fire whose actual path is known, then compares.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader

from data_loader import WildfireDataLoader
from grid_graph_builder import GridBasedGraphBuilder
from models import MultiModalGCN
from trainer import WildfireTrainer

print("=" * 60)
print("TESTING ON KNOWN FIRE EVENT")
print("=" * 60)

# Load all data
loader = WildfireDataLoader()
df = loader.merge_data()
df = loader.prepare_features(include_temporal=True, include_spatial=True)
df = df.sort_values('time').ffill().fillna(0)

feature_groups = loader.get_feature_columns(df)

# Find a specific fire event to test
print("\nFinding a test fire event...")
# Get fires with high intensity
high_intensity_fires = df[df['frp'] > df['frp'].quantile(0.9)].sort_values('time')

if len(high_intensity_fires) > 0:
    # Pick a fire event
    test_fire = high_intensity_fires.iloc[0]
    test_time = test_fire['time']
    test_location = (test_fire['latitude'], test_fire['longitude'])
    
    print(f"Selected test fire:")
    print(f"  Time: {test_time}")
    print(f"  Location: ({test_location[0]:.4f}, {test_location[1]:.4f})")
    print(f"  FRP: {test_fire['frp']:.2f}")
    
    # Get data BEFORE this fire (for prediction)
    prediction_time = test_time - pd.Timedelta(hours=1)
    historical_data = df[df['time'] <= prediction_time]
    
    # Get actual fire data at test_time (for comparison)
    actual_fire_data = df[df['time'] == test_time]
    
    print(f"\nHistorical data (for prediction): {len(historical_data)} events")
    print(f"Actual fire data (at test time): {len(actual_fire_data)} events")
    
    if len(historical_data) > 100 and len(actual_fire_data) > 0:
        # Build graph from historical data
        print("\nBuilding prediction graph...")
        graph_builder = GridBasedGraphBuilder(grid_size=0.01, temporal_window=1)
        pred_graph = graph_builder.build_graph(historical_data, feature_groups)
        
        # Load trained model
        print("Loading model...")
        model = MultiModalGCN(input_dim=pred_graph.x.shape[1], hidden_dim=64, num_layers=3, output_dim=1, dropout=0.2)
        checkpoint = torch.load('models/best_model.pt', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Make prediction
        print("Making prediction...")
        pred_loader = DataLoader([pred_graph], batch_size=1, shuffle=False)
        trainer = WildfireTrainer(model)
        predictions = trainer.predict(pred_loader)
        
        # Get actual fire locations at test time
        # Assign actual fires to same grid
        graph_builder.grid_cells = pred_graph.grid_cells
        graph_builder.cell_to_index = pred_graph.cell_to_index
        actual_with_grid = graph_builder.assign_events_to_grid(actual_fire_data)
        
        # Create actual intensity map
        actual_intensity = np.zeros(len(pred_graph.grid_cells))
        for idx, row in actual_with_grid.iterrows():
            cell_idx = row['grid_cell_idx']
            if cell_idx < len(actual_intensity):
                # Use max FRP if multiple fires in same cell
                if 'frp' in row:
                    actual_intensity[cell_idx] = max(actual_intensity[cell_idx], row['frp'])
        
        # Visualize comparison
        print("\nCreating comparison visualization...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        grid_coords = pred_graph.grid_cells
        
        # Predicted
        scatter1 = axes[0].scatter(grid_coords[:, 1], grid_coords[:, 0], 
                                  c=predictions, cmap='YlOrRd', s=10, alpha=0.7, 
                                  edgecolors='black', linewidth=0.2)
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        axes[0].set_title(f'Predicted Fire (at {test_time})')
        axes[0].scatter(test_location[1], test_location[0], c='blue', s=100, 
                       marker='*', edgecolors='black', linewidth=2, label='Test Fire Location')
        axes[0].legend()
        plt.colorbar(scatter1, ax=axes[0], label='Predicted Intensity')
        
        # Actual
        scatter2 = axes[1].scatter(grid_coords[:, 1], grid_coords[:, 0], 
                                  c=actual_intensity, cmap='YlOrRd', s=10, alpha=0.7, 
                                  edgecolors='black', linewidth=0.2)
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        axes[1].set_title(f'Actual Fire (at {test_time})')
        axes[1].scatter(test_location[1], test_location[0], c='blue', s=100, 
                       marker='*', edgecolors='black', linewidth=2, label='Test Fire Location')
        axes[1].legend()
        plt.colorbar(scatter2, ax=axes[1], label='Actual Intensity')
        
        # Difference
        diff = np.abs(predictions - actual_intensity)
        scatter3 = axes[2].scatter(grid_coords[:, 1], grid_coords[:, 0], 
                                  c=diff, cmap='Reds', s=10, alpha=0.7, 
                                  edgecolors='black', linewidth=0.2)
        axes[2].set_xlabel('Longitude')
        axes[2].set_ylabel('Latitude')
        axes[2].set_title('Prediction Error')
        plt.colorbar(scatter3, ax=axes[2], label='Absolute Error')
        
        plt.tight_layout()
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/known_fire_test.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: results/known_fire_test.png")
        
        # Calculate metrics
        print("\n" + "=" * 60)
        print("EVALUATION METRICS")
        print("=" * 60)
        
        # Binary classification
        pred_binary = (predictions > 0.5).astype(int)
        actual_binary = (actual_intensity > 0.5).astype(int)
        
        correct = (pred_binary == actual_binary).sum()
        total = len(predictions)
        
        print(f"\nBinary Classification:")
        print(f"  Correct predictions: {correct}/{total} ({100*correct/total:.2f}%)")
        print(f"  Predicted fires: {pred_binary.sum()} cells")
        print(f"  Actual fires: {actual_binary.sum()} cells")
        print(f"  Overlap: {(pred_binary & actual_binary).sum()} cells")
        
        # Continuous metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(actual_intensity, predictions)
        rmse = np.sqrt(mean_squared_error(actual_intensity, predictions))
        r2 = r2_score(actual_intensity, predictions)
        
        print(f"\nContinuous Metrics:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
        
        # Spatial accuracy
        if actual_binary.sum() > 0:
            # Find predicted fire cells closest to actual fires
            actual_fire_coords = grid_coords[actual_binary == 1]
            pred_fire_coords = grid_coords[pred_binary == 1]
            
            if len(pred_fire_coords) > 0 and len(actual_fire_coords) > 0:
                from scipy.spatial.distance import cdist
                distances = cdist(actual_fire_coords, pred_fire_coords)
                min_distances = distances.min(axis=1)
                
                print(f"\nSpatial Accuracy:")
                print(f"  Mean distance to nearest prediction: {min_distances.mean():.4f} degrees")
                print(f"  Max distance: {min_distances.max():.4f} degrees")
                print(f"  Cells within 0.05° (~5.5 km): {(min_distances <= 0.05).sum()}/{len(min_distances)}")
        
        print("\n" + "=" * 60)
        print("INTERPRETATION")
        print("=" * 60)
        print("""
This test shows how well the model predicts a specific known fire event.

Good signs:
  - High overlap between predicted and actual fire cells
  - Low spatial distance between predictions and actual fires
  - High R² and low MAE/RMSE

Areas for improvement:
  - If predictions are far from actual fires → improve spatial modeling
  - If predictions miss fires → improve target creation
  - If predictions are too spread out → reduce false positives
        """)
        
    else:
        print("Not enough data for this test. Try a different fire event.")
else:
    print("No high-intensity fires found in dataset.")

