# Complete End-to-End Guide: Wildfire Spread Prediction with Multi-Modal GNN

## Overview

This guide walks you through the complete workflow for building a Spatio-Temporal Wildfire Spread Prediction system using Multi-Modal Graph Neural Networks.

---

## Phase 1: Environment Setup

### 1.1 Install Dependencies

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install required packages
pip install -r requirements.txt
```

**Key Libraries:**

- `torch` & `torch-geometric`: For GNN implementation
- `pandas`, `numpy`: Data processing
- `scikit-learn`: Evaluation metrics
- `matplotlib`, `seaborn`: Visualization
- `jupyter`: For notebooks

### 1.2 Verify Installation

```python
import torch
import torch_geometric
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch Geometric: {torch_geometric.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
```

---

## Phase 2: Data Understanding & Exploration

### 2.1 Load and Explore Data

**File: `notebooks/01_data_exploration.ipynb`**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
fire = pd.read_csv("../fire_data.csv")
weather = pd.read_csv("../weather_data.csv")
topo = pd.read_csv("../topo_data_cleaned.csv")

# Basic statistics
print("Fire Data:")
print(fire.describe())
print(f"\nShape: {fire.shape}")
print(f"\nColumns: {fire.columns.tolist()}")

print("\nWeather Data:")
print(weather.describe())

print("\nTopo Data:")
print(topo.describe())
```

### 2.2 Key Insights to Gather

1. **Temporal Distribution**: When do fires occur? (time of day, season)
2. **Spatial Distribution**: Where do fires occur? (geographic clusters)
3. **Feature Distributions**: Ranges, outliers, missing values
4. **Correlations**: Which features correlate with fire intensity (FRP)?

### 2.3 Visualizations

- Temporal distribution of fires
- Spatial heatmap of fire locations
- Feature correlation matrix
- Weather patterns over time

---

## Phase 3: Data Preprocessing

### 3.1 Use the DataLoader Class

**File: `notebooks/02_data_preprocessing.ipynb`**

```python
import sys
sys.path.append('../src')
from data_loader import WildfireDataLoader

# Initialize loader
loader = WildfireDataLoader(
    fire_path="../fire_data.csv",
    weather_path="../weather_data.csv",
    topo_path="../topo_data_cleaned.csv"
)

# Load and merge data
df = loader.merge_data()
print(f"Combined data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Create temporal and spatial features
df = loader.prepare_features(include_temporal=True, include_spatial=True)

# Get feature groups
feature_groups = loader.get_feature_columns(df)
print("\nFeature Groups:")
for group, cols in feature_groups.items():
    print(f"{group}: {cols}")
```

### 3.2 Handle Missing Values

```python
# Check missing values
missing = df.isna().sum()
print("Missing values:")
print(missing[missing > 0])

# Fill missing values (example strategies)
# Option 1: Forward fill for temporal data
df = df.sort_values('time')
df = df.fillna(method='ffill')

# Option 2: Fill with median/mean
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)
```

### 3.3 Feature Engineering

- **Temporal Features**: hour, day_of_year, cyclical encodings
- **Spatial Features**: Normalized coordinates
- **Derived Features**: Fire spread rate, weather trends

---

## Phase 4: Graph Construction

### 4.1 Build Spatial-Temporal Graph

**File: `notebooks/03_graph_construction.ipynb`**

```python
import sys
sys.path.append('../src')
from graph_builder import SpatialTemporalGraphBuilder
from torch_geometric.data import Data

# Initialize graph builder
graph_builder = SpatialTemporalGraphBuilder(
    spatial_threshold=0.05,  # ~5.5 km in degrees
    temporal_window=1  # 1 hour
)

# Build graph
graph = graph_builder.build_graph(
    df,
    feature_groups,
    include_temporal=True,
    include_spatial=True
)

print(f"Graph nodes: {graph.x.shape[0]}")
print(f"Graph features: {graph.x.shape[1]}")
print(f"Graph edges: {graph.edge_index.shape[1]}")
```

### 4.2 Understand Graph Structure

```python
# Visualize graph (if networkx available)
from src.utils import visualize_graph_structure
visualize_graph_structure(graph, title="Wildfire Spatial-Temporal Graph")
```

### 4.3 Create Targets

```python
# Create prediction targets (future fire spread)
target_locations, target_intensities = graph_builder.create_targets(
    df,
    prediction_horizon=1  # Predict 1 hour ahead
)

# Add targets to graph
graph.y = target_intensities  # or target_locations for binary classification
```

### 4.4 Split Data Temporally

```python
# Split into train/val/test (temporal split, not random!)
train_df, val_df, test_df = graph_builder.split_temporal(
    df,
    train_ratio=0.7,
    val_ratio=0.15
)

# Build graphs for each split
train_graph = graph_builder.build_graph(train_df, feature_groups)
val_graph = graph_builder.build_graph(val_df, feature_groups)
test_graph = graph_builder.build_graph(test_df, feature_groups)
```

---

## Phase 5: Model Development

### 5.1 Choose Model Architecture

**File: `notebooks/04_model_training.ipynb`**

**Option 1: Multi-Modal GCN (Recommended for starters)**

```python
import sys
sys.path.append('../src')
from models import MultiModalGCN
import torch

# Get input dimension
input_dim = train_graph.x.shape[1]

# Initialize model
model = MultiModalGCN(
    input_dim=input_dim,
    hidden_dim=64,
    num_layers=3,
    output_dim=1,  # For regression (fire intensity)
    dropout=0.2
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
```

**Option 2: Graph Attention Network (GAT)**

```python
from models import GraphAttentionWildfire

model = GraphAttentionWildfire(
    input_dim=input_dim,
    hidden_dim=64,
    num_layers=3,
    num_heads=4,
    output_dim=1,
    dropout=0.2
)
```

**Option 3: Temporal GCN (for time-series)**

```python
from models import TemporalGCN

model = TemporalGCN(
    input_dim=input_dim,
    hidden_dim=64,
    num_gcn_layers=2,
    lstm_hidden=64,
    num_lstm_layers=2,
    output_dim=1,
    dropout=0.2
)
```

### 5.2 Prepare Data Loaders

```python
from torch_geometric.loader import DataLoader

# Create data loaders
train_loader = DataLoader([train_graph], batch_size=1, shuffle=False)
val_loader = DataLoader([val_graph], batch_size=1, shuffle=False)
test_loader = DataLoader([test_graph], batch_size=1, shuffle=False)
```

### 5.3 Train Model

```python
from trainer import WildfireTrainer

# Initialize trainer
trainer = WildfireTrainer(
    model=model,
    learning_rate=0.001,
    weight_decay=1e-5
)

# Train
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    task='regression',  # or 'classification'
    save_path='../models/best_model.pt',
    patience=10
)
```

### 5.4 Monitor Training

```python
# Plot training history
from src.utils import plot_training_history

plot_training_history(
    trainer.train_losses,
    trainer.val_losses,
    save_path='../results/training_history.png'
)
```

---

## Phase 6: Evaluation & Prediction

### 6.1 Evaluate on Test Set

**File: `notebooks/05_evaluation.ipynb`**

```python
# Load best model
checkpoint = torch.load('../models/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Make predictions
trainer = WildfireTrainer(model)
predictions = trainer.predict(test_loader)

# Get actual values
test_targets = test_graph.y.numpy()

# Compute metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(test_targets, predictions)
rmse = np.sqrt(mean_squared_error(test_targets, predictions))
r2 = r2_score(test_targets, predictions)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
```

### 6.2 Visualize Results

```python
from src.utils import plot_predictions_vs_actual, create_spatial_visualization

# Predictions vs actual
plot_predictions_vs_actual(
    predictions,
    test_targets,
    title="Fire Intensity Predictions",
    save_path='../results/predictions_vs_actual.png'
)

# Spatial visualization
create_spatial_visualization(
    test_df,
    predictions=predictions,
    title="Predicted Fire Spread",
    save_path='../results/spatial_predictions.png'
)
```

### 6.3 Spatial Metrics

```python
from src.utils import compute_spatial_metrics

spatial_metrics = compute_spatial_metrics(
    predictions,
    test_targets,
    test_df[['latitude', 'longitude']].values,
    threshold=0.05
)

print("Spatial Metrics:")
print(spatial_metrics)
```

---

## Phase 7: Advanced Techniques

### 7.1 Multi-Modal Fusion

If you have separate modality features:

```python
from models import MultiModalFusionGNN

# Get feature dimensions for each modality
fire_dim = len(feature_groups.get('fire', []))
weather_dim = len(feature_groups.get('weather', []))
terrain_dim = len(feature_groups.get('terrain', []))

model = MultiModalFusionGNN(
    fire_dim=fire_dim,
    weather_dim=weather_dim,
    terrain_dim=terrain_dim,
    hidden_dim=64,
    num_layers=3,
    output_dim=1,
    fusion_method='attention'  # or 'concat', 'add'
)
```

### 7.2 Hyperparameter Tuning

```python
# Example: Grid search for learning rate
learning_rates = [0.0001, 0.001, 0.01]
best_lr = None
best_val_loss = float('inf')

for lr in learning_rates:
    model = MultiModalGCN(input_dim=input_dim, hidden_dim=64)
    trainer = WildfireTrainer(model, learning_rate=lr)
    trainer.train(train_loader, val_loader, num_epochs=10)

    val_loss, _ = trainer.validate(val_loader)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_lr = lr

print(f"Best learning rate: {best_lr}")
```

### 7.3 Handling Large Graphs

If graphs are too large:

```python
# Option 1: Subgraph sampling
from torch_geometric.utils import subgraph

# Sample nodes
num_nodes = 1000
sampled_nodes = torch.randperm(graph.x.shape[0])[:num_nodes]
sub_edge_index, sub_edge_attr = subgraph(sampled_nodes, graph.edge_index)
sub_x = graph.x[sampled_nodes]

# Option 2: Batch processing
# Split graph into smaller subgraphs by time or space
```

---

## Phase 8: Deployment & Production

### 8.1 Save Complete Pipeline

```python
import pickle

# Save preprocessor
with open('../models/preprocessor.pkl', 'wb') as f:
    pickle.dump({
        'feature_groups': feature_groups,
        'graph_builder': graph_builder
    }, f)

# Save model (already saved during training)
```

### 8.2 Prediction Function

```python
def predict_fire_spread(fire_data, weather_data, topo_data, model_path):
    """Predict fire spread for new data."""
    # Load preprocessor
    with open('../models/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)

    # Load model
    model = MultiModalGCN(input_dim=...)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Preprocess
    loader = WildfireDataLoader(...)
    df = loader.merge_data()
    df = loader.prepare_features()

    # Build graph
    graph = preprocessor['graph_builder'].build_graph(df, feature_groups)

    # Predict
    with torch.no_grad():
        predictions = model(graph.x, graph.edge_index)

    return predictions.numpy()
```

---

## Common Issues & Solutions

### Issue 1: Out of Memory

- **Solution**: Reduce batch size, use subgraph sampling, or use CPU

### Issue 2: Poor Performance

- **Solution**:
  - Increase model capacity (hidden_dim, num_layers)
  - Tune hyperparameters
  - Add more features
  - Check data quality

### Issue 3: Graph Too Sparse/Dense

- **Solution**: Adjust `spatial_threshold` and `temporal_window` in graph builder

### Issue 4: Missing Topographic Data

- **Solution**: Use interpolation or remove terrain features if not critical

---

## Next Steps

1. **Experiment with architectures**: Try different GNN variants
2. **Feature engineering**: Create domain-specific features
3. **Ensemble methods**: Combine multiple models
4. **Real-time prediction**: Deploy for live fire monitoring
5. **Uncertainty quantification**: Add confidence intervals to predictions

---

## Resources

- **PyTorch Geometric Docs**: https://pytorch-geometric.readthedocs.io/
- **GNN Papers**:
  - Graph Convolutional Networks (Kipf & Welling)
  - Graph Attention Networks (Veličković et al.)
- **Wildfire Prediction Literature**: Search for "wildfire spread prediction GNN"

---

## Project Checklist

- [ ] Environment setup complete
- [ ] Data exploration done
- [ ] Data preprocessing complete
- [ ] Graph construction working
- [ ] Model training successful
- [ ] Evaluation metrics computed
- [ ] Visualizations created
- [ ] Model saved and documented
- [ ] Results analyzed and interpreted

Good luck with your project! 🔥🌲
