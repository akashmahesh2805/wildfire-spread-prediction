# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install packages
pip install -r requirements.txt
```

### Step 2: Run the Complete Workflow

```bash
# Option A: Run as Python script
python notebooks/complete_workflow_example.py

# Option B: Open in Jupyter
jupyter notebook notebooks/complete_workflow_example.py
```

### Step 3: Check Results

After running, you'll find:

- **Model**: `models/best_model.pt`
- **Results**: `results/` folder with plots and metrics

---

## 📚 Understanding the Project Structure

```
wildfire-spread-prediction/
├── data/                    # Your CSV files (already here)
│   ├── fire_data.csv
│   ├── weather_data.csv
│   └── topo_data_cleaned.csv
│
├── src/                     # Core implementation
│   ├── data_loader.py      # Load and preprocess data
│   ├── graph_builder.py    # Build spatial-temporal graphs
│   ├── models.py           # GNN model architectures
│   ├── trainer.py          # Training utilities
│   └── utils.py            # Helper functions
│
├── notebooks/              # Jupyter notebooks
│   └── complete_workflow_example.py  # Full workflow
│
├── models/                 # Saved models (created after training)
├── results/                # Plots and metrics (created after training)
│
├── README.md               # Project overview
├── END_TO_END_GUIDE.md    # Detailed guide
└── requirements.txt        # Dependencies
```

---

## 🎯 Key Concepts

### 1. **Spatial-Temporal Graph**

- **Nodes** = Fire events at specific locations and times
- **Spatial Edges** = Connect nearby fire locations
- **Temporal Edges** = Connect fire events across time

### 2. **Multi-Modal Features**

- **Fire**: brightness, FRP, confidence
- **Weather**: temperature, humidity, wind, precipitation
- **Terrain**: elevation, slope, aspect, vegetation

### 3. **Prediction Task**

- **Input**: Current fire state + weather + terrain
- **Output**: Future fire intensity (1 hour ahead)

---

## 🔧 Customization

### Change Model Architecture

```python
from models import GraphAttentionWildfire  # Instead of MultiModalGCN

model = GraphAttentionWildfire(
    input_dim=input_dim,
    hidden_dim=128,  # Increase capacity
    num_layers=4,
    num_heads=8
)
```

### Adjust Graph Structure

```python
graph_builder = SpatialTemporalGraphBuilder(
    spatial_threshold=0.1,   # Larger = more connections
    temporal_window=2        # Connect events 2 hours apart
)
```

### Modify Training

```python
trainer = WildfireTrainer(
    model=model,
    learning_rate=0.0001,  # Lower learning rate
    weight_decay=1e-4
)
```

---

## 📊 Expected Results

After training, you should see:

- **Training/Validation Loss**: Decreasing over epochs
- **Test Metrics**:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² (Coefficient of Determination)
- **Visualizations**:
  - Training curves
  - Predictions vs actual
  - Spatial fire spread map

---

## 🐛 Troubleshooting

### Issue: "Out of Memory"

**Solution**: Reduce batch size or use CPU

```python
device = torch.device('cpu')  # Force CPU
```

### Issue: "Graph has no edges"

**Solution**: Increase spatial_threshold

```python
spatial_threshold=0.1  # Increase from 0.05
```

### Issue: "Poor model performance"

**Solutions**:

1. Increase model capacity (hidden_dim, num_layers)
2. Add more features
3. Tune hyperparameters
4. Check data quality

---

## 📖 Next Steps

1. **Read the detailed guide**: `END_TO_END_GUIDE.md`
2. **Experiment**: Try different models and hyperparameters
3. **Add features**: Create domain-specific features
4. **Visualize**: Explore the data and results
5. **Deploy**: Create prediction pipeline for new data

---

## 💡 Tips

- Start with the default settings, then experiment
- Monitor training loss to avoid overfitting
- Use temporal split (not random) for train/test
- Visualize your graphs to understand structure
- Save intermediate results for analysis

---

## 📞 Need Help?

1. Check `END_TO_END_GUIDE.md` for detailed explanations
2. Review code comments in `src/` modules
3. Experiment with the example script
4. Adjust parameters based on your data characteristics

Good luck! 🔥🌲
