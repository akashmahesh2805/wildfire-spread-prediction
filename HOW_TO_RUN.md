# How to Run the Grid-Based Model

## Quick Start

### Option 1: Using Batch File (Easiest)
```powershell
.\run_workflow.bat
```

### Option 2: Direct Python Command
```powershell
.\venv\Scripts\python.exe notebooks\complete_workflow_example.py
```

### Option 3: Activate Venv First
```powershell
.\venv\Scripts\Activate.ps1
python notebooks\complete_workflow_example.py
```

## What Happens When You Run

1. **Data Loading** (Step 1)
   - Loads fire_data.csv, weather_data.csv, topo_data_cleaned.csv
   - Merges all data sources
   - Creates temporal and spatial features

2. **Grid Creation** (Step 2)
   - Divides study area into grid cells (0.01° ≈ 1.1 km)
   - Assigns fire events to nearest grid cell
   - Aggregates features per cell

3. **Graph Building** (Step 2)
   - Creates nodes: one per grid cell
   - Creates spatial edges: connects adjacent cells
   - Creates temporal edges: connects same cell across time

4. **Model Training** (Step 5)
   - Trains Multi-Modal GCN
   - Saves best model

5. **Evaluation** (Step 6)
   - Tests on held-out data
   - Computes metrics (MAE, RMSE, R²)

6. **Visualization** (Step 7)
   - Training curves
   - Predictions vs actual
   - Spatial fire spread map

## Expected Output

```
STEP 1: Loading and Preprocessing Data
✓ Loaded combined data: (29208, 22)
✓ Created features: (29208, 32)

STEP 2: Building Spatial-Temporal Graph
  Created grid with X cells
  Aggregated features: (X, 23)
  Spatial edges: Y
  Temporal edges: Z
✓ Train graph: X nodes, Y edges

STEP 3: Initializing Model
Input feature dimension: 23
✓ Model initialized: 16,129 parameters

STEP 5: Training Model
Epoch 1/50
Train Loss: X.XXXX, Val Loss: X.XXXX
...

STEP 6: Evaluating Model
Test Set Metrics:
  MAE:  X.XXXX
  RMSE: X.XXXX
  R²:   X.XXXX

STEP 7: Visualizing Results
✓ All visualizations saved to results/
```

## Troubleshooting

### Issue: ModuleNotFoundError
**Solution**: Make sure you're in the project root directory and venv is activated.

### Issue: Out of Memory
**Solution**: The grid-based approach is more memory efficient, but if issues persist:
- Reduce `grid_size` in `GridBasedGraphBuilder` (e.g., 0.02 instead of 0.01)
- This creates fewer, larger cells

### Issue: Slow Training
**Solution**: This is normal. Grid-based should be faster than event-based, but training still takes time.

