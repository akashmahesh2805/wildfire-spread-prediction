# How to Read and Evaluate Fire Spread Predictions

## 📊 Reading the Prediction Map

### What You're Seeing

The spatial visualization shows **predicted fire intensity** across your study area:

1. **Grid Cells**: Each small square is a grid cell (~1.1 km × 1.1 km)
2. **Color Scale**: 
   - **Light yellow/beige** = Low predicted intensity (0.2-0.4)
   - **Orange** = Moderate intensity (0.6-1.0)
   - **Dark red/maroon** = High intensity (1.0-1.6+)
3. **Hotspots**: Dark red areas are where the model predicts fires will be most intense

### Coordinates
- **X-axis (Longitude)**: -119.1 to -117.5 (West to East)
- **Y-axis (Latitude)**: 33.7 to 34.9 (South to North)
- Each point = a grid cell center

## 🔍 Comparing Predictions to Actual Fires

### Current Status: **You're Only Seeing Predictions**

Right now, the visualization shows **only predictions**, not actual fire paths. To compare:

1. **You need actual fire data** for the same time period
2. **Overlay actual fire perimeters** on the prediction map
3. **Calculate overlap** between predicted and actual areas

## 🧪 Testing on a Known Fire

Yes! You can test on a fire whose path you know. Here's how:

### Step 1: Identify a Known Fire Event

From your data, find a specific fire event:
- Pick a fire that occurred at a known time
- Note its actual spread path/locations
- Use data from BEFORE that fire to predict it

### Step 2: Create Test Scenario

```python
# Example: Predict a specific fire event
# 1. Use data up to time T
# 2. Predict fire at time T+1
# 3. Compare to actual fire at time T+1
```

### Step 3: Visualize Comparison

Create side-by-side or overlaid maps showing:
- **Predicted hotspots** (from model)
- **Actual fire locations** (from your data)
- **Overlap areas** (where both agree)

## 📈 Current Model Performance

### Metrics from Your Run:
- **R² = 0.0109**: Very low (model explains only 1% of variance)
- **MAE = 0.4940**: Mean absolute error
- **RMSE = 8.5110**: High root mean squared error

### What This Means:
- **Model is learning** (loss decreasing)
- **But performance is poor** (R² near 0)
- **Overfitting** (val loss increasing after epoch 12)

### Why Performance is Low:
1. **Target creation** might not match actual fire dynamics
2. **Many zero targets** (most cells have no fire)
3. **Feature scaling** might need adjustment
4. **Model capacity** might be insufficient

## ✅ What's Left to Do

### Priority 1: Model Validation & Comparison

1. **Create comparison visualizations**
   - Actual vs predicted side-by-side
   - Overlay actual fire perimeters
   - Calculate spatial accuracy metrics

2. **Test on known fires**
   - Pick specific fire events
   - Predict them using historical data
   - Compare predictions to actual paths

### Priority 2: Improve Model Performance

1. **Better target creation**
   - Predict fire spread, not just continuation
   - Handle class imbalance (many zeros)

2. **Feature engineering**
   - Fire spread rate
   - Weather gradients
   - Historical fire patterns

3. **Model improvements**
   - Try different architectures (GAT, TemporalGCN)
   - Add edge weights
   - Hyperparameter tuning

### Priority 3: Evaluation Metrics

1. **Spatial accuracy**
   - How close are predictions to actual locations?
   - Overlap area between predicted and actual

2. **Temporal accuracy**
   - Correct prediction timing?
   - Early/late predictions?

3. **Fire size/area**
   - Predicted fire area vs actual

## 🎯 Next Steps Checklist

- [ ] Create actual vs predicted comparison script
- [ ] Test on a known fire event
- [ ] Calculate spatial overlap metrics
- [ ] Improve target creation
- [ ] Try different model architectures
- [ ] Add better evaluation metrics
- [ ] Create animated fire spread visualization
- [ ] Document findings and improvements

## 📝 Project Status

**Current State:**
- ✅ Complete pipeline working
- ✅ Grid-based graph structure
- ✅ Model training successfully
- ✅ Predictions generated
- ⚠️ Model performance needs improvement (R² = 0.01)
- ⚠️ No comparison with actual fires yet

**For Research/Paper:**
- Need better performance (R² > 0.3 minimum)
- Need validation against actual fires
- Need comparison with baselines
- Need ablation studies

**For Demonstration:**
- ✅ Working end-to-end system
- ✅ Good visualizations
- ✅ Complete documentation

