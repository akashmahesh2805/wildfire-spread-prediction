# Critical Issues Found & How to Fix Them

## 🚨 Problems Identified from Comparison

### 1. **Scale Mismatch** (CRITICAL)
- **Actual fires**: 0-500 intensity
- **Predicted fires**: 0.2-1.6 intensity
- **Problem**: Model is predicting on completely wrong scale!
- **Cause**: Target aggregation loses actual FRP values

### 2. **False Positives** (HIGH)
- Model predicts fires where none exist
- Multiple false hotspots across the map
- **Problem**: Model is too "optimistic"
- **Cause**: Class imbalance (95% zeros) - model learns to predict small values everywhere

### 3. **Poor Localization** (HIGH)
- **Actual**: Concentrated, high-intensity spots
- **Predicted**: Diffuse, spread-out areas
- **Problem**: Can't pinpoint exact fire locations
- **Cause**: Grid aggregation + lack of spatial attention

### 4. **Intensity Underestimation** (MEDIUM)
- **Actual**: 300-400 intensity
- **Predicted**: 0.8-1.0 intensity
- **Problem**: Severely underestimates fire intensity
- **Cause**: Scale mismatch + normalization issues

## ✅ Solutions

### Fix 1: Use Actual FRP Values (Not Aggregated)

**Current Problem:**
```python
# Current: Aggregates FRP (loses scale)
target = future_fires['frp'].max()  # But this is after aggregation
```

**Fix:**
```python
# Use actual FRP from original events
target = df[df['time'] == future_time]['frp'].max()  # Direct from events
```

### Fix 2: Weighted Loss for Class Imbalance

**Current Problem:**
- 95% of targets are zero
- Model learns to predict zeros

**Fix:**
```python
# Weight non-zero targets more heavily
class_weights = [weight_for_zero, weight_for_non_zero]
loss = weighted_mse_loss(predictions, targets, weights)
```

### Fix 3: Better Target Definition

**Current Problem:**
- Predicts continuation, not spread
- Doesn't capture fire dynamics

**Fix:**
```python
# Predict fire spread to nearby cells (spatial radius)
# Use actual FRP values
# Handle multiple time steps properly
```

### Fix 4: Post-Processing

**Add thresholding:**
```python
# Remove isolated predictions
# Apply spatial smoothing
# Remove predictions below threshold
```

## 🚀 Run Improved Version

```powershell
# Run improved workflow with all fixes
.\venv\Scripts\python.exe improved_workflow.py
```

This will:
1. ✅ Use actual FRP values (fix scale)
2. ✅ Apply weighted loss (fix false positives)
3. ✅ Use GAT model (better localization)
4. ✅ Proper normalization (fix intensity)

## 📊 Expected Improvements

After fixes:
- **Scale**: Predictions should match actual FRP scale (0-500 range)
- **False Positives**: Should decrease significantly
- **Localization**: More concentrated predictions
- **R²**: Should improve from 0.01 to > 0.3

## 🎯 Immediate Actions

1. **Run improved workflow**:
   ```powershell
   .\venv\Scripts\python.exe improved_workflow.py
   ```

2. **Compare results**:
   - Check new R² value
   - Look at new comparison visualizations
   - See if scale matches

3. **If still poor**:
   - Adjust spatial radius in target creation
   - Tune class weights
   - Try different grid sizes
   - Add post-processing

The main issue is **target creation** - it's not using actual FRP values properly. The improved workflow fixes this!

