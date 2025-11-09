# Graph Structure Explained

## 🎯 Current Approach: Event-Based Graph

### What Are Nodes?
**Each node = ONE fire detection event** at a specific location and time.

**Example:**
```
Node 1: Fire detected at (34.33°N, -118.52°W) on 2020-01-01 10:00
  - Features: brightness=298, FRP=0.57, temp=17.2°C, elevation=500m, ...
  
Node 2: Fire detected at (34.34°N, -118.51°W) on 2020-01-01 10:00
  - Features: brightness=299, FRP=1.03, temp=17.2°C, elevation=520m, ...
  
Node 3: Fire detected at (34.33°N, -118.52°W) on 2020-01-01 11:00
  - Features: brightness=305, FRP=0.81, temp=18.5°C, elevation=500m, ...
```

**NOT land parcels!** Each node is a fire event, not a geographic area.

### What Are Edges?

#### Spatial Edges
Connect fire events that are **nearby in space** (within ~1.1 km).

```
Node 1 (34.33, -118.52) ←→ Node 2 (34.34, -118.51)
  (Spatial edge: they're close geographically)
```

#### Temporal Edges
Connect fire events that are **close in time** (within 1 hour).

```
Node 1 (10:00) → Node 3 (11:00)
  (Temporal edge: same location, 1 hour later)
```

### How Features Come Together

```
┌─────────────────────────────────────────┐
│           NODE FEATURES                 │
├─────────────────────────────────────────┤
│ Fire:     brightness, FRP, scan, ...   │
│ Weather:  temp, humidity, wind, ...    │
│ Terrain:  elevation, slope, aspect, ... │
│ Temporal: hour, day (cyclical)         │
│ Spatial:  normalized lat/lon           │
└─────────────────────────────────────────┘
           ↓
    [GCN Layer 1]
           ↓
    Aggregate from neighbors via edges
           ↓
    [GCN Layer 2]
           ↓
    [GCN Layer 3]
           ↓
    [Output Layer]
           ↓
    Prediction: Future fire intensity
```

### Edge Aggregation Process

When GCN processes a node:
1. **Collects features** from connected neighbors
2. **Averages/aggregates** neighbor features
3. **Combines** with own features
4. **Updates** node representation

**Example:**
```
Node 1 wants to predict future fire intensity:

1. Looks at Node 2 (spatial neighbor) → gets its features
2. Looks at Node 3 (temporal neighbor) → gets its features  
3. Aggregates: (Node1_features + Node2_features + Node3_features) / 3
4. Passes through neural network
5. Outputs prediction
```

## 🏞️ Alternative: Parcel/Grid-Based Approach

### What Would Parcel-Based Look Like?

Instead of event-based nodes, you could have:

#### Option 1: Grid Cells
```
Divide study area into grid:
  Cell (0,0): 34.30-34.31°N, -118.55 to -118.54°W
  Cell (0,1): 34.30-34.31°N, -118.54 to -118.53°W
  ...

Each cell = 1 node
Features = aggregated fire events in that cell
Edges = connect adjacent cells
```

#### Option 2: Land Parcels
```
Use administrative boundaries or land parcels:
  Parcel A: National Forest Area
  Parcel B: Residential Zone
  Parcel C: Agricultural Land
  ...

Each parcel = 1 node
Features = fire risk, weather, terrain for that parcel
Edges = connect neighboring parcels
```

### Why We're NOT Using Parcels Currently

1. **Data structure**: Your data has fire events, not pre-defined parcels
2. **Flexibility**: Event-based captures actual fire dynamics
3. **Simplicity**: No need to define grid/parcel boundaries

### When to Use Parcels?

- If you have administrative boundaries
- If you want regular grid structure
- If you need to aggregate over areas
- If you have land use/ownership data

## 🔍 Why R² is Negative

**R² = -0.0166** means the model performs **worse than predicting the mean**.

### Possible Causes:

1. **Target Mismatch**
   - Current: Predicts if fire continues at exact same location
   - Problem: Fires spread, don't just continue
   - Solution: Predict spread to nearby locations

2. **Sparse Targets**
   - Many nodes have target = 0 (no fire in future)
   - Model learns to predict 0 for everything
   - Solution: Better target definition, weighted loss

3. **Feature Issues**
   - Features might not be predictive
   - Normalization might be wrong
   - Solution: Feature analysis, better scaling

4. **Graph Structure**
   - Edges might not capture fire dynamics
   - Too sparse or too dense
   - Solution: Better edge weighting, different thresholds

## ✅ What's Ready vs What Needs Work

### ✅ READY (Complete & Working):
- Data pipeline (loading, merging, preprocessing)
- Graph construction (spatial-temporal)
- Model architectures (GCN, GAT, TemporalGCN)
- Training infrastructure
- Evaluation framework
- Visualization tools
- Documentation

### ⚠️ NEEDS IMPROVEMENT:
- **Model performance** (negative R²)
- **Target creation** (predict spread, not continuation)
- **Edge features** (add weights)
- **Evaluation metrics** (spatial accuracy, etc.)
- **Graph structure** (consider grid-based alternative)

## 🚀 Immediate Next Steps

### Step 1: Understand the Problem
Run diagnostic analysis:
```bash
python improve_model.py
```

### Step 2: Fix Target Creation
- Predict fire spread to nearby locations (not just same location)
- Use spatial radius (e.g., 0.05° ≈ 5.5 km)
- Predict maximum FRP in nearby area

### Step 3: Try Different Models
- Graph Attention Network (GAT) - attention mechanism
- Temporal GCN with LSTM - better time modeling
- Add edge weights based on distance/time

### Step 4: Improve Features
- Fire spread rate (change in location)
- Weather gradients
- Terrain-based fire risk indices

### Step 5: Better Evaluation
- Spatial accuracy (how close are predictions?)
- Temporal accuracy (correct timing?)
- Fire area predictions

## 📊 Project Readiness Assessment

### For Learning/Demonstration: ✅ **READY**
- Complete end-to-end pipeline
- Working code
- Good documentation
- Demonstrates GNN concepts

### For Research Paper: ⚠️ **NEEDS WORK**
- Model performance must improve (R² > 0.3 minimum)
- Better evaluation metrics needed
- Comparison with baselines
- Ablation studies

### For Production: ❌ **NOT READY**
- Model performance too poor
- Need real-time capabilities
- Need uncertainty quantification
- Need deployment infrastructure

## 🎓 Summary

**Current Structure:**
- Nodes = Fire events (not parcels)
- Edges = Spatial + temporal connections
- Features = Multi-modal (fire + weather + terrain)
- Prediction = Future fire intensity at same/nearby location

**Key Insight:**
The graph connects **fire events** in space and time, allowing the model to learn how fires influence each other and spread over time.

**Next Priority:**
Fix target creation to predict **fire spread** rather than just **fire continuation** at the same location.

