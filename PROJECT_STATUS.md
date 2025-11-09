# Project Status & Next Steps

## ✅ What's Complete

### 1. **Data Pipeline** ✅

- Data loading and merging (fire, weather, terrain)
- Feature engineering (temporal, spatial, multi-modal)
- Missing value handling
- Data preprocessing

### 2. **Graph Construction** ✅

- Spatial-temporal graph building
- Node and edge creation
- Memory optimization (sampling)

### 3. **Model Architecture** ✅

- Multi-Modal GCN implementation
- Training pipeline
- Evaluation metrics

### 4. **Infrastructure** ✅

- Complete codebase
- Documentation
- Visualization tools

## ⚠️ Current Issues

### 1. **Model Performance** (R² = -0.0166)

**Problem**: Negative R² means model performs worse than predicting the mean.

**Possible Causes**:

- Target creation might not align with actual fire spread
- Graph structure may not capture fire dynamics well
- Model capacity or architecture limitations
- Feature scaling/normalization issues

### 2. **Graph Structure Understanding**

**Current Approach**:

- **Nodes** = Individual fire events (detections) at specific (lat, lon, time)
- **Spatial Edges** = Connect fire events within ~1.1 km
- **Temporal Edges** = Connect fire events within 1 hour
- **Node Features** = Combined fire + weather + terrain features

**This is NOT land parcel-based!** Each node is a fire detection event, not a land parcel.

## 🔍 How Node & Edge Features Work

### Node Features (Current)

Each node represents a **fire event** with:

- **Fire features**: brightness, FRP, scan, track, bright_t31
- **Weather features**: temperature, humidity, wind speed/direction, precipitation
- **Terrain features**: elevation, slope, aspect, vegetation cover
- **Temporal features**: hour, day (cyclical encodings)
- **Spatial features**: normalized latitude/longitude

### Edge Features (Current)

- **Spatial edges**: Connect nearby fire events (within 0.01° ≈ 1.1 km)
- **Temporal edges**: Connect fire events within 1 hour
- **No edge features**: Currently just binary connections

### How They Come Together

1. **Node features** are fed into GCN layers
2. **GCN aggregates** neighbor features through edges
3. **Spatial edges** allow nearby fires to influence each other
4. **Temporal edges** allow past fires to influence future predictions

## 🎯 Next Steps to Improve

### Priority 1: Fix Model Performance

#### A. Improve Target Creation

Current target creation looks for fires at same location 1 hour later. This might be too restrictive.

**Better approach**:

```python
# Instead of exact location match, predict:
# 1. Will fire spread to nearby locations?
# 2. Will fire intensity increase/decrease?
# 3. Will new fires ignite nearby?
```

#### B. Alternative Graph Structure

Consider **grid-based** or **parcel-based** approach:

**Option 1: Grid-Based Nodes**

- Divide study area into grid cells (e.g., 0.01° × 0.01°)
- Each cell is a node
- Edges connect adjacent cells
- Features: aggregated fire events in cell + weather + terrain

**Option 2: Parcel-Based Nodes**

- Use administrative boundaries or land parcels
- Each parcel is a node
- Features: fire risk, weather, terrain for that parcel

**Option 3: Hybrid (Recommended)**

- Keep event-based nodes
- Add spatial aggregation features
- Better edge weighting (distance-based, not binary)

### Priority 2: Model Improvements

#### A. Try Different Architectures

- Graph Attention Network (GAT) - already implemented
- Temporal GCN with LSTM - already implemented
- Multi-Modal Fusion GNN - already implemented

#### B. Add Edge Features

```python
# Distance-based edge weights
edge_weight = 1 / (distance + epsilon)

# Temporal edge weights
edge_weight = exp(-time_diff / tau)
```

#### C. Feature Engineering

- Fire spread rate (change in location over time)
- Fire intensity trends
- Weather gradients
- Terrain-based fire risk indices

### Priority 3: Evaluation Improvements

#### A. Better Metrics

- Spatial accuracy (how close are predictions to actual locations?)
- Temporal accuracy (correct prediction timing?)
- Fire size/area predictions

#### B. Visualization

- Animated fire spread over time
- Heat maps of predicted vs actual
- Error analysis by region/time

## 📋 Recommended Action Plan

### Phase 1: Quick Fixes (1-2 days)

1. **Improve target creation** - predict spread to nearby locations
2. **Add edge weights** - distance and time-based
3. **Try GAT model** - attention might help
4. **Feature scaling** - ensure proper normalization

### Phase 2: Structural Changes (3-5 days)

1. **Grid-based graph** - more structured approach
2. **Better aggregation** - spatial-temporal pooling
3. **Multi-scale features** - local + regional context

### Phase 3: Advanced (1 week+)

1. **Ensemble methods** - combine multiple models
2. **Uncertainty quantification** - confidence intervals
3. **Real-time prediction** - deployment pipeline

## 🎓 Understanding the Current Graph

### Current Structure:

```
Nodes: Fire Events
├── Node 1: Fire at (34.33, -118.52) at 2020-01-01 10:00
├── Node 2: Fire at (34.34, -118.51) at 2020-01-01 10:00
└── Node 3: Fire at (34.33, -118.52) at 2020-01-01 11:00

Edges:
├── Spatial: Node 1 ↔ Node 2 (nearby locations)
└── Temporal: Node 1 → Node 3 (same location, next hour)

Features per Node:
├── Fire: brightness=298, FRP=0.57, ...
├── Weather: temp=17.2°C, wind=3.2 m/s, ...
├── Terrain: elevation=500m, slope=5°, ...
└── Temporal: hour=10, day=1, ...
```

### This is Event-Based, NOT Parcel-Based!

If you want **parcel-based**:

1. Define grid cells or parcels
2. Aggregate fire events within each cell
3. Create nodes for each cell
4. Connect adjacent cells

## ✅ Project Readiness

### For Learning/Demonstration: **READY** ✅

- Complete pipeline
- Working code
- Good structure
- Documentation

### For Production/Research: **NEEDS IMPROVEMENT** ⚠️

- Model performance (negative R²)
- Graph structure could be better
- Target definition needs refinement
- Evaluation metrics need expansion

## 🚀 Immediate Next Steps

1. **Run analysis script** to understand why R² is negative
2. **Improve target creation** - predict spread, not just continuation
3. **Try GAT model** - might perform better
4. **Add edge weights** - distance/time-based
5. **Visualize predictions** - see where model fails

Would you like me to create:

- A grid-based graph builder?
- An improved target creation function?
- A diagnostic script to analyze model failures?
- A comparison of different model architectures?
