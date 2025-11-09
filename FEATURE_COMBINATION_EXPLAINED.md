# How Node and Edge Features Work Together

## 🔍 Current Implementation

### Node Features: ✅ **ALL COMBINED**

**What happens:**
1. All features from different modalities are **concatenated** into one feature vector per node
2. This combined vector is fed into the GCN layers
3. The model learns to use all features together

**Example:**
```
Node (Grid Cell) Features:
├── Fire features:      [brightness, FRP, scan, track, bright_t31]
├── Weather features:   [temp, humidity, wind_speed, wind_dir, precip]
├── Terrain features:   [elevation, slope, aspect, veg_cover, ...]
├── Temporal features:  [hour_sin, hour_cos, day_sin, day_cos]
└── Spatial features:   [lat_norm, lon_norm]
                        ↓
        Combined: [23 features total] → GCN Input
```

**Code location:** `src/grid_graph_builder.py` → `aggregate_features()`
- Aggregates features per grid cell
- Normalizes all features together
- Returns single feature matrix: `(num_cells, num_features)`

### Edge Features: ❌ **NOT YET IMPLEMENTED**

**Current state:**
- Edges are **binary** (connected or not)
- No edge features (distance, time difference, etc.)
- Edges only define **connectivity**, not **strength**

**What we could add:**
```python
Edge Features (Future):
├── Spatial edges:
│   ├── Distance between cells
│   ├── Direction (N, S, E, W)
│   └── Terrain similarity
└── Temporal edges:
    ├── Time difference
    └── Fire spread direction
```

## 🧠 How GCN Uses Features

### Step-by-Step Process:

```
1. Node Features (Input)
   Each grid cell has: [fire + weather + terrain + temporal + spatial]
   Shape: (num_cells, 23)

2. GCN Layer 1
   ├── For each node:
   │   ├── Collect neighbor features (via edges)
   │   ├── Aggregate neighbors (mean/sum)
   │   └── Combine with own features
   └── Output: (num_cells, hidden_dim)

3. GCN Layer 2
   └── Same process, but on transformed features

4. GCN Layer 3
   └── Final transformation

5. Output Layer
   └── Predicts: Future fire intensity per cell
```

### Feature Aggregation Example:

```
Grid Cell A wants to predict fire intensity:

1. Looks at neighbors (via spatial edges):
   - Cell B (north): [features_B]
   - Cell C (south): [features_C]
   - Cell D (east):  [features_D]
   - Cell E (west):  [features_E]

2. Aggregates neighbor features:
   aggregated = (features_B + features_C + features_D + features_E) / 4

3. Combines with own features:
   combined = [features_A, aggregated]

4. Passes through neural network:
   prediction = model(combined)
```

## 📊 Current Feature Combination

### In `grid_graph_builder.py`:

```python
def aggregate_features(self, df, feature_groups):
    # Collects ALL features from ALL modalities
    feature_list = []
    for group, cols in feature_groups.items():
        for col in cols:
            feature_list.append(col)  # All features collected
    
    # Aggregates per grid cell (mean for numerical)
    # Returns: (num_cells, total_features)
    # ALL features combined into one vector per cell
```

### In `models.py` (MultiModalGCN):

```python
def forward(self, x, edge_index):
    # x shape: (num_nodes, input_dim)
    # input_dim = ALL features combined (23)
    
    # Projects all features together
    x = self.input_proj(x)  # (num_nodes, hidden_dim)
    
    # GCN layers use ALL features together
    for gcn in self.gcn_layers:
        x = gcn(x, edge_index)  # Aggregates via edges
```

## ✅ Summary

### Node Features:
- ✅ **ALL modalities combined** into one feature vector
- ✅ Fire + Weather + Terrain + Temporal + Spatial
- ✅ 23 features total per grid cell
- ✅ Normalized together

### Edge Features:
- ❌ **Not implemented yet**
- ❌ Edges are binary (connected/not connected)
- ✅ Edges define which cells can influence each other
- 🔮 Future: Could add distance, time, direction as edge features

### How They Work Together:
1. **Node features** = What's in each grid cell (all modalities combined)
2. **Edges** = Which cells are connected (spatial + temporal)
3. **GCN** = Aggregates neighbor features through edges
4. **Model** = Learns to predict using all features together

## 🚀 To Add Edge Features (Future Enhancement)

Would need to modify:
1. `grid_graph_builder.py` → `compute_spatial_edges()` to return edge attributes
2. `models.py` → GCN layers to use edge features
3. Use `edge_attr` parameter in PyTorch Geometric

Example:
```python
# In graph builder
edge_attr = compute_edge_weights(edge_index, distances, time_diffs)

# In model
x = gcn(x, edge_index, edge_attr=edge_attr)
```

