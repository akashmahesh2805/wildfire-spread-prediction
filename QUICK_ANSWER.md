# Quick Answers

## How to Run

**Easiest way:**
```powershell
.\run_workflow.bat
```

**Or directly:**
```powershell
.\venv\Scripts\python.exe notebooks\complete_workflow_example.py
```

## Node Features: ✅ ALL COMBINED

**Yes!** All node features from all modalities are combined into one feature vector:

```
Per Grid Cell:
├── Fire features (5): brightness, FRP, scan, track, bright_t31
├── Weather features (5): temp, humidity, wind_speed, wind_dir, precip  
├── Terrain features (7): elevation, slope, aspect, veg_cover, ...
├── Temporal features (4): hour_sin, hour_cos, day_sin, day_cos
└── Spatial features (2): lat_norm, lon_norm
                        ↓
        Combined: [23 features] → Single vector per cell
```

**How it works:**
1. `aggregate_features()` collects ALL features
2. Aggregates per grid cell (mean for numerical)
3. Normalizes all together
4. Returns: `(num_cells, 23)` feature matrix
5. Model receives ALL features as one input vector

## Edge Features: ❌ NOT YET

**No edge features currently.** Edges are binary (connected/not connected).

**What edges do:**
- **Spatial edges**: Connect adjacent grid cells
- **Temporal edges**: Connect cells with fires at consecutive times
- **No weights**: All edges treated equally

**Future enhancement:** Could add:
- Distance between cells
- Time difference
- Fire spread direction

## Summary

- **Node features**: ✅ All modalities combined (23 features per cell)
- **Edge features**: ❌ Not implemented (binary connections only)
- **How they work**: GCN aggregates neighbor node features through edges

See `FEATURE_COMBINATION_EXPLAINED.md` for detailed explanation!

