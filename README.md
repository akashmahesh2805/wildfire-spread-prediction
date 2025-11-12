# Spatio-Temporal Wildfire Spread Prediction using Multi-Modal Graph Neural Network

## Project Overview

This project implements a **Spatio-Temporal Wildfire Spread Prediction** system using a **Multi-Modal Graph Neural Network (GNN)** for the Los Angeles region. The system leverages multiple data modalities (fire detection, weather, and topographic data) to predict wildfire spread patterns over time and space.

## Dataset Structure

The project uses three primary datasets covering the Los Angeles area:

### 1. Fire Detection Data (`fire_data.csv`)

**Size:** 29,208 fire detection records

**Columns:**

- **Spatial Features:**
  - `latitude`, `longitude`: Geographic coordinates of fire detections
- **Temporal Features:**
  - `acq_date`: Acquisition date (format: DD-MM-YYYY)
  - `acq_time`: Acquisition time (HHMM format)
- **Fire Characteristics:**
  - `brightness`: Brightness temperature of the fire pixel (Kelvin)
  - `bright_t31`: Brightness temperature at channel 31 (Kelvin)
  - `frp`: Fire Radiative Power (MW) - measures fire intensity
  - `confidence`: Confidence level of fire detection
- **Satellite Metadata:**
  - `satellite`: Satellite identifier (e.g., N20)
  - `instrument`: Sensor type (e.g., VIIRS)
  - `scan`, `track`: Pixel scan and track dimensions
  - `daynight`: Day (D) or Night (N) detection
  - `type`: Fire type classification
  - `version`: Data version

**Purpose:** Provides historical fire occurrence data with spatial and temporal information, serving as ground truth for training and validation.

---

### 2. Weather Data (`output_final_temp_celsius_fixed.csv`)

**Size:** 1,048,575 weather records

**Columns:**

- **Spatial Features:**
  - `latitude`, `longitude`: Geographic coordinates
- **Temporal Features:**
  - `date`: Date (format: DD-MM-YYYY)
  - `time`: Time (HHMM format)
- **Weather Variables:**
  - `u10`: U-component of wind at 10m height (m/s) - eastward wind
  - `v10`: V-component of wind at 10m height (m/s) - northward wind
  - `t2m`: Temperature at 2m height (°C)

**Purpose:** Provides meteorological conditions that significantly influence wildfire spread. Wind speed and direction (derived from u10, v10) are critical factors in fire propagation.

---

### 3. Topographic Data (`topo_data_cleaned.csv`)

**Size:** 11,461 topographic records

**Columns:**

- **Spatial Features:**
  - `longitude`, `latitude`: Geographic coordinates
- **Topographic Features:**
  - `elevation`: Elevation above sea level (meters)
  - `slope`: Terrain slope (degrees or percentage)
  - `aspect`: Terrain aspect/orientation (degrees, typically 0-360°)
- **Vegetation Features:**
  - `vegetation_cover`: Percentage or index of vegetation cover
  - `vegetation_type`: Classification code for vegetation type
  - `fuel_vegetation_cover`: Fuel load vegetation cover
  - `fuel_vegetation_height`: Height of fuel vegetation (meters)

**Purpose:** Provides static terrain and fuel characteristics that affect fire behavior. Elevation, slope, and aspect influence fire spread direction and speed, while vegetation data indicates fuel availability.

---

## System Architecture

### Multi-Modal Graph Neural Network Approach

The wildfire spread prediction system uses a **graph-based representation** where:

1. **Graph Construction:**

   - **Nodes:** Represent spatial locations (grid cells or points of interest) in the Los Angeles region
   - **Edges:** Connect neighboring nodes based on spatial proximity (e.g., k-nearest neighbors or distance threshold)
   - **Node Features:** Multi-modal features from all three datasets

2. **Multi-Modal Feature Integration:**

   - **Fire Modality:** Historical fire occurrences, intensity (FRP), brightness
   - **Weather Modality:** Wind vectors (u10, v10), temperature (t2m)
   - **Topographic Modality:** Elevation, slope, aspect, vegetation/fuel characteristics

3. **Temporal Modeling:**
   - Uses **Temporal Graph Neural Networks (TGNN)** or **Graph Convolutional Recurrent Networks (GCRN)**
   - Captures temporal dependencies in fire spread patterns
   - Processes sequences of graph snapshots over time

### Architecture Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Preprocessing                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │  Fire    │  │ Weather  │  │ Topo     │                   │
│  │  Data    │  │  Data    │  │  Data    │                   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                   │
│       │             │             │                         │
│       └─────────────┴─────────────┘                         │
│                    │                                        │
│              Feature Fusion                                 │
└─────────────────────┼───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              Graph Construction                             │
│  • Spatial Grid/Point Cloud                                 │
│  • Edge Creation (k-NN or distance-based)                   │
│  • Multi-modal Node Features                                │
└─────────────────────┼───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│         Multi-Modal Graph Neural Network                    │
│  ┌──────────────────────────────────────────────┐           │
│  │  Modality-Specific Encoders                  │           │
│  │  • Fire Encoder (CNN/MLP)                    │           │
│  │  • Weather Encoder (MLP)                     │           │
│  │  • Topo Encoder (MLP)                        │           │
│  └──────────────┬───────────────────────────────┘           │
│                 │                                           │
│  ┌──────────────▼───────────────────────────────┐           │
│  │  Feature Fusion Layer                        │           │
│  │  (Concatenation/Attention/Weighted Sum)      │           │
│  └──────────────┬───────────────────────────────┘           │
│                 │                                           │
│  ┌──────────────▼───────────────────────────────┐           │
│  │  Graph Convolution Layers                    │           │
│  │  (GCN/GAT/GraphSAGE)                         │           │
│  └──────────────┬───────────────────────────────┘           │
│                 │                                           │
│  ┌──────────────▼───────────────────────────────┐           │
│  │  Temporal Modeling                           │           │
│  │  (LSTM/GRU/Transformer on Graph Sequences)   │           │
│  └──────────────┬───────────────────────────────┘           │
└─────────────────┼───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│              Prediction Head                                │
│  • Fire Spread Probability                                  │
│  • Fire Intensity (FRP)                                     │
│  • Spatial Spread Direction                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## How It Works

### 1. Data Preprocessing

**Spatial Alignment:**

- All datasets are spatially aligned to a common grid or point cloud
- Interpolation may be needed to match spatial resolutions
- Temporal alignment ensures features correspond to the same time steps

**Feature Engineering:**

- **Wind Speed & Direction:** Calculate from u10 and v10:
  - Wind speed = √(u10² + v10²)
  - Wind direction = arctan2(v10, u10)
- **Fire History:** Aggregate historical fire occurrences per location
- **Temporal Features:** Extract day-of-year, hour-of-day, time since last fire

**Graph Construction:**

- Create nodes for each spatial location (grid cell or point)
- Connect nodes based on:
  - **Spatial proximity:** k-nearest neighbors or distance threshold
  - **Topographic connectivity:** Consider elevation changes
  - **Wind direction:** Weight edges based on prevailing wind patterns

### 2. Multi-Modal Feature Encoding

Each modality is processed through specialized encoders:

- **Fire Modality Encoder:**

  - Input: brightness, FRP, confidence, historical fire counts
  - Output: Fire feature embeddings

- **Weather Modality Encoder:**

  - Input: u10, v10, t2m, derived wind speed/direction
  - Output: Weather feature embeddings

- **Topographic Modality Encoder:**
  - Input: elevation, slope, aspect, vegetation features
  - Output: Topographic feature embeddings

### 3. Graph Neural Network Processing

**Graph Convolution:**

- Nodes aggregate information from neighboring nodes
- Uses message passing to propagate fire spread information
- Multiple layers capture multi-hop dependencies

**Key Operations:**

- **Message Passing:** Each node receives messages from neighbors
- **Aggregation:** Combine neighbor messages (mean, max, attention)
- **Update:** Update node representations based on aggregated messages

### 4. Temporal Modeling

**Sequence Processing:**

- Process graph snapshots at each time step
- Use recurrent or transformer architectures to model temporal dynamics
- Capture how fire spreads over time

**Temporal Features:**

- Historical fire states
- Weather evolution
- Time-dependent patterns (seasonality, diurnal cycles)

### 5. Prediction

**Outputs:**

- **Fire Spread Probability:** Likelihood of fire at each location at future time steps
- **Fire Intensity:** Predicted FRP values
- **Spread Direction:** Predicted direction of fire propagation

**Loss Functions:**

- Binary cross-entropy for fire occurrence
- Mean squared error for fire intensity
- Custom loss combining spatial and temporal accuracy

---

## Implementation Workflow

### Step 1: Data Loading and Exploration

```python
# Load datasets
fire_df = pd.read_csv('fire_data.csv')
weather_df = pd.read_csv('output_final_temp_celsius_fixed.csv')
topo_df = pd.read_csv('topo_data_cleaned.csv')

# Explore data distributions, missing values, temporal ranges
```

### Step 2: Data Preprocessing

```python
# Spatial alignment to common grid
# Temporal alignment and feature engineering
# Create unified feature matrix per time step
```

### Step 3: Graph Construction

```python
# Create spatial graph
# Define node features from multi-modal data
# Create edges based on spatial relationships
```

### Step 4: Model Architecture

```python
# Define modality encoders
# Graph convolution layers
# Temporal modeling layers
# Prediction head
```

### Step 5: Training

```python
# Split data into train/validation/test sets
# Define loss function
# Train model with temporal sequences
# Validate on held-out data
```

### Step 6: Evaluation

```python
# Evaluate on test set
# Metrics: Accuracy, Precision, Recall, F1-score
# Spatial accuracy metrics
# Temporal prediction accuracy
```

---

## Key Challenges and Solutions

### Challenge 1: Multi-Modal Data Integration

**Solution:** Use modality-specific encoders followed by fusion layers (concatenation, attention mechanisms, or learned weighted combinations)

### Challenge 2: Spatial-Temporal Alignment

**Solution:**

- Create spatial grid with appropriate resolution
- Temporal interpolation for missing time steps
- Handle different temporal resolutions across modalities

### Challenge 3: Imbalanced Data

**Solution:**

- Fire occurrences are rare events
- Use weighted loss functions, focal loss, or oversampling techniques

### Challenge 4: Graph Construction

**Solution:**

- Use adaptive graph construction (learn edge weights)
- Consider multiple edge types (spatial, topographic, wind-based)

### Challenge 5: Temporal Dependencies

**Solution:**

- Use sequence models (LSTM, GRU, Transformers) on graph sequences
- Capture long-term dependencies with attention mechanisms

---

## Expected Outcomes

The model should be able to:

1. **Predict Fire Occurrence:** Identify locations likely to experience fires
2. **Predict Fire Spread:** Forecast how fires will spread spatially over time
3. **Predict Fire Intensity:** Estimate fire radiative power (FRP) at different locations
4. **Understand Influencing Factors:** Identify which factors (weather, topography, vegetation) most influence fire spread

---

## Evaluation Metrics

- **Spatial Accuracy:**
  - Intersection over Union (IoU) for fire areas
  - Distance-based metrics (Hausdorff distance)
- **Temporal Accuracy:**
  - Time-to-event prediction accuracy
  - Sequence prediction metrics
- **Classification Metrics:**
  - Precision, Recall, F1-score
  - Area Under ROC Curve (AUC-ROC)
- **Regression Metrics:**
  - Mean Absolute Error (MAE) for FRP prediction
  - Root Mean Squared Error (RMSE)

---

## Dependencies

Typical libraries required:

- `torch` / `torch-geometric` - Graph neural networks
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - Preprocessing and evaluation
- `matplotlib`, `seaborn` - Visualization
- `geopandas` - Spatial data handling (optional)

---

## Future Enhancements

1. **Real-time Prediction:** Deploy model for real-time wildfire monitoring
2. **Uncertainty Quantification:** Provide confidence intervals for predictions
3. **Explainability:** Visualize which features contribute most to predictions
4. **Multi-scale Modeling:** Combine fine-grained and coarse-grained predictions
5. **Integration with Remote Sensing:** Incorporate real-time satellite imagery

---

## References

- Graph Neural Networks for Spatio-Temporal Forecasting
- Multi-Modal Learning in Remote Sensing
- Wildfire Spread Modeling
- Temporal Graph Neural Networks

---

## Contact

For questions or contributions, please refer to the project repository.
