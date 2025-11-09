# Spatio-Temporal Wildfire Spread Prediction using Multi-Modal Graph Neural Network

## Project Overview

This project aims to predict wildfire spread using a Multi-Modal Graph Neural Network (GNN) that combines:

- **Spatial data**: Fire locations (latitude, longitude), topographic features
- **Temporal data**: Time-series weather conditions and fire progression
- **Multi-modal features**: Fire characteristics, weather, and terrain data

## Project Structure

```
wildfire-spread-prediction/
├── data/
│   ├── fire_data.csv              # Fire detection data (29,208 records)
│   ├── weather_data.csv            # Hourly weather data (48,192 records)
│   ├── topo_data_cleaned.csv    # Topographic features (11,461 records)
│   └── windspeed_cleaned.csv      # Wind speed data
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Data exploration and analysis
│   ├── 02_data_preprocessing.ipynb # Data cleaning and feature engineering
│   ├── 03_graph_construction.ipynb # Building spatial-temporal graphs
│   ├── 04_model_training.ipynb    # GNN model training
│   └── 05_evaluation.ipynb        # Model evaluation and visualization
├── src/
│   ├── data_loader.py             # Data loading utilities
│   ├── graph_builder.py           # Graph construction from spatial data
│   ├── models.py                  # GNN model architectures
│   ├── trainer.py                 # Training loop and utilities
│   └── utils.py                   # Helper functions
├── models/                         # Saved model checkpoints
├── results/                        # Evaluation results and plots
└── requirements.txt                # Python dependencies

```

## Data Description

### 1. Fire Data (`fire_data.csv`)

- **29,208 fire detection records**
- Features: latitude, longitude, brightness, FRP (Fire Radiative Power), scan, track, satellite info, confidence, timestamps
- Key columns: `latitude`, `longitude`, `brightness`, `frp`, `acq_date`, `acq_time`

### 2. Weather Data (`weather_data.csv`)

- **48,192 hourly weather records**
- Features: temperature, relative humidity, wind speed, wind direction, precipitation
- Key columns: `time`, `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, `wind_direction_10m`, `precipitation`

### 3. Topographic Data (`topo_data_cleaned.csv`)

- **11,461 location records**
- Features: elevation, slope, aspect, vegetation cover, vegetation type, fuel characteristics
- Key columns: `latitude`, `longitude`, `elevation`, `slope`, `aspect`, `vegetation_cover`, `fuel_vegetation_cover`

## Workflow Overview

### Phase 1: Data Understanding & Preprocessing

1. **Data Exploration**: Understand distributions, missing values, temporal patterns
2. **Data Cleaning**: Handle missing values, outliers, coordinate alignment
3. **Feature Engineering**: Create temporal features, spatial features, derived metrics

### Phase 2: Graph Construction

1. **Spatial Graph**: Connect nearby fire locations based on distance
2. **Temporal Graph**: Connect fire events across time steps
3. **Multi-Modal Features**: Combine fire, weather, and topographic features

### Phase 3: Model Development

1. **Graph Neural Network Architecture**:
   - Spatial GNN layers (GCN, GAT, or GraphSAGE)
   - Temporal modeling (LSTM/GRU or Temporal GNN)
   - Multi-modal feature fusion
2. **Training**: Train on historical fire spread patterns
3. **Validation**: Evaluate on held-out temporal periods

### Phase 4: Prediction & Evaluation

1. **Spread Prediction**: Predict future fire locations and intensity
2. **Evaluation Metrics**: Accuracy, precision, recall, spatial error
3. **Visualization**: Map-based visualizations of predictions

## Installation

```bash
# Create virtual environment (if not already created)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

1. **Explore the data**:

   ```python
   jupyter notebook notebooks/01_data_exploration.ipynb
   ```

2. **Preprocess data**:

   ```python
   jupyter notebook notebooks/02_data_preprocessing.ipynb
   ```

3. **Build graphs**:

   ```python
   jupyter notebook notebooks/03_graph_construction.ipynb
   ```

4. **Train model**:

   ```python
   jupyter notebook notebooks/04_model_training.ipynb
   ```

5. **Evaluate**:
   ```python
   jupyter notebook notebooks/05_evaluation.ipynb
   ```

## Key Concepts

### Spatial-Temporal Graph

- **Nodes**: Fire events at specific locations and times
- **Spatial Edges**: Connect nearby fire locations (within distance threshold)
- **Temporal Edges**: Connect fire events across consecutive time steps
- **Node Features**: Multi-modal features (fire intensity, weather, terrain)

### Multi-Modal Features

- **Fire Modality**: brightness, FRP, confidence
- **Weather Modality**: temperature, humidity, wind, precipitation
- **Terrain Modality**: elevation, slope, aspect, vegetation

### Prediction Task

- **Input**: Current fire state + weather + terrain
- **Output**: Future fire spread (next time step locations and intensities)

## Dependencies

- PyTorch / PyTorch Geometric (for GNN)
- NumPy, Pandas (data processing)
- Scikit-learn (evaluation)
- Matplotlib, Seaborn (visualization)
- Jupyter (notebooks)

## References

- Graph Neural Networks for Spatial-Temporal Forecasting
- Multi-Modal Learning in GNNs
- Wildfire Spread Prediction Literature
