"""
Step 2: Data Preprocessing
===========================
This module processes raw fire, weather, and topographic datasets to generate
node-level features over time, ready for graph construction.
"""

import pandas as pd
import numpy as np
import os
import pickle
from datetime import timedelta
from data_loading_1 import DataLoader

class DataPreprocessor:
    """Class to preprocess wildfire datasets for graph-based modeling."""

    def __init__(self, data_dir='.', grid_resolution=0.01, time_window_hours=24):
        """
        Initialize the DataPreprocessor.

        Args:
            data_dir: Directory containing datasets
            grid_resolution: Spatial grid resolution (degrees)
            time_window_hours: Temporal aggregation window
        """
        self.data_dir = data_dir
        self.grid_resolution = grid_resolution
        self.time_window_hours = time_window_hours
        self.loader = DataLoader(data_dir)
        self.fire_df = None
        self.weather_df = None
        self.topo_df = None
        self.grid_points = None
        self.processed_df = None

    def load_datasets(self):
        """Load datasets using DataLoader."""
        print("="*60)
        print("LOADING DATASETS")
        print("="*60)
        self.fire_df, self.weather_df, self.topo_df = self.loader.load_all_datasets()

        # Parse datetime columns
        self.fire_df['acq_datetime'] = pd.to_datetime(
            self.fire_df['acq_date'] + ' ' + self.fire_df['acq_time'].astype(str).str.zfill(4),
            format='%d-%m-%Y %H%M'
        )
        self.weather_df['datetime'] = pd.to_datetime(
            self.weather_df['date'] + ' ' + self.weather_df['time'].astype(str).str.zfill(4),
            format='%d-%m-%Y %H%M'
        )
        return self.fire_df, self.weather_df, self.topo_df

    def create_spatial_grid(self):
        """Create a uniform spatial grid covering all fire locations."""
        print("\nCREATING SPATIAL GRID")
        lat_min, lat_max = self.fire_df['latitude'].min(), self.fire_df['latitude'].max()
        lon_min, lon_max = self.fire_df['longitude'].min(), self.fire_df['longitude'].max()

        lat_grid = np.arange(lat_min, lat_max + self.grid_resolution, self.grid_resolution)
        lon_grid = np.arange(lon_min, lon_max + self.grid_resolution, self.grid_resolution)
        grid_lat, grid_lon = np.meshgrid(lat_grid, lon_grid)
        self.grid_points = pd.DataFrame({
            'latitude': grid_lat.ravel(),
            'longitude': grid_lon.ravel()
        })
        print(f"   Grid points created: {len(self.grid_points)}")
        return self.grid_points

    def aggregate_fire_features(self):
        """Aggregate fire features to grid nodes per time window."""
        print("\nAGGREGATING FIRE FEATURES")
        df = self.fire_df.copy()
        df['time_window'] = df['acq_datetime'].dt.floor(f'{self.time_window_hours}h')

        # Assign nearest grid point
        df['grid_lat'] = df['latitude'].round(2)
        df['grid_lon'] = df['longitude'].round(2)
        self.fire_df = df.groupby(['grid_lat', 'grid_lon', 'time_window']).agg({
            'brightness': 'mean',
            'frp': 'mean',
            'type': lambda x: x.mode()[0] if len(x) > 0 else np.nan
        }).reset_index().rename(columns={'grid_lat': 'latitude', 'grid_lon': 'longitude'})
        print(f"   Aggregated {len(self.fire_df)} fire records")
        return self.fire_df

    def aggregate_weather_features(self):
        """Aggregate weather features to grid nodes per time window."""
        print("\nAGGREGATING WEATHER FEATURES")
        df = self.weather_df.copy()
        df['time_window'] = df['datetime'].dt.floor(f'{self.time_window_hours}h')
        df['grid_lat'] = df['latitude'].round(2)
        df['grid_lon'] = df['longitude'].round(2)
        self.weather_df = df.groupby(['grid_lat', 'grid_lon', 'time_window']).agg({
            't2m': 'mean',
            'u10': 'mean',
            'v10': 'mean'
        }).reset_index().rename(columns={'grid_lat': 'latitude', 'grid_lon': 'longitude'})
        print(f"   Aggregated {len(self.weather_df)} weather records")
        return self.weather_df

    def interpolate_topo_features(self):
        """Interpolate topo features onto grid nodes."""
        print("\nINTERPOLATING TOPOGRAPHIC FEATURES")
        df = self.topo_df.copy()
        grid = self.grid_points.copy()

        topo_cols = ['elevation', 'slope', 'aspect', 'vegetation_cover', 
                     'vegetation_type', 'fuel_vegetation_cover', 'fuel_vegetation_height']

        # Nearest neighbor interpolation
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=1).fit(df[['latitude', 'longitude']])
        distances, indices = nbrs.kneighbors(grid[['latitude', 'longitude']])
        interpolated = df.iloc[indices.flatten()][topo_cols].reset_index(drop=True)
        self.topo_interp = pd.concat([grid.reset_index(drop=True), interpolated], axis=1)
        print(f"   Interpolated topo features for {len(self.topo_interp)} grid nodes")
        return self.topo_interp

    def merge_all_features(self):
        """Merge fire, weather, and topo features into a single dataframe."""
        print("\nMERGING ALL FEATURES")
        # Start with cartesian product of grid nodes and unique time windows
        time_windows = pd.DataFrame({'time_window': sorted(self.fire_df['time_window'].unique())})
        grid = self.grid_points
        full_df = pd.merge(
            grid.assign(key=0), time_windows.assign(key=0), on='key'
        ).drop('key', axis=1)

        # Merge fire features
        full_df = pd.merge(
            full_df, self.fire_df, on=['latitude', 'longitude', 'time_window'], how='left'
        )

        # Merge weather features
        full_df = pd.merge(
            full_df, self.weather_df, on=['latitude', 'longitude', 'time_window'], how='left'
        )

        # Merge topo features
        full_df = pd.merge(
            full_df, self.topo_interp, on=['latitude', 'longitude'], how='left'
        )

        # Fill missing numeric features with 0
        numeric_cols = full_df.select_dtypes(include=np.number).columns
        full_df[numeric_cols] = full_df[numeric_cols].fillna(0)

        self.processed_df = full_df
        print(f"   Merged dataframe shape: {self.processed_df.shape}")
        return self.processed_df

    def save_processed_data(self, output_file='processed_data.pkl'):
        """Save processed dataframe to file."""
        with open(output_file, 'wb') as f:
            pickle.dump(self.processed_df, f)
        print(f"\nProcessed data saved to {output_file}")

def main():
    print("="*60)
    print("STEP 2: DATA PREPROCESSING")
    print("="*60)
    preprocessor = DataPreprocessor()
    preprocessor.load_datasets()
    preprocessor.create_spatial_grid()
    preprocessor.aggregate_fire_features()
    preprocessor.aggregate_weather_features()
    preprocessor.interpolate_topo_features()
    preprocessor.merge_all_features()
    preprocessor.save_processed_data()
    print("\nDATA PREPROCESSING COMPLETE!")
    return preprocessor, preprocessor.processed_df

if __name__ == "__main__":
    preprocessor, processed_df = main()