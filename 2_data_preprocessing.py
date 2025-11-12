"""
Step 2: Data Preprocessing
==========================
This module handles spatial and temporal alignment of multi-modal data,
feature engineering, and creation of unified feature matrices.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import os

class DataPreprocessor:
    """Class to preprocess and align multi-modal wildfire data."""
    
    def __init__(self, fire_df, weather_df, topo_df):
        """
        Initialize the preprocessor.
        
        Args:
            fire_df: Fire detection dataframe
            weather_df: Weather dataframe
            topo_df: Topographic dataframe
        """
        self.fire_df = fire_df.copy()
        self.weather_df = weather_df.copy()
        self.topo_df = topo_df.copy()
        
        # Preprocessed data
        self.processed_data = None
        self.scalers = {}
        
    def parse_dates(self):
        """Parse date columns in all dataframes."""
        print("Parsing dates...")
        
        # Parse fire data dates
        self.fire_df['acq_date_parsed'] = pd.to_datetime(
            self.fire_df['acq_date'], format='%d-%m-%Y', errors='coerce'
        )
        # Convert time to datetime component
        self.fire_df['acq_time_str'] = self.fire_df['acq_time'].astype(str).str.zfill(4)
        self.fire_df['acq_datetime'] = pd.to_datetime(
            self.fire_df['acq_date_parsed'].astype(str) + ' ' + 
            self.fire_df['acq_time_str'].str[:2] + ':' + 
            self.fire_df['acq_time_str'].str[2:], 
            errors='coerce'
        )
        
        # Parse weather data dates
        self.weather_df['date_parsed'] = pd.to_datetime(
            self.weather_df['date'], format='%d-%m-%Y', errors='coerce'
        )
        # Convert time to datetime component
        self.weather_df['time_str'] = self.weather_df['time'].astype(str).str.zfill(4)
        self.weather_df['datetime'] = pd.to_datetime(
            self.weather_df['date_parsed'].astype(str) + ' ' + 
            self.weather_df['time_str'].str[:2] + ':' + 
            self.weather_df['time_str'].str[2:], 
            errors='coerce'
        )
        
        print(f"   Fire data date range: {self.fire_df['acq_date_parsed'].min()} to {self.fire_df['acq_date_parsed'].max()}")
        print(f"   Weather data date range: {self.weather_df['date_parsed'].min()} to {self.weather_df['date_parsed'].max()}")
    
    def create_spatial_grid(self, lat_min=None, lat_max=None, lon_min=None, lon_max=None, 
                           grid_resolution=0.01):
        """
        Create a spatial grid for aligning all data.
        
        Args:
            lat_min, lat_max, lon_min, lon_max: Bounds for the grid
            grid_resolution: Resolution in degrees (approximately 1km per 0.01 degrees)
        
        Returns:
            Grid coordinates
        """
        print(f"\nCreating spatial grid with resolution {grid_resolution} degrees...")
        
        # Determine bounds from all datasets if not provided
        if lat_min is None:
            lat_min = min(self.fire_df['latitude'].min(), 
                         self.weather_df['latitude'].min(),
                         self.topo_df['latitude'].min())
        if lat_max is None:
            lat_max = max(self.fire_df['latitude'].max(),
                         self.weather_df['latitude'].max(),
                         self.topo_df['latitude'].max())
        if lon_min is None:
            lon_min = min(self.fire_df['longitude'].min(),
                         self.weather_df['longitude'].min(),
                         self.topo_df['longitude'].min())
        if lon_max is None:
            lon_max = max(self.fire_df['longitude'].max(),
                         self.weather_df['longitude'].max(),
                         self.topo_df['longitude'].max())
        
        # Create grid
        lat_grid = np.arange(lat_min, lat_max + grid_resolution, grid_resolution)
        lon_grid = np.arange(lon_min, lon_max + grid_resolution, grid_resolution)
        
        # Create meshgrid
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        # Flatten to get grid points
        grid_points = pd.DataFrame({
            'latitude': lat_mesh.flatten(),
            'longitude': lon_mesh.flatten()
        })
        
        print(f"   Grid bounds: Lat [{lat_min:.4f}, {lat_max:.4f}], Lon [{lon_min:.4f}, {lon_max:.4f}]")
        print(f"   Grid size: {len(grid_points)} points")
        
        return grid_points, lat_grid, lon_grid
    
    def aggregate_fire_features(self, grid_points, time_window_hours=24):
        """
        Aggregate fire features to grid points and time windows.
        
        Args:
            grid_points: DataFrame with latitude and longitude
            time_window_hours: Time window for aggregation
        """
        print("\nAggregating fire features...")
        
        # Round coordinates to grid
        self.fire_df['lat_rounded'] = round(self.fire_df['latitude'] / 0.01) * 0.01
        self.fire_df['lon_rounded'] = round(self.fire_df['longitude'] / 0.01) * 0.01
        
        # Create time windows
        self.fire_df['time_window'] = self.fire_df['acq_datetime'].dt.floor(f'{time_window_hours}H')
        
        # Aggregate fire features
        fire_agg = self.fire_df.groupby(['lat_rounded', 'lon_rounded', 'time_window']).agg({
            'frp': ['sum', 'mean', 'max', 'count'],
            'brightness': 'mean',
            'bright_t31': 'mean',
            'confidence': 'mean'
        }).reset_index()
        
        # Flatten column names
        fire_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                           for col in fire_agg.columns.values]
        fire_agg.rename(columns={'lat_rounded': 'latitude', 'lon_rounded': 'longitude'}, 
                       inplace=True)
        
        print(f"   Aggregated {len(fire_agg)} fire records")
        return fire_agg
    
    def aggregate_weather_features(self, grid_points, time_window_hours=24):
        """
        Aggregate weather features to grid points and time windows.
        
        Args:
            grid_points: DataFrame with latitude and longitude
            time_window_hours: Time window for aggregation
        """
        print("\nAggregating weather features...")
        
        # Round coordinates to grid
        self.weather_df['lat_rounded'] = round(self.weather_df['latitude'] / 0.01) * 0.01
        self.weather_df['lon_rounded'] = round(self.weather_df['longitude'] / 0.01) * 0.01
        
        # Create time windows
        self.weather_df['time_window'] = self.weather_df['datetime'].dt.floor(f'{time_window_hours}H')
        
        # Calculate wind speed and direction
        self.weather_df['wind_speed'] = np.sqrt(self.weather_df['u10']**2 + self.weather_df['v10']**2)
        self.weather_df['wind_direction'] = np.arctan2(self.weather_df['v10'], self.weather_df['u10'])
        
        # Aggregate weather features
        weather_agg = self.weather_df.groupby(['lat_rounded', 'lon_rounded', 'time_window']).agg({
            't2m': ['mean', 'min', 'max', 'std'],
            'u10': 'mean',
            'v10': 'mean',
            'wind_speed': ['mean', 'max'],
            'wind_direction': 'mean'
        }).reset_index()
        
        # Flatten column names
        weather_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                              for col in weather_agg.columns.values]
        weather_agg.rename(columns={'lat_rounded': 'latitude', 'lon_rounded': 'longitude'}, 
                          inplace=True)
        
        print(f"   Aggregated {len(weather_agg)} weather records")
        return weather_agg
    
    def interpolate_topo_features(self, grid_points):
        """
        Interpolate topographic features to grid points.
        
        Args:
            grid_points: DataFrame with latitude and longitude
        """
        print("\nInterpolating topographic features...")
        
        from scipy.spatial.distance import cdist
        from scipy.interpolate import griddata
        
        # Round coordinates to grid
        grid_points['lat_rounded'] = round(grid_points['latitude'] / 0.01) * 0.01
        grid_points['lon_rounded'] = round(grid_points['longitude'] / 0.01) * 0.01
        
        # Get unique grid points
        unique_grid = grid_points[['lat_rounded', 'lon_rounded']].drop_duplicates()
        
        # Interpolate each topographic feature
        topo_features = ['elevation', 'slope', 'aspect', 'vegetation_cover', 
                        'vegetation_type', 'fuel_vegetation_cover', 'fuel_vegetation_height']
        
        topo_interp = unique_grid.copy()
        topo_interp.rename(columns={'lat_rounded': 'latitude', 'lon_rounded': 'longitude'}, 
                          inplace=True)
        
        for feature in topo_features:
            # Use nearest neighbor interpolation
            points = self.topo_df[['latitude', 'longitude']].values
            values = self.topo_df[feature].values
            grid_coords = unique_grid[['lat_rounded', 'lon_rounded']].values
            
            # Find nearest neighbors
            distances = cdist(grid_coords, points)
            nearest_indices = np.argmin(distances, axis=1)
            topo_interp[feature] = values[nearest_indices]
        
        print(f"   Interpolated {len(topo_interp)} topographic records")
        return topo_interp
    
    def merge_all_features(self, fire_agg, weather_agg, topo_interp, grid_points):
        """
        Merge all features into a unified dataset.
        
        Args:
            fire_agg: Aggregated fire features
            weather_agg: Aggregated weather features
            topo_interp: Interpolated topographic features
            grid_points: Grid points
        """
        print("\nMerging all features...")
        
        # Get all unique time windows
        time_windows = set()
        if fire_agg is not None and 'time_window' in fire_agg.columns:
            time_windows.update(fire_agg['time_window'].unique())
        if weather_agg is not None and 'time_window' in weather_agg.columns:
            time_windows.update(weather_agg['time_window'].unique())
        time_windows = sorted(list(time_windows))
        
        print(f"   Found {len(time_windows)} unique time windows")
        
        # Get unique grid points
        unique_grid = grid_points[['latitude', 'longitude']].drop_duplicates()
        
        # Create cartesian product of grid points and time windows
        merged_data = []
        for time_window in time_windows:
            grid_copy = unique_grid.copy()
            grid_copy['time_window'] = time_window
            merged_data.append(grid_copy)
        
        merged_df = pd.concat(merged_data, ignore_index=True)
        
        # Merge fire features
        if fire_agg is not None:
            merged_df = merged_df.merge(
                fire_agg,
                on=['latitude', 'longitude', 'time_window'],
                how='left'
            )
            # Fill missing fire values with 0 (no fire)
            fire_cols = [col for col in fire_agg.columns if col not in ['latitude', 'longitude', 'time_window']]
            merged_df[fire_cols] = merged_df[fire_cols].fillna(0)
        
        # Merge weather features
        if weather_agg is not None:
            merged_df = merged_df.merge(
                weather_agg,
                on=['latitude', 'longitude', 'time_window'],
                how='left'
            )
            # Forward fill weather data (persist last known values)
            weather_cols = [col for col in weather_agg.columns if col not in ['latitude', 'longitude', 'time_window']]
            merged_df[weather_cols] = merged_df.groupby(['latitude', 'longitude'])[weather_cols].ffill()
            merged_df[weather_cols] = merged_df[weather_cols].fillna(merged_df[weather_cols].mean())
        
        # Merge topographic features (static, no time component)
        if topo_interp is not None:
            merged_df = merged_df.merge(
                topo_interp,
                on=['latitude', 'longitude'],
                how='left'
            )
            # Fill missing topo values with mean
            topo_cols = [col for col in topo_interp.columns if col not in ['latitude', 'longitude']]
            merged_df[topo_cols] = merged_df[topo_cols].fillna(merged_df[topo_cols].mean())
        
        print(f"   Merged dataset shape: {merged_df.shape}")
        return merged_df
    
    def engineer_features(self, merged_df):
        """
        Engineer additional features.
        
        Args:
            merged_df: Merged dataframe
        """
        print("\nEngineering additional features...")
        
        df = merged_df.copy()
        
        # Temporal features
        if 'time_window' in df.columns:
            df['time_window'] = pd.to_datetime(df['time_window'])
            df['hour'] = df['time_window'].dt.hour
            df['day_of_year'] = df['time_window'].dt.dayofyear
            df['month'] = df['time_window'].dt.month
            df['day_of_week'] = df['time_window'].dt.dayofweek
        
        # Fire binary indicator
        if 'frp_count' in df.columns:
            df['has_fire'] = (df['frp_count'] > 0).astype(int)
        
        # Fire intensity categories
        if 'frp_mean' in df.columns:
            df['fire_intensity_low'] = ((df['frp_mean'] > 0) & (df['frp_mean'] <= 1)).astype(int)
            df['fire_intensity_medium'] = ((df['frp_mean'] > 1) & (df['frp_mean'] <= 5)).astype(int)
            df['fire_intensity_high'] = (df['frp_mean'] > 5).astype(int)
        
        # Wind features
        if 'wind_speed_mean' in df.columns:
            df['wind_speed_squared'] = df['wind_speed_mean'] ** 2
            df['high_wind'] = (df['wind_speed_mean'] > df['wind_speed_mean'].quantile(0.75)).astype(int)
        
        # Topographic features
        if 'slope' in df.columns:
            df['steep_slope'] = (df['slope'] > df['slope'].quantile(0.75)).astype(int)
        
        if 'elevation' in df.columns:
            df['high_elevation'] = (df['elevation'] > df['elevation'].quantile(0.75)).astype(int)
        
        # Vegetation-fuel interaction
        if 'vegetation_cover' in df.columns and 'fuel_vegetation_height' in df.columns:
            df['fuel_load_index'] = df['vegetation_cover'] * df['fuel_vegetation_height'] / 100
        
        print(f"   Added {len(df.columns) - len(merged_df.columns)} new features")
        return df
    
    def normalize_features(self, df, feature_groups=None):
        """
        Normalize features by group.
        
        Args:
            df: Dataframe to normalize
            feature_groups: Dictionary mapping group names to feature lists
        """
        print("\nNormalizing features...")
        
        if feature_groups is None:
            # Auto-detect feature groups
            feature_groups = {
                'fire': [col for col in df.columns if 'frp' in col.lower() or 'brightness' in col.lower()],
                'weather': [col for col in df.columns if any(x in col.lower() for x in ['t2m', 'u10', 'v10', 'wind'])],
                'topo': [col for col in df.columns if any(x in col.lower() for x in ['elevation', 'slope', 'aspect', 'vegetation', 'fuel'])]
            }
        
        df_normalized = df.copy()
        
        for group_name, features in feature_groups.items():
            if features:
                valid_features = [f for f in features if f in df.columns]
                if valid_features:
                    scaler = StandardScaler()
                    df_normalized[valid_features] = scaler.fit_transform(df[valid_features])
                    self.scalers[group_name] = scaler
                    print(f"   Normalized {len(valid_features)} {group_name} features")
        
        return df_normalized
    
    def prepare_sequences(self, df, sequence_length=7, prediction_horizon=1):
        """
        Prepare temporal sequences for time series prediction.
        
        Args:
            df: Preprocessed dataframe
            sequence_length: Number of time steps to use as input
            prediction_horizon: Number of time steps to predict ahead
        """
        print(f"\nPreparing sequences (length={sequence_length}, horizon={prediction_horizon})...")
        
        # Sort by location and time
        df_sorted = df.sort_values(['latitude', 'longitude', 'time_window'])
        
        sequences = []
        targets = []
        locations = []
        
        # Group by location
        for (lat, lon), group in df_sorted.groupby(['latitude', 'longitude']):
            group = group.reset_index(drop=True)
            
            # Create sequences
            for i in range(len(group) - sequence_length - prediction_horizon + 1):
                seq = group.iloc[i:i+sequence_length]
                target_idx = i + sequence_length + prediction_horizon - 1
                
                if target_idx < len(group):
                    target = group.iloc[target_idx]
                    
                    sequences.append(seq)
                    targets.append(target)
                    locations.append((lat, lon))
        
        print(f"   Created {len(sequences)} sequences")
        return sequences, targets, locations
    
    def save_processed_data(self, df, output_file='processed_data.pkl'):
        """Save processed data to file."""
        print(f"\nSaving processed data to {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(df, f)
        print("   Data saved successfully")
    
    def save_scalers(self, output_file='feature_scalers.pkl'):
        """Save feature scalers to file."""
        print(f"\nSaving feature scalers to {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(self.scalers, f)
        print("   Scalers saved successfully")


def main():
    """Main function to run data preprocessing."""
    print("="*60)
    print("STEP 2: DATA PREPROCESSING")
    print("="*60)
    
    # Load data (assuming data was loaded in step 1)
    try:
        import sys
        import importlib.util
        spec = importlib.util.spec_from_file_location("data_loading", "1_data_loading.py")
        data_loading = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_loading)
        loader = data_loading.DataLoader()
        fire_df, weather_df, topo_df = loader.load_all_datasets()
    except:
        # Fallback: load directly
        fire_df = pd.read_csv('fire_data.csv')
        weather_df = pd.read_csv('output_final_temp_celsius_fixed.csv')
        topo_df = pd.read_csv('topo_data_cleaned.csv')
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(fire_df, weather_df, topo_df)
    
    # Parse dates
    preprocessor.parse_dates()
    
    # Create spatial grid
    grid_points, lat_grid, lon_grid = preprocessor.create_spatial_grid(grid_resolution=0.01)
    
    # Aggregate features
    fire_agg = preprocessor.aggregate_fire_features(grid_points)
    weather_agg = preprocessor.aggregate_weather_features(grid_points)
    topo_interp = preprocessor.interpolate_topo_features(grid_points)
    
    # Merge all features
    merged_df = preprocessor.merge_all_features(fire_agg, weather_agg, topo_interp, grid_points)
    
    # Engineer features
    processed_df = preprocessor.engineer_features(merged_df)
    
    # Normalize features
    normalized_df = preprocessor.normalize_features(processed_df)
    
    # Save processed data
    preprocessor.save_processed_data(normalized_df)
    preprocessor.save_scalers()
    
    print("\n" + "="*60)
    print("DATA PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"\nFinal dataset shape: {normalized_df.shape}")
    print(f"Columns: {list(normalized_df.columns)}")
    
    return preprocessor, normalized_df


if __name__ == "__main__":
    preprocessor, processed_data = main()

