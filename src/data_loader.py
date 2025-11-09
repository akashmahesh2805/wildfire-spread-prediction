"""
Data loading and preprocessing utilities for wildfire spread prediction.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from datetime import datetime, timedelta


class WildfireDataLoader:
    """Load and preprocess wildfire, weather, and topographic data."""
    
    def __init__(self, 
                 fire_path: str = "fire_data.csv",
                 weather_path: str = "weather_data.csv",
                 topo_path: str = "topo_data_cleaned.csv"):
        """
        Initialize data loader.
        
        Args:
            fire_path: Path to fire detection data
            weather_path: Path to weather data
            topo_path: Path to topographic data
        """
        self.fire_path = fire_path
        self.weather_path = weather_path
        self.topo_path = topo_path
        self.combined_data = None
        
    def load_fire_data(self) -> pd.DataFrame:
        """Load and preprocess fire detection data."""
        fire = pd.read_csv(self.fire_path)
        
        # Parse datetime
        fire['acq_time'] = fire['acq_time'].astype(str).str.zfill(4)
        fire['acq_datetime'] = pd.to_datetime(
            fire['acq_date'] + ' ' + fire['acq_time'],
            format='%d-%m-%Y %H%M',
            errors='coerce'
        )
        fire = fire.dropna(subset=['acq_datetime', 'latitude', 'longitude'])
        
        # Round coordinates for merging
        fire['latitude'] = fire['latitude'].round(3)
        fire['longitude'] = fire['longitude'].round(3)
        
        # Create hourly time column
        fire['time'] = fire['acq_datetime'].dt.floor('h')
        
        # Select relevant columns
        fire_cols = ['latitude', 'longitude', 'brightness', 'frp', 'scan', 
                    'track', 'bright_t31', 'confidence', 'time', 'acq_datetime']
        fire = fire[[col for col in fire_cols if col in fire.columns]]
        
        return fire
    
    def load_weather_data(self) -> pd.DataFrame:
        """Load and preprocess weather data."""
        weather = pd.read_csv(self.weather_path)
        
        # Parse datetime
        weather['time'] = pd.to_datetime(weather['time'], errors='coerce')
        weather = weather.dropna(subset=['time'])
        weather = weather.sort_values('time').reset_index(drop=True)
        
        return weather
    
    def load_topo_data(self) -> pd.DataFrame:
        """Load and preprocess topographic data."""
        topo = pd.read_csv(self.topo_path)
        
        # Round coordinates
        topo['latitude'] = topo['latitude'].round(3)
        topo['longitude'] = topo['longitude'].round(3)
        
        # Remove duplicates
        topo = topo.drop_duplicates(subset=['latitude', 'longitude'])
        
        return topo
    
    def merge_data(self) -> pd.DataFrame:
        """
        Merge fire, weather, and topographic data.
        
        Returns:
            Combined dataframe with all features
        """
        fire = self.load_fire_data()
        weather = self.load_weather_data()
        topo = self.load_topo_data()
        
        # Merge fire with topographic data
        fire_topo = pd.merge(
            fire,
            topo,
            on=['latitude', 'longitude'],
            how='left'
        )
        
        # Merge with weather data (temporal merge)
        fire_topo = fire_topo.sort_values('time').reset_index(drop=True)
        weather = weather.sort_values('time').reset_index(drop=True)
        
        combined = pd.merge_asof(
            fire_topo,
            weather,
            on='time',
            direction='nearest'
        )
        
        self.combined_data = combined
        return combined
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from datetime."""
        df = df.copy()
        df['hour'] = df['time'].dt.hour
        df['day_of_year'] = df['time'].dt.dayofyear
        df['month'] = df['time'].dt.month
        df['day_of_week'] = df['time'].dt.dayofweek
        
        # Cyclical encoding for temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        return df
    
    def create_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create spatial features from coordinates."""
        df = df.copy()
        
        # Normalize coordinates (optional: use min-max scaling)
        df['lat_norm'] = (df['latitude'] - df['latitude'].min()) / (df['latitude'].max() - df['latitude'].min())
        df['lon_norm'] = (df['longitude'] - df['longitude'].min()) / (df['longitude'].max() - df['longitude'].min())
        
        return df
    
    def prepare_features(self, 
                        include_temporal: bool = True,
                        include_spatial: bool = True) -> pd.DataFrame:
        """
        Prepare final feature set.
        
        Args:
            include_temporal: Whether to include temporal features
            include_spatial: Whether to include spatial features
            
        Returns:
            Dataframe with prepared features
        """
        if self.combined_data is None:
            df = self.merge_data()
        else:
            df = self.combined_data.copy()
        
        if include_temporal:
            df = self.create_temporal_features(df)
        
        if include_spatial:
            df = self.create_spatial_features(df)
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> Dict[str, list]:
        """
        Get feature columns grouped by modality.
        
        Returns:
            Dictionary with keys: 'fire', 'weather', 'terrain', 'temporal', 'spatial'
        """
        feature_groups = {
            'fire': ['brightness', 'frp', 'scan', 'track', 'bright_t31'],
            'weather': ['temperature_2m', 'relative_humidity_2m', 
                       'wind_speed_10m', 'wind_direction_10m', 'precipitation'],
            'terrain': ['elevation', 'slope', 'aspect', 'vegetation_cover', 
                       'vegetation_type', 'fuel_vegetation_cover', 'fuel_vegetation_height'],
            'temporal': ['hour_sin', 'hour_cos', 'day_sin', 'day_cos'],
            'spatial': ['lat_norm', 'lon_norm']
        }
        
        # Filter to only columns that exist in dataframe
        available_features = {}
        for group, cols in feature_groups.items():
            available = [col for col in cols if col in df.columns]
            if available:
                available_features[group] = available
        
        return available_features

