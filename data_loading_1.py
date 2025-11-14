"""
Step 1: Data Loading and Exploration
=====================================
This module loads the three datasets (fire, weather, topographic) and performs
exploratory data analysis to understand the data structure, distributions, and quality.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class DataLoader:
    """Class to load and explore wildfire datasets."""
    
    def __init__(self, data_dir='.'):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Directory containing the CSV files
        """
        self.data_dir = data_dir
        self.fire_df = None
        self.weather_df = None
        self.topo_df = None
        
    def load_all_datasets(self):
        """Load all three datasets."""
        print("Loading datasets...")
        
        # Load fire data
        print("\n1. Loading fire_data.csv...")
        self.fire_df = pd.read_csv(os.path.join(self.data_dir, 'fire_data.csv'))
        print(f"   Shape: {self.fire_df.shape}")
        print(f"   Columns: {list(self.fire_df.columns)}")
        
        # Load weather data
        print("\n2. Loading output_final_temp_celsius_fixed.csv...")
        self.weather_df = pd.read_csv(os.path.join(self.data_dir, 'output_final_temp_celsius_fixed.csv'))
        print(f"   Shape: {self.weather_df.shape}")
        print(f"   Columns: {list(self.weather_df.columns)}")
        
        # Load topographic data
        print("\n3. Loading topo_data_cleaned.csv...")
        self.topo_df = pd.read_csv(os.path.join(self.data_dir, 'topo_data_cleaned.csv'))
        print(f"   Shape: {self.topo_df.shape}")
        print(f"   Columns: {list(self.topo_df.columns)}")
        
        return self.fire_df, self.weather_df, self.topo_df
    
    def explore_fire_data(self):
        """Explore fire detection dataset."""
        print("\n" + "="*60)
        print("FIRE DATA EXPLORATION")
        print("="*60)
        
        print("\nBasic Information:")
        print(self.fire_df.info())
        
        print("\nFirst few rows:")
        print(self.fire_df.head())
        
        print("\nStatistical Summary:")
        print(self.fire_df.describe())
        
        print("\nMissing Values:")
        print(self.fire_df.isnull().sum())
        
        print("\nTemporal Range:")
        # Parse dates
        self.fire_df['acq_date_parsed'] = pd.to_datetime(self.fire_df['acq_date'], format='%d-%m-%Y')
        print(f"   Start Date: {self.fire_df['acq_date_parsed'].min()}")
        print(f"   End Date: {self.fire_df['acq_date_parsed'].max()}")
        print(f"   Total Days: {(self.fire_df['acq_date_parsed'].max() - self.fire_df['acq_date_parsed'].min()).days}")
        
        print("\nSpatial Range:")
        print(f"   Latitude: [{self.fire_df['latitude'].min():.4f}, {self.fire_df['latitude'].max():.4f}]")
        print(f"   Longitude: [{self.fire_df['longitude'].min():.4f}, {self.fire_df['longitude'].max():.4f}]")
        
        print("\nFire Characteristics:")
        print(f"   Unique fire detections: {len(self.fire_df)}")
        print(f"   Average FRP: {self.fire_df['frp'].mean():.2f} MW")
        print(f"   Max FRP: {self.fire_df['frp'].max():.2f} MW")
        print(f"   Average Brightness: {self.fire_df['brightness'].mean():.2f} K")
        
        print("\nCategorical Variables:")
        print(f"   Satellites: {self.fire_df['satellite'].unique()}")
        print(f"   Instruments: {self.fire_df['instrument'].unique()}")
        print(f"   Day/Night: {self.fire_df['daynight'].value_counts().to_dict()}")
        print(f"   Fire Types: {self.fire_df['type'].value_counts().to_dict()}")
        
        return self.fire_df
    
    def explore_weather_data(self):
        """Explore weather dataset."""
        print("\n" + "="*60)
        print("WEATHER DATA EXPLORATION")
        print("="*60)
        
        print("\nBasic Information:")
        print(self.weather_df.info())
        
        print("\nFirst few rows:")
        print(self.weather_df.head())
        
        print("\nStatistical Summary:")
        print(self.weather_df.describe())
        
        print("\nMissing Values:")
        print(self.weather_df.isnull().sum())
        
        print("\nTemporal Range:")
        # Parse dates
        self.weather_df['date_parsed'] = pd.to_datetime(self.weather_df['date'], format='%d-%m-%Y')
        print(f"   Start Date: {self.weather_df['date_parsed'].min()}")
        print(f"   End Date: {self.weather_df['date_parsed'].max()}")
        print(f"   Total Days: {(self.weather_df['date_parsed'].max() - self.weather_df['date_parsed'].min()).days}")
        
        print("\nSpatial Range:")
        print(f"   Latitude: [{self.weather_df['latitude'].min():.4f}, {self.weather_df['latitude'].max():.4f}]")
        print(f"   Longitude: [{self.weather_df['longitude'].min():.4f}, {self.weather_df['longitude'].max():.4f}]")
        
        print("\nWeather Characteristics:")
        print(f"   Average Temperature: {self.weather_df['t2m'].mean():.2f} °C")
        print(f"   Temperature Range: [{self.weather_df['t2m'].min():.2f}, {self.weather_df['t2m'].max():.2f}] °C")
        print(f"   Average Wind U: {self.weather_df['u10'].mean():.2f} m/s")
        print(f"   Average Wind V: {self.weather_df['v10'].mean():.2f} m/s")
        
        # Calculate wind speed
        wind_speed = np.sqrt(self.weather_df['u10']**2 + self.weather_df['v10']**2)
        print(f"   Average Wind Speed: {wind_speed.mean():.2f} m/s")
        print(f"   Max Wind Speed: {wind_speed.max():.2f} m/s")
        
        return self.weather_df
    
    def explore_topo_data(self):
        """Explore topographic dataset."""
        print("\n" + "="*60)
        print("TOPOGRAPHIC DATA EXPLORATION")
        print("="*60)
        
        print("\nBasic Information:")
        print(self.topo_df.info())
        
        print("\nFirst few rows:")
        print(self.topo_df.head())
        
        print("\nStatistical Summary:")
        print(self.topo_df.describe())
        
        print("\nMissing Values:")
        print(self.topo_df.isnull().sum())
        
        print("\nSpatial Range:")
        print(f"   Latitude: [{self.topo_df['latitude'].min():.4f}, {self.topo_df['latitude'].max():.4f}]")
        print(f"   Longitude: [{self.topo_df['longitude'].min():.4f}, {self.topo_df['longitude'].max():.4f}]")
        
        print("\nTopographic Characteristics:")
        print(f"   Elevation Range: [{self.topo_df['elevation'].min():.2f}, {self.topo_df['elevation'].max():.2f}] m")
        print(f"   Average Elevation: {self.topo_df['elevation'].mean():.2f} m")
        print(f"   Average Slope: {self.topo_df['slope'].mean():.2f} degrees")
        print(f"   Average Aspect: {self.topo_df['aspect'].mean():.2f} degrees")
        print(f"   Average Vegetation Cover: {self.topo_df['vegetation_cover'].mean():.2f}%")
        print(f"   Average Fuel Height: {self.topo_df['fuel_vegetation_height'].mean():.2f} m")
        
        print("\nUnique Vegetation Types:")
        print(f"   Count: {self.topo_df['vegetation_type'].nunique()}")
        print(f"   Top 10: {self.topo_df['vegetation_type'].value_counts().head(10).to_dict()}")
        
        return self.topo_df
    
    def visualize_data_distributions(self, save_dir='plots'):
        """Create visualization plots for data exploration."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Fire data visualizations
        if self.fire_df is not None:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # FRP distribution
            axes[0, 0].hist(self.fire_df['frp'], bins=50, edgecolor='black')
            axes[0, 0].set_xlabel('Fire Radiative Power (MW)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Distribution of Fire Radiative Power')
            axes[0, 0].set_yscale('log')
            
            # Brightness distribution
            axes[0, 1].hist(self.fire_df['brightness'], bins=50, edgecolor='black', color='orange')
            axes[0, 1].set_xlabel('Brightness Temperature (K)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Distribution of Brightness Temperature')
            
            # Temporal distribution
            fire_counts = self.fire_df.groupby('acq_date_parsed').size()
            axes[1, 0].plot(fire_counts.index, fire_counts.values)
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Number of Fire Detections')
            axes[1, 0].set_title('Fire Detections Over Time')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Spatial scatter
            axes[1, 1].scatter(self.fire_df['longitude'], self.fire_df['latitude'], 
                             c=self.fire_df['frp'], cmap='YlOrRd', s=1, alpha=0.5)
            axes[1, 1].set_xlabel('Longitude')
            axes[1, 1].set_ylabel('Latitude')
            axes[1, 1].set_title('Fire Locations (colored by FRP)')
            plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='FRP (MW)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'fire_data_exploration.png'), dpi=300, bbox_inches='tight')
            print(f"\nSaved fire data visualization to {save_dir}/fire_data_exploration.png")
            plt.close()
        
        # Weather data visualizations
        if self.weather_df is not None:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Temperature distribution
            axes[0, 0].hist(self.weather_df['t2m'], bins=50, edgecolor='black', color='blue')
            axes[0, 0].set_xlabel('Temperature (°C)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Distribution of Temperature')
            
            # Wind speed distribution
            wind_speed = np.sqrt(self.weather_df['u10']**2 + self.weather_df['v10']**2)
            axes[0, 1].hist(wind_speed, bins=50, edgecolor='black', color='green')
            axes[0, 1].set_xlabel('Wind Speed (m/s)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Distribution of Wind Speed')
            
            # Temperature over time (sample)
            sample_weather = self.weather_df.sample(min(10000, len(self.weather_df)))
            axes[1, 0].scatter(sample_weather['date_parsed'], sample_weather['t2m'], 
                             s=1, alpha=0.3)
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Temperature (°C)')
            axes[1, 0].set_title('Temperature Over Time (Sample)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Wind vectors (sample)
            sample_weather = self.weather_df.sample(min(5000, len(self.weather_df)))
            axes[1, 1].quiver(sample_weather['longitude'], sample_weather['latitude'],
                            sample_weather['u10'], sample_weather['v10'],
                            scale=20, width=0.002, alpha=0.5)
            axes[1, 1].set_xlabel('Longitude')
            axes[1, 1].set_ylabel('Latitude')
            axes[1, 1].set_title('Wind Vectors (Sample)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'weather_data_exploration.png'), dpi=300, bbox_inches='tight')
            print(f"Saved weather data visualization to {save_dir}/weather_data_exploration.png")
            plt.close()
        
        # Topographic data visualizations
        if self.topo_df is not None:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Elevation distribution
            axes[0, 0].hist(self.topo_df['elevation'], bins=50, edgecolor='black', color='brown')
            axes[0, 0].set_xlabel('Elevation (m)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Distribution of Elevation')
            
            # Slope distribution
            axes[0, 1].hist(self.topo_df['slope'], bins=50, edgecolor='black', color='gray')
            axes[0, 1].set_xlabel('Slope (degrees)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Distribution of Slope')
            
            # Elevation map
            scatter = axes[1, 0].scatter(self.topo_df['longitude'], self.topo_df['latitude'],
                                       c=self.topo_df['elevation'], cmap='terrain', s=1)
            axes[1, 0].set_xlabel('Longitude')
            axes[1, 0].set_ylabel('Latitude')
            axes[1, 0].set_title('Elevation Map')
            plt.colorbar(scatter, ax=axes[1, 0], label='Elevation (m)')
            
            # Vegetation cover map
            scatter = axes[1, 1].scatter(self.topo_df['longitude'], self.topo_df['latitude'],
                                       c=self.topo_df['vegetation_cover'], cmap='YlGn', s=1)
            axes[1, 1].set_xlabel('Longitude')
            axes[1, 1].set_ylabel('Latitude')
            axes[1, 1].set_title('Vegetation Cover Map')
            plt.colorbar(scatter, ax=axes[1, 1], label='Vegetation Cover (%)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'topo_data_exploration.png'), dpi=300, bbox_inches='tight')
            print(f"Saved topographic data visualization to {save_dir}/topo_data_exploration.png")
            plt.close()
    
    def generate_summary_report(self, output_file='data_summary_report.txt'):
        """Generate a comprehensive summary report."""
        with open(output_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("WILDFIRE DATA SUMMARY REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write("DATASET OVERVIEW\n")
            f.write("-"*60 + "\n")
            f.write(f"Fire Data: {self.fire_df.shape[0]} records, {self.fire_df.shape[1]} features\n")
            f.write(f"Weather Data: {self.weather_df.shape[0]} records, {self.weather_df.shape[1]} features\n")
            f.write(f"Topographic Data: {self.topo_df.shape[0]} records, {self.topo_df.shape[1]} features\n\n")
            
            f.write("SPATIAL COVERAGE\n")
            f.write("-"*60 + "\n")
            f.write(f"Fire Data - Lat: [{self.fire_df['latitude'].min():.4f}, {self.fire_df['latitude'].max():.4f}], "
                   f"Lon: [{self.fire_df['longitude'].min():.4f}, {self.fire_df['longitude'].max():.4f}]\n")
            f.write(f"Weather Data - Lat: [{self.weather_df['latitude'].min():.4f}, {self.weather_df['latitude'].max():.4f}], "
                   f"Lon: [{self.weather_df['longitude'].min():.4f}, {self.weather_df['longitude'].max():.4f}]\n")
            f.write(f"Topo Data - Lat: [{self.topo_df['latitude'].min():.4f}, {self.topo_df['latitude'].max():.4f}], "
                   f"Lon: [{self.topo_df['longitude'].min():.4f}, {self.topo_df['longitude'].max():.4f}]\n\n")
            
            f.write("TEMPORAL COVERAGE\n")
            f.write("-"*60 + "\n")
            f.write(f"Fire Data: {self.fire_df['acq_date_parsed'].min()} to {self.fire_df['acq_date_parsed'].max()}\n")
            f.write(f"Weather Data: {self.weather_df['date_parsed'].min()} to {self.weather_df['date_parsed'].max()}\n\n")
            
            f.write("DATA QUALITY\n")
            f.write("-"*60 + "\n")
            f.write(f"Fire Data Missing Values: {self.fire_df.isnull().sum().sum()}\n")
            f.write(f"Weather Data Missing Values: {self.weather_df.isnull().sum().sum()}\n")
            f.write(f"Topo Data Missing Values: {self.topo_df.isnull().sum().sum()}\n")
        
        print(f"\nSummary report saved to {output_file}")


def main():
    """Main function to run data loading and exploration."""
    print("="*60)
    print("STEP 1: DATA LOADING AND EXPLORATION")
    print("="*60)
    
    # Initialize loader
    loader = DataLoader()
    
    # Load all datasets
    fire_df, weather_df, topo_df = loader.load_all_datasets()
    
    # Explore each dataset
    loader.explore_fire_data()
    loader.explore_weather_data()
    loader.explore_topo_data()
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    loader.visualize_data_distributions()
    
    # Generate summary report
    loader.generate_summary_report()
    
    print("\n" + "="*60)
    print("DATA LOADING COMPLETE!")
    print("="*60)
    
    return loader


if __name__ == "__main__":
    loader = main()

