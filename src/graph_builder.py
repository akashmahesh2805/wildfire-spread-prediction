"""
Graph construction utilities for spatial-temporal wildfire data.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from scipy.spatial.distance import cdist
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected


class SpatialTemporalGraphBuilder:
    """Build spatial-temporal graphs from wildfire data."""
    
    def __init__(self,
                 spatial_threshold: float = 0.05,  # degrees (~5.5 km)
                 temporal_window: int = 1):  # hours
        """
        Initialize graph builder.
        
        Args:
            spatial_threshold: Maximum distance (in degrees) for spatial edges
            temporal_window: Time window (in hours) for temporal edges
        """
        self.spatial_threshold = spatial_threshold
        self.temporal_window = temporal_window
    
    def compute_spatial_edges(self, 
                             coordinates: np.ndarray,
                             spatial_threshold: Optional[float] = None) -> np.ndarray:
        """
        Compute spatial edges based on distance.
        
        Args:
            coordinates: Array of shape (N, 2) with [latitude, longitude]
            spatial_threshold: Distance threshold (overrides default if provided)
            
        Returns:
            Edge index array of shape (2, E) with spatial edges
        """
        if spatial_threshold is None:
            spatial_threshold = self.spatial_threshold
        
        # Compute pairwise distances
        distances = cdist(coordinates, coordinates, metric='euclidean')
        
        # Create edges for pairs within threshold
        edge_indices = np.where((distances > 0) & (distances <= spatial_threshold))
        
        # Create edge_index in PyTorch Geometric format [2, E]
        edge_index = np.array([edge_indices[0], edge_indices[1]], dtype=np.int64)
        
        return edge_index
    
    def compute_temporal_edges(self,
                              timestamps: pd.Series,
                              temporal_window: Optional[int] = None) -> np.ndarray:
        """
        Compute temporal edges connecting consecutive time steps.
        
        Args:
            timestamps: Series of timestamps
            temporal_window: Time window in hours (overrides default if provided)
            
        Returns:
            Edge index array of shape (2, E) with temporal edges
        """
        if temporal_window is None:
            temporal_window = self.temporal_window
        
        # Sort by time and get indices as numpy array
        sorted_indices = timestamps.argsort().values  # Convert to numpy array
        sorted_times = timestamps.iloc[sorted_indices].values  # Convert to numpy array
        
        edge_list = []
        
        # Connect nodes within temporal window
        for i in range(len(sorted_times)):
            t1 = pd.Timestamp(sorted_times[i])
            for j in range(i + 1, len(sorted_times)):
                t2 = pd.Timestamp(sorted_times[j])
                time_diff = (t2 - t1).total_seconds() / 3600  # hours
                
                if time_diff <= temporal_window:
                    # Connect both directions
                    edge_list.append([sorted_indices[i], sorted_indices[j]])
                    edge_list.append([sorted_indices[j], sorted_indices[i]])
                else:
                    break  # Times are sorted, so we can break
        
        if len(edge_list) == 0:
            return np.array([[], []], dtype=np.int64)
        
        edge_index = np.array(edge_list, dtype=np.int64).T
        return edge_index
    
    def build_node_features(self,
                           df: pd.DataFrame,
                           feature_groups: dict) -> torch.Tensor:
        """
        Build node feature matrix from dataframe.
        
        Args:
            df: Dataframe with features
            feature_groups: Dictionary mapping modality to feature columns
            
        Returns:
            Feature tensor of shape (N, F)
        """
        feature_list = []
        
        # Collect features from all modalities
        for group, cols in feature_groups.items():
            for col in cols:
                if col in df.columns:
                    values = df[col].values
                    # Handle missing values
                    if pd.isna(values).any():
                        values = pd.Series(values).fillna(0).values
                    feature_list.append(values)
        
        if len(feature_list) == 0:
            raise ValueError("No features found!")
        
        # Stack into feature matrix
        features = np.column_stack(feature_list)
        
        # Normalize features
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        return torch.FloatTensor(features)
    
    def create_targets(self,
                      df: pd.DataFrame,
                      prediction_horizon: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create prediction targets (future fire locations/intensities).
        
        Args:
            df: Dataframe sorted by time
            prediction_horizon: Hours ahead to predict
            
        Returns:
            Tuple of (target_locations, target_intensities)
        """
        # Sort by time and location
        df_sorted = df.sort_values(['time', 'latitude', 'longitude']).reset_index(drop=True)
        
        target_locations = []
        target_intensities = []
        valid_indices = []
        
        for idx, row in df_sorted.iterrows():
            current_time = row['time']
            future_time = current_time + pd.Timedelta(hours=prediction_horizon)
            
            # Round coordinates for comparison (row values are floats, not Series)
            lat_rounded = round(row['latitude'], 3)
            lon_rounded = round(row['longitude'], 3)
            
            # Find fire events at same location in future
            future_events = df_sorted[
                (df_sorted['latitude'].round(3) == lat_rounded) &
                (df_sorted['longitude'].round(3) == lon_rounded) &
                (df_sorted['time'] == future_time)
            ]
            
            if len(future_events) > 0:
                # Fire continues at this location
                target_locations.append(1.0)
                target_intensities.append(future_events.iloc[0]['frp'] if 'frp' in future_events.columns else 0.0)
                valid_indices.append(idx)
            else:
                # Check if fire spreads to nearby locations
                nearby_future = df_sorted[
                    (df_sorted['time'] == future_time) &
                    (np.abs(df_sorted['latitude'] - row['latitude']) <= 0.05) &
                    (np.abs(df_sorted['longitude'] - row['longitude']) <= 0.05)
                ]
                
                if len(nearby_future) > 0:
                    target_locations.append(1.0)
                    target_intensities.append(nearby_future.iloc[0]['frp'] if 'frp' in nearby_future.columns else 0.0)
                    valid_indices.append(idx)
                else:
                    # No fire in future
                    target_locations.append(0.0)
                    target_intensities.append(0.0)
                    valid_indices.append(idx)
        
        target_locations = torch.FloatTensor(target_locations)
        target_intensities = torch.FloatTensor(target_intensities)
        
        return target_locations, target_intensities
    
    def build_graph(self,
                   df: pd.DataFrame,
                   feature_groups: dict,
                   include_temporal: bool = True,
                   include_spatial: bool = True,
                   max_edges: Optional[int] = None,
                   sample_nodes: Optional[int] = None) -> Data:
        """
        Build complete spatial-temporal graph.
        
        Args:
            df: Dataframe with all features
            feature_groups: Dictionary mapping modality to feature columns
            include_temporal: Whether to include temporal edges
            include_spatial: Whether to include spatial edges
            max_edges: Maximum number of edges (samples if exceeded)
            sample_nodes: If provided, randomly sample this many nodes
            
        Returns:
            PyTorch Geometric Data object
        """
        # Sample nodes if requested (for memory efficiency)
        sampled_df = df.copy()
        if sample_nodes is not None and len(df) > sample_nodes:
            sampled_df = df.sample(n=min(sample_nodes, len(df)), random_state=42).reset_index(drop=True)
            print(f"  Sampled {len(sampled_df)} nodes from {len(df)} total nodes")
        
        # Build node features
        node_features = self.build_node_features(sampled_df, feature_groups)
        
        # Get coordinates and timestamps
        coordinates = sampled_df[['latitude', 'longitude']].values
        timestamps = sampled_df['time']
        
        # Build edges
        edge_list = []
        
        if include_spatial:
            spatial_edges = self.compute_spatial_edges(coordinates)
            edge_list.append(spatial_edges)
            print(f"  Spatial edges: {spatial_edges.shape[1]}")
        
        if include_temporal:
            temporal_edges = self.compute_temporal_edges(timestamps)
            if temporal_edges.shape[1] > 0:
                edge_list.append(temporal_edges)
                print(f"  Temporal edges: {temporal_edges.shape[1]}")
        
        # Combine edges
        if len(edge_list) > 0:
            edge_index = np.concatenate(edge_list, axis=1)
            # Remove duplicate edges
            edge_index = np.unique(edge_index, axis=1)
            
            # Sample edges if too many (for memory efficiency)
            if max_edges is not None and edge_index.shape[1] > max_edges:
                print(f"  Sampling {max_edges} edges from {edge_index.shape[1]} total edges")
                sample_idx = np.random.choice(edge_index.shape[1], max_edges, replace=False)
                edge_index = edge_index[:, sample_idx]
            
            edge_index = torch.LongTensor(edge_index)
            edge_index = to_undirected(edge_index)  # Make undirected
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Create positions (coordinates) for visualization
        pos = torch.FloatTensor(coordinates)
        
        # Store sampled dataframe reference for target creation
        data = Data(
            x=node_features,
            edge_index=edge_index,
            pos=pos
        )
        # Store the sampled dataframe for later use (optional)
        data.sampled_df = sampled_df
        
        return data
    
    def split_temporal(self,
                      df: pd.DataFrame,
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally (not randomly).
        
        Args:
            df: Dataframe sorted by time
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        df_sorted = df.sort_values('time').reset_index(drop=True)
        
        n = len(df_sorted)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df_sorted.iloc[:train_end]
        val_df = df_sorted.iloc[train_end:val_end]
        test_df = df_sorted.iloc[val_end:]
        
        return train_df, val_df, test_df

