"""
Grid-based graph construction for wildfire spread prediction.
Creates nodes for grid cells instead of individual fire events.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from scipy.spatial.distance import cdist
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


class GridBasedGraphBuilder:
    """Build grid-based spatial-temporal graphs from wildfire data."""
    
    def __init__(self,
                 grid_size: float = 0.01,  # degrees (~1.1 km)
                 temporal_window: int = 1):  # hours
        """
        Initialize grid-based graph builder.
        
        Args:
            grid_size: Size of grid cells in degrees
            temporal_window: Time window (in hours) for temporal edges
        """
        self.grid_size = grid_size
        self.temporal_window = temporal_window
        self.grid_cells = None
        self.cell_to_index = None
    
    def create_grid(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        Create grid cells covering the study area.
        
        Args:
            df: Dataframe with latitude and longitude
            
        Returns:
            Tuple of (grid_centers, cell_to_index_map)
        """
        min_lat = df['latitude'].min()
        max_lat = df['latitude'].max()
        min_lon = df['longitude'].min()
        max_lon = df['longitude'].max()
        
        # Create grid
        lat_range = np.arange(min_lat, max_lat + self.grid_size, self.grid_size)
        lon_range = np.arange(min_lon, max_lon + self.grid_size, self.grid_size)
        
        grid_centers = []
        cell_to_index = {}
        idx = 0
        
        for lat in lat_range:
            for lon in lon_range:
                # Round to grid cell center
                cell_lat = round(lat / self.grid_size) * self.grid_size
                cell_lon = round(lon / self.grid_size) * self.grid_size
                cell_key = (cell_lat, cell_lon)
                
                if cell_key not in cell_to_index:
                    grid_centers.append([cell_lat, cell_lon])
                    cell_to_index[cell_key] = idx
                    idx += 1
        
        grid_centers = np.array(grid_centers)
        self.grid_cells = grid_centers
        self.cell_to_index = cell_to_index
        
        return grid_centers, cell_to_index
    
    def assign_events_to_grid(self, df: pd.DataFrame, use_existing_grid: bool = True) -> pd.DataFrame:
        """
        Assign fire events to grid cells.
        
        Args:
            df: Dataframe with fire events
            use_existing_grid: If True and grid exists, use it. Otherwise create new grid.
            
        Returns:
            Dataframe with grid cell assignments
        """
        df = df.copy()
        
        # Create grid if not exists or if explicitly requested
        if self.grid_cells is None or not use_existing_grid:
            self.create_grid(df)
        
        if self.grid_cells is None:
            raise ValueError("Grid not created. Call create_grid() first.")
        
        # Assign each event to nearest grid cell
        event_coords = df[['latitude', 'longitude']].values
        distances = cdist(event_coords, self.grid_cells, metric='euclidean')
        nearest_cell_idx = np.argmin(distances, axis=1)
        
        df['grid_lat'] = self.grid_cells[nearest_cell_idx, 0]
        df['grid_lon'] = self.grid_cells[nearest_cell_idx, 1]
        df['grid_cell_idx'] = nearest_cell_idx
        
        return df
    
    def aggregate_features(self, df: pd.DataFrame, feature_groups: Dict) -> np.ndarray:
        """
        Aggregate features for each grid cell.
        
        Args:
            df: Dataframe with grid assignments
            feature_groups: Dictionary mapping modality to feature columns
            
        Returns:
            Feature matrix of shape (num_cells, num_features)
        """
        num_cells = len(self.grid_cells)
        feature_list = []
        feature_names = []
        
        # Collect all feature columns
        for group, cols in feature_groups.items():
            for col in cols:
                if col in df.columns:
                    feature_names.append(col)
        
        # Aggregate features per grid cell
        for cell_idx in range(num_cells):
            cell_data = df[df['grid_cell_idx'] == cell_idx]
            
            if len(cell_data) == 0:
                # Empty cell - use zeros or default values
                cell_features = [0.0] * len(feature_names)
            else:
                # Aggregate: mean for continuous, max for FRP (preserve intensity), mode for categorical
                cell_features = []
                for col in feature_names:
                    if col in cell_data.columns:
                        if col == 'frp':
                            # For FRP, use MAX to preserve fire intensity (not mean)
                            val = cell_data[col].max()
                        elif cell_data[col].dtype in ['int64', 'float64']:
                            # Mean for other numerical features
                            val = cell_data[col].mean()
                        else:
                            # Mode for categorical, or first value
                            val = cell_data[col].iloc[0] if len(cell_data) > 0 else 0
                        cell_features.append(float(val) if not pd.isna(val) else 0.0)
                    else:
                        cell_features.append(0.0)
            
            feature_list.append(cell_features)
        
        features = np.array(feature_list)
        
        # Normalize features
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        return features
    
    def compute_spatial_edges(self) -> np.ndarray:
        """
        Compute spatial edges connecting adjacent grid cells.
        
        Returns:
            Edge index array of shape (2, E)
        """
        if self.grid_cells is None:
            raise ValueError("Grid not created. Call create_grid() first.")
        
        edge_list = []
        num_cells = len(self.grid_cells)
        
        # Connect adjacent cells (within 1.5 * grid_size to include diagonals)
        threshold = 1.5 * self.grid_size
        distances = cdist(self.grid_cells, self.grid_cells, metric='euclidean')
        
        # Create edges for adjacent cells
        for i in range(num_cells):
            for j in range(i + 1, num_cells):
                if distances[i, j] <= threshold:
                    edge_list.append([i, j])
                    edge_list.append([j, i])  # Undirected
        
        if len(edge_list) == 0:
            return np.array([[], []], dtype=np.int64)
        
        edge_index = np.array(edge_list, dtype=np.int64).T
        return edge_index
    
    def compute_temporal_edges(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute temporal edges for grid cells across time steps.
        For grid-based: connects cells that have fires at consecutive times.
        
        Args:
            df: Dataframe with grid assignments and timestamps
            
        Returns:
            Edge index array of shape (2, E)
        """
        if self.grid_cells is None:
            raise ValueError("Grid not created. Call create_grid() first.")
        
        # Get unique time steps
        unique_times = sorted(df['time'].unique())
        num_cells = len(self.grid_cells)
        edge_list = []
        
        # Connect cells that have fires at consecutive time steps
        for t_idx in range(len(unique_times) - 1):
            current_time = unique_times[t_idx]
            next_time = unique_times[t_idx + 1]
            
            time_diff = (next_time - current_time).total_seconds() / 3600  # hours
            
            if time_diff <= self.temporal_window:
                # Get cells with fires at current and next time
                current_cells = set(df[df['time'] == current_time]['grid_cell_idx'].unique())
                next_cells = set(df[df['time'] == next_time]['grid_cell_idx'].unique())
                
                # Connect cells that have fires at both times, or same cell across time
                for cell_idx in current_cells | next_cells:
                    if cell_idx in current_cells and cell_idx in next_cells:
                        # Same cell has fire at both times
                        edge_list.append([cell_idx, cell_idx])
                    elif cell_idx in current_cells:
                        # Fire spreads from this cell - connect to nearby cells with fires
                        for next_cell in next_cells:
                            # Check if cells are adjacent
                            dist = np.sqrt(np.sum((self.grid_cells[cell_idx] - self.grid_cells[next_cell])**2))
                            if dist <= self.grid_size * 2:  # Within 2 cells
                                edge_list.append([cell_idx, next_cell])
                                edge_list.append([next_cell, cell_idx])
        
        if len(edge_list) == 0:
            return np.array([[], []], dtype=np.int64)
        
        edge_index = np.array(edge_list, dtype=np.int64).T
        # Remove duplicates
        edge_index = np.unique(edge_index, axis=1)
        return edge_index
    
    def create_targets(self, df: pd.DataFrame, prediction_horizon: int = 1) -> torch.Tensor:
        """
        Create prediction targets for grid cells.
        Predicts maximum fire intensity in each cell at future time.
        
        Args:
            df: Dataframe with grid assignments and timestamps
            prediction_horizon: Hours ahead to predict
            
        Returns:
            Target tensor of shape (num_cells,)
        """
        if self.grid_cells is None:
            raise ValueError("Grid not created. Call create_grid() or build_graph() first.")
        
        # Ensure df has grid assignments
        if 'grid_cell_idx' not in df.columns:
            df = self.assign_events_to_grid(df)
        
        df_sorted = df.sort_values(['time', 'grid_cell_idx']).reset_index(drop=True)
        num_cells = len(self.grid_cells)
        
        # Group by time and grid cell
        targets = []
        
        # Get all unique times
        unique_times = sorted(df_sorted['time'].unique())
        
        for cell_idx in range(num_cells):
            # Find all times when this cell has fires
            cell_times = df_sorted[df_sorted['grid_cell_idx'] == cell_idx]['time'].unique()
            
            if len(cell_times) == 0:
                # No fires in this cell - predict 0
                targets.append(0.0)
                continue
            
            # For each time with fire in this cell, check future
            cell_targets = []
            for current_time in cell_times:
                future_time = current_time + pd.Timedelta(hours=prediction_horizon)
                
                # Find fires in this cell at future time
                future_fires = df_sorted[
                    (df_sorted['grid_cell_idx'] == cell_idx) &
                    (df_sorted['time'] == future_time)
                ]
                
                if len(future_fires) > 0 and 'frp' in future_fires.columns:
                    # Maximum FRP in cell at future time
                    cell_targets.append(future_fires['frp'].max())
                else:
                    # Check nearby cells for fire spread
                    cell_coords = self.grid_cells[cell_idx]
                    nearby_future = df_sorted[
                        (df_sorted['time'] == future_time) &
                        (np.abs(df_sorted['latitude'] - cell_coords[0]) <= self.grid_size * 2) &
                        (np.abs(df_sorted['longitude'] - cell_coords[1]) <= self.grid_size * 2)
                    ]
                    
                    if len(nearby_future) > 0 and 'frp' in nearby_future.columns:
                        cell_targets.append(nearby_future['frp'].max())
                    else:
                        cell_targets.append(0.0)
            
            # Use maximum target across all time steps for this cell
            target = max(cell_targets) if len(cell_targets) > 0 else 0.0
            targets.append(float(target))
        
        return torch.FloatTensor(targets)
    
    def build_graph(self,
                   df: pd.DataFrame,
                   feature_groups: Dict,
                   include_temporal: bool = True,
                   include_spatial: bool = True) -> Data:
        """
        Build grid-based spatial-temporal graph.
        
        Args:
            df: Dataframe with all features
            feature_groups: Dictionary mapping modality to feature columns
            include_temporal: Whether to include temporal edges
            include_spatial: Whether to include spatial edges
            
        Returns:
            PyTorch Geometric Data object
        """
        # Create grid
        self.create_grid(df)
        print(f"  Created grid with {len(self.grid_cells)} cells")
        
        # Assign events to grid
        df_with_grid = self.assign_events_to_grid(df)
        
        # Aggregate features per grid cell
        node_features = self.aggregate_features(df_with_grid, feature_groups)
        print(f"  Aggregated features: {node_features.shape}")
        
        # Build edges
        edge_list = []
        
        if include_spatial:
            spatial_edges = self.compute_spatial_edges()
            edge_list.append(spatial_edges)
            print(f"  Spatial edges: {spatial_edges.shape[1]}")
        
        if include_temporal:
            temporal_edges = self.compute_temporal_edges(df_with_grid)
            if temporal_edges.shape[1] > 0:
                edge_list.append(temporal_edges)
                print(f"  Temporal edges: {temporal_edges.shape[1]}")
        
        # Combine edges
        if len(edge_list) > 0:
            edge_index = np.concatenate(edge_list, axis=1)
            edge_index = np.unique(edge_index, axis=1)
            edge_index = torch.LongTensor(edge_index)
            edge_index = to_undirected(edge_index)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Create positions (grid cell centers)
        pos = torch.FloatTensor(self.grid_cells)
        
        # Create graph
        data = Data(
            x=torch.FloatTensor(node_features),
            edge_index=edge_index,
            pos=pos
        )
        
        # Store grid info
        data.grid_cells = self.grid_cells
        data.cell_to_index = self.cell_to_index
        
        return data
    
    def split_temporal(self,
                      df: pd.DataFrame,
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data temporally."""
        df_sorted = df.sort_values('time').reset_index(drop=True)
        n = len(df_sorted)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        return (df_sorted.iloc[:train_end],
                df_sorted.iloc[train_end:val_end],
                df_sorted.iloc[val_end:])

