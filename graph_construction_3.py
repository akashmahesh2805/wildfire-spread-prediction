"""
Step 3: Graph Construction
===========================
This module constructs spatial graphs from the preprocessed data,
creating nodes (spatial locations) and edges (spatial relationships).
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Batch
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy.spatial.distance import cdist
import pickle
import os

# Set device globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

class GraphBuilder:
    """Class to build spatial graphs from wildfire data."""
    
    def __init__(self, processed_df=None):
        """
        Initialize the GraphBuilder.
        
        Args:
            processed_df: Preprocessed dataframe with features
        """
        self.processed_df = processed_df
        self.graph_data = None
        self.node_features = None
        self.edge_index = None
        self.edge_attr = None
        self.node_coords = None
        
    def load_processed_data(self, file_path='processed_data.pkl'):
        """Load processed data from file."""
        print(f"Loading processed data from {file_path}...")
        with open(file_path, 'rb') as f:
            self.processed_df = pickle.load(f)
        print(f"   Loaded {len(self.processed_df)} records")
    
    def extract_node_coordinates(self, df=None):
        """
        Extract unique spatial coordinates as graph nodes.
        
        Args:
            df: Dataframe (uses self.processed_df if None)
        
        Returns:
            Array of node coordinates
        """
        if df is None:
            df = self.processed_df
        
        print("\nExtracting node coordinates...")
        
        # Get unique locations
        unique_locations = df[['latitude', 'longitude']].drop_duplicates().values
        
        self.node_coords = unique_locations
        print(f"   Found {len(unique_locations)} unique nodes")
        
        return unique_locations
    
    def create_spatial_edges_knn(self, k=8, distance_threshold=None):
        """
        Create edges using k-nearest neighbors.
        
        Args:
            k: Number of nearest neighbors
            distance_threshold: Maximum distance for edges (optional)
        
        Returns:
            Edge index and edge attributes
        """
        print(f"\nCreating spatial edges using k-NN (k={k})...")
        
        if self.node_coords is None:
            self.extract_node_coordinates()
        
        # Use k-NN to find neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='haversine').fit(
            np.radians(self.node_coords)
        )
        distances, indices = nbrs.kneighbors(np.radians(self.node_coords))
        
        # Remove self-connections (first neighbor is itself)
        indices = indices[:, 1:]
        distances = distances[:, 1:] * 6371  # Convert to km (Earth radius)
        
        # Create edge list
        edges = []
        edge_distances = []
        
        for i in range(len(self.node_coords)):
            for j, neighbor_idx in enumerate(indices[i]):
                dist = distances[i, j]
                
                # Apply distance threshold if specified
                if distance_threshold is None or dist <= distance_threshold:
                    edges.append([i, neighbor_idx])
                    edge_distances.append(dist)
        
        # Convert to edge index format (2 x num_edges)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_distances, dtype=torch.float).unsqueeze(1)
        
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        
        print(f"   Created {edge_index.shape[1]} edges")
        print(f"   Average degree: {edge_index.shape[1] / len(self.node_coords):.2f}")
        
        return edge_index, edge_attr
    
    def create_spatial_edges_distance(self, distance_threshold_km=10.0):
        """
        Create edges based on distance threshold.
        
        Args:
            distance_threshold_km: Maximum distance for edges in kilometers
        
        Returns:
            Edge index and edge attributes
        """
        print(f"\nCreating spatial edges using distance threshold ({distance_threshold_km} km)...")
        
        if self.node_coords is None:
            self.extract_node_coordinates()
        
        # Calculate pairwise distances using Haversine formula
        from math import radians, sin, cos, sqrt, atan2
        
        def haversine_distance(coord1, coord2):
            """Calculate Haversine distance between two coordinates."""
            lat1, lon1 = radians(coord1[0]), radians(coord1[1])
            lat2, lon2 = radians(coord2[0]), radians(coord2[1])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            
            return 6371 * c  # Earth radius in km
        
        # Create edge list
        edges = []
        edge_distances = []
        
        for i in range(len(self.node_coords)):
            for j in range(i+1, len(self.node_coords)):
                dist = haversine_distance(self.node_coords[i], self.node_coords[j])
                
                if dist <= distance_threshold_km:
                    # Add bidirectional edges
                    edges.append([i, j])
                    edges.append([j, i])
                    edge_distances.append(dist)
                    edge_distances.append(dist)
        
        # Convert to edge index format
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_distances, dtype=torch.float).unsqueeze(1)
        
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        
        print(f"   Created {edge_index.shape[1]} edges")
        print(f"   Average degree: {edge_index.shape[1] / len(self.node_coords):.2f}")
        
        return edge_index, edge_attr
    
    def create_topographic_edges(self, elevation_threshold=100.0, slope_threshold=15.0):
        """
        Create edges based on topographic connectivity.
        Only connect nodes with similar elevation and slope.
        
        Args:
            elevation_threshold: Maximum elevation difference (meters)
            slope_threshold: Maximum slope difference (degrees)
        
        Returns:
            Edge index and edge attributes
        """
        print(f"\nCreating topographic edges...")
        
        if self.node_coords is None:
            self.extract_node_coordinates()
        
        # Get topographic features for each node
        if self.processed_df is None:
            raise ValueError("Processed dataframe required for topographic edges")
        
        # Get unique locations with their features
        node_features_df = self.processed_df[['latitude', 'longitude', 'elevation', 'slope']].drop_duplicates()
        
        # Match coordinates to get features
        node_elevations = []
        node_slopes = []
        
        for coord in self.node_coords:
            lat, lon = coord
            matches = node_features_df[
                (np.abs(node_features_df['latitude'] - lat) < 0.001) &
                (np.abs(node_features_df['longitude'] - lon) < 0.001)
            ]
            if len(matches) > 0:
                node_elevations.append(matches.iloc[0]['elevation'])
                node_slopes.append(matches.iloc[0]['slope'])
            else:
                node_elevations.append(0)
                node_slopes.append(0)
        
        node_elevations = np.array(node_elevations)
        node_slopes = np.array(node_slopes)
        
        # Create edge list based on topographic similarity
        edges = []
        edge_attrs = []
        
        for i in range(len(self.node_coords)):
            for j in range(i+1, len(self.node_coords)):
                elev_diff = abs(node_elevations[i] - node_elevations[j])
                slope_diff = abs(node_slopes[i] - node_slopes[j])
                
                if elev_diff <= elevation_threshold and slope_diff <= slope_threshold:
                    # Calculate distance
                    from math import radians, sin, cos, sqrt, atan2
                    lat1, lon1 = radians(self.node_coords[i][0]), radians(self.node_coords[i][1])
                    lat2, lon2 = radians(self.node_coords[j][0]), radians(self.node_coords[j][1])
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                    c = 2 * atan2(sqrt(a), sqrt(1-a))
                    dist = 6371 * c
                    
                    edges.append([i, j])
                    edges.append([j, i])
                    edge_attrs.append([dist, elev_diff, slope_diff])
                    edge_attrs.append([dist, elev_diff, slope_diff])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        print(f"   Created {edge_index.shape[1]} topographic edges")
        
        return edge_index, edge_attr
    
    def extract_node_features(self, time_window=None, feature_cols=None):
        """
        Extract node features for a specific time window.
        
        Args:
            time_window: Specific time window to extract features for
            feature_cols: List of feature columns to include
        
        Returns:
            Node feature matrix
        """
        print(f"\nExtracting node features (optimized)...")
    
        if self.processed_df is None:
            raise ValueError("Processed dataframe required")
        
        df = self.processed_df.copy()
        
        # Filter by time window if specified
        if time_window is not None:
            df = df[df['time_window'] == time_window]
        
        # Default feature columns (exclude spatial and temporal)
        if feature_cols is None:
            exclude_cols = ['latitude', 'longitude', 'time_window', 'acq_date_parsed', 
                            'date_parsed', 'datetime', 'acq_datetime']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Group by (lat, lon) and take mean
        grouped = df.groupby(['latitude', 'longitude'])[feature_cols].mean().reset_index()
        
        # Make sure node_coords is set
        if self.node_coords is None:
            self.extract_node_coordinates()
        
        # Merge features with node coordinates
        merged = pd.DataFrame(self.node_coords, columns=['latitude', 'longitude'])
        merged = merged.merge(grouped, on=['latitude', 'longitude'], how='left')
        
        # Fill missing features with zeros
        merged.fillna(0, inplace=True)
        
        node_features = merged[feature_cols].values
        self.node_features = torch.tensor(node_features, dtype=torch.float)
        
        print(f"   Extracted {node_features.shape[1]} features for {node_features.shape[0]} nodes")
        
        return self.node_features
    
    def build_graph(self, method='knn', k=8, distance_threshold=None, 
                   time_window=None, feature_cols=None):
        """
        Build a complete graph data structure.
        
        Args:
            method: 'knn' or 'distance'
            k: Number of neighbors for k-NN
            distance_threshold: Distance threshold for distance method
            time_window: Time window for node features
            feature_cols: Feature columns to include
        
        Returns:
            PyTorch Geometric Data object
        """
        print("\n" + "="*60)
        print("BUILDING GRAPH")
        print("="*60)
        
        # Extract node coordinates
        self.extract_node_coordinates()
        
        # Create edges
        if method == 'knn':
            self.create_spatial_edges_knn(k=k, distance_threshold=distance_threshold)
        elif method == 'distance':
            self.create_spatial_edges_distance(distance_threshold_km=distance_threshold or 10.0)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Extract node features
        self.extract_node_features(time_window=time_window, feature_cols=feature_cols)
        
        # Create PyTorch Geometric Data object
        self.graph_data = Data(
            x=self.node_features,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            pos=torch.tensor(self.node_coords, dtype=torch.float)
        ).to(device)
        
        print(f"\nGraph Summary:")
        print(f"   Nodes: {self.graph_data.num_nodes}")
        print(f"   Edges: {self.graph_data.num_edges}")
        print(f"   Node Features: {self.graph_data.num_node_features}")
        print(f"   Edge Features: {self.graph_data.num_edge_features}")
        
        return self.graph_data
    
    def build_temporal_graphs(self, sequence_length=7, method='knn', k=8):
        """
        Build temporal sequence of graphs.
        
        Args:
            sequence_length: Number of time steps
            method: Graph construction method
            k: Number of neighbors for k-NN
        
        Returns:
            List of graph data objects
        """
        print(f"\nBuilding temporal graph sequence (length={sequence_length})...")
        
        if self.processed_df is None:
            raise ValueError("Processed dataframe required")
        
        # Get unique time windows
        time_windows = sorted(self.processed_df['time_window'].unique())[:sequence_length]
        
        graphs = []
        
        for time_window in time_windows:
            print(f"   Building graph for time window: {time_window}")
            graph = self.build_graph(method=method, k=k, time_window=time_window)
            graphs.append(graph)
        
        print(f"   Built {len(graphs)} graphs")
        
        return graphs
    
    def save_graph(self, output_file='graph_data.pkl'):
        """Save graph data to file."""
        print(f"\nSaving graph to {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(self.graph_data, f)
        print("   Graph saved successfully")
    
    def save_temporal_graphs(self, graphs, output_file='temporal_graphs.pkl'):
        """Save temporal graph sequence to file."""
        print(f"\nSaving temporal graphs to {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(graphs, f)
        print(f"   Saved {len(graphs)} graphs")


def main():
    """Main function to run graph construction."""
    print("="*60)
    print("STEP 3: GRAPH CONSTRUCTION")
    print("="*60)
    
    # Load processed data
    builder = GraphBuilder()
    builder.load_processed_data('processed_data.pkl')
    
    # Build graph using k-NN method
    graph = builder.build_graph(method='knn', k=8)

    # Move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)
    
    # Save graph
    builder.save_graph()
    
    # Optionally build temporal graphs
    print("\n" + "="*60)
    print("BUILDING TEMPORAL GRAPH SEQUENCE")
    print("="*60)
    temporal_graphs = builder.build_temporal_graphs(sequence_length=7)

    # Move all graphs to GPU
    temporal_graphs = [g.to(device) for g in temporal_graphs]

    builder.save_temporal_graphs(temporal_graphs)
    
    print("\n" + "="*60)
    print("GRAPH CONSTRUCTION COMPLETE!")
    print("="*60)
    
    return builder, graph


if __name__ == "__main__":
    builder, graph = main()

