"""
Step 4: Model Architecture
===========================
This module defines the Multi-Modal Graph Neural Network architecture
for spatio-temporal wildfire spread prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphSAGE, global_mean_pool, global_max_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import numpy as np

class ModalityEncoder(nn.Module):
    """Encoder for each data modality (Fire, Weather, Topographic)."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        """
        Initialize modality encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of layers
        """
        super(ModalityEncoder, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Dropout(0.2))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.2))
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        return self.encoder(x)


class MultiModalFusion(nn.Module):
    """Fusion layer for combining multi-modal features."""
    
    def __init__(self, fire_dim, weather_dim, topo_dim, output_dim, fusion_type='attention'):
        """
        Initialize fusion layer.
        
        Args:
            fire_dim: Fire modality embedding dimension
            weather_dim: Weather modality embedding dimension
            topo_dim: Topographic modality embedding dimension
            output_dim: Output dimension after fusion
            fusion_type: 'concat', 'attention', or 'weighted'
        """
        super(MultiModalFusion, self).__init__()
        
        self.fusion_type = fusion_type
        total_dim = fire_dim + weather_dim + topo_dim
        
        if fusion_type == 'concat':
            self.fusion = nn.Linear(total_dim, output_dim)
        elif fusion_type == 'attention':
            # Attention-based fusion
            self.attention = nn.MultiheadAttention(
                embed_dim=output_dim, num_heads=4, batch_first=True
            )
            self.fire_proj = nn.Linear(fire_dim, output_dim)
            self.weather_proj = nn.Linear(weather_dim, output_dim)
            self.topo_proj = nn.Linear(topo_dim, output_dim)
            self.fusion = nn.Linear(output_dim, output_dim)
        elif fusion_type == 'weighted':
            self.fire_weight = nn.Parameter(torch.ones(1))
            self.weather_weight = nn.Parameter(torch.ones(1))
            self.topo_weight = nn.Parameter(torch.ones(1))
            self.fusion = nn.Linear(total_dim, output_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(self, fire_emb, weather_emb, topo_emb):
        """Forward pass."""
        if self.fusion_type == 'concat':
            combined = torch.cat([fire_emb, weather_emb, topo_emb], dim=-1)
            return self.fusion(combined)
        elif self.fusion_type == 'attention':
            fire_proj = self.fire_proj(fire_emb).unsqueeze(1)
            weather_proj = self.weather_proj(weather_emb).unsqueeze(1)
            topo_proj = self.topo_proj(topo_emb).unsqueeze(1)
            
            # Stack modalities
            modalities = torch.cat([fire_proj, weather_proj, topo_proj], dim=1)
            
            # Self-attention
            attn_out, _ = self.attention(modalities, modalities, modalities)
            
            # Pool and fuse
            pooled = attn_out.mean(dim=1)
            return self.fusion(pooled)
        else:  # weighted
            combined = torch.cat([
                self.fire_weight * fire_emb,
                self.weather_weight * weather_emb,
                self.topo_weight * topo_emb
            ], dim=-1)
            return self.fusion(combined)


class GraphTemporalBlock(nn.Module):
    """Graph convolution block with temporal processing."""
    
    def __init__(self, in_dim, out_dim, conv_type='GCN', num_heads=4):
        """
        Initialize graph temporal block.
        
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            conv_type: 'GCN', 'GAT', or 'GraphSAGE'
            num_heads: Number of attention heads for GAT
        """
        super(GraphTemporalBlock, self).__init__()
        
        self.conv_type = conv_type
        
        if conv_type == 'GCN':
            self.conv = GCNConv(in_dim, out_dim)
        elif conv_type == 'GAT':
            self.conv = GATConv(in_dim, out_dim, heads=num_heads, concat=False)
        elif conv_type == 'GraphSAGE':
            self.conv = GraphSAGE(in_dim, out_dim, num_layers=2)
        else:
            raise ValueError(f"Unknown conv type: {conv_type}")
        
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, edge_index, edge_attr=None):
        """Forward pass."""
        x = self.conv(x, edge_index, edge_attr)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class TemporalGNN(nn.Module):
    """Temporal Graph Neural Network for sequence processing."""
    
    def __init__(self, input_dim, hidden_dim, num_layers=2, conv_type='GCN'):
        """
        Initialize temporal GNN.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of graph convolution layers
            conv_type: Type of graph convolution
        """
        super(TemporalGNN, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(GraphTemporalBlock(input_dim, hidden_dim, conv_type))
        
        # Additional layers
        for _ in range(num_layers - 1):
            self.layers.append(GraphTemporalBlock(hidden_dim, hidden_dim, conv_type))
    
    def forward(self, x, edge_index, edge_attr=None):
        """Forward pass through graph layers."""
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x


class WildfirePredictionModel(nn.Module):
    """Complete Multi-Modal GNN model for wildfire spread prediction."""
    
    def __init__(self, 
                 fire_feature_dim=10,
                 weather_feature_dim=8,
                 topo_feature_dim=9,
                 modality_embed_dim=64,
                 graph_hidden_dim=128,
                 num_graph_layers=3,
                 sequence_length=7,
                 prediction_horizon=1,
                 conv_type='GCN',
                 fusion_type='attention'):
        """
        Initialize the complete model.
        
        Args:
            fire_feature_dim: Dimension of fire features
            weather_feature_dim: Dimension of weather features
            topo_feature_dim: Dimension of topographic features
            modality_embed_dim: Embedding dimension for each modality
            graph_hidden_dim: Hidden dimension for graph layers
            num_graph_layers: Number of graph convolution layers
            sequence_length: Length of input sequence
            prediction_horizon: Number of time steps to predict ahead
            conv_type: Type of graph convolution ('GCN', 'GAT', 'GraphSAGE')
            fusion_type: Type of modality fusion ('concat', 'attention', 'weighted')
        """
        super(WildfirePredictionModel, self).__init__()
        
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Modality encoders
        self.fire_encoder = ModalityEncoder(
            fire_feature_dim, modality_embed_dim, modality_embed_dim
        )
        self.weather_encoder = ModalityEncoder(
            weather_feature_dim, modality_embed_dim, modality_embed_dim
        )
        self.topo_encoder = ModalityEncoder(
            topo_feature_dim, modality_embed_dim, modality_embed_dim
        )
        
        # Multi-modal fusion
        self.fusion = MultiModalFusion(
            modality_embed_dim, modality_embed_dim, modality_embed_dim,
            graph_hidden_dim, fusion_type=fusion_type
        )
        
        # Temporal GNN layers
        self.temporal_gnn = TemporalGNN(
            graph_hidden_dim, graph_hidden_dim, num_graph_layers, conv_type
        )
        
        # Temporal sequence processing (LSTM)
        self.lstm = nn.LSTM(
            graph_hidden_dim, graph_hidden_dim,
            num_layers=2, batch_first=True, dropout=0.2
        )
        
        # Prediction heads
        self.fire_occurrence_head = nn.Sequential(
            nn.Linear(graph_hidden_dim, graph_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(graph_hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.fire_intensity_head = nn.Sequential(
            nn.Linear(graph_hidden_dim, graph_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(graph_hidden_dim // 2, 1),
            nn.ReLU()  # FRP is non-negative
        )
        
        self.spread_direction_head = nn.Sequential(
            nn.Linear(graph_hidden_dim, graph_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(graph_hidden_dim // 2, 2)  # 2D direction vector
        )
    
    def forward(self, graph_sequence, edge_index, edge_attr=None):
        """
        Forward pass through the model.
        
        Args:
            graph_sequence: List of graph data objects or batched sequence
            edge_index: Edge indices
            edge_attr: Edge attributes
        
        Returns:
            Dictionary with predictions
        """
        batch_size = len(graph_sequence) if isinstance(graph_sequence, list) else 1
        
        # Process each graph in the sequence
        graph_embeddings = []
        
        for graph in graph_sequence:
            # Extract features (assuming features are already separated by modality)
            # In practice, you would separate features here
            x = graph.x if hasattr(graph, 'x') else graph
            
            # For simplicity, assume features are concatenated
            # In real implementation, split features by modality
            feature_dim = x.shape[1]
            fire_dim = feature_dim // 3
            weather_dim = feature_dim // 3
            topo_dim = feature_dim - fire_dim - weather_dim
            
            fire_features = x[:, :fire_dim]
            weather_features = x[:, fire_dim:fire_dim+weather_dim]
            topo_features = x[:, fire_dim+weather_dim:]
            
            # Encode each modality
            fire_emb = self.fire_encoder(fire_features)
            weather_emb = self.weather_encoder(weather_features)
            topo_emb = self.topo_encoder(topo_features)
            
            # Fuse modalities
            fused = self.fusion(fire_emb, weather_emb, topo_emb)
            
            # Graph convolution
            graph_emb = self.temporal_gnn(fused, edge_index, edge_attr)
            
            # Global pooling (if needed for sequence-level representation)
            # For node-level predictions, use graph_emb directly
            graph_embeddings.append(graph_emb)
        
        # Stack sequence
        sequence_tensor = torch.stack(graph_embeddings, dim=1)  # [nodes, seq_len, hidden_dim]
        
        # Temporal processing with LSTM
        lstm_out, _ = self.lstm(sequence_tensor)
        
        # Use last time step for prediction
        final_emb = lstm_out[:, -1, :]  # [nodes, hidden_dim]
        
        # Predictions
        predictions = {
            'fire_occurrence': self.fire_occurrence_head(final_emb),
            'fire_intensity': self.fire_intensity_head(final_emb),
            'spread_direction': self.spread_direction_head(final_emb)
        }
        
        return predictions
    
    def predict_single_step(self, graph_sequence, edge_index, edge_attr=None):
        """Predict for a single time step."""
        return self.forward(graph_sequence, edge_index, edge_attr)
    
    def predict_multi_step(self, graph_sequence, edge_index, edge_attr=None, steps=5):
        """Predict multiple time steps ahead using autoregressive approach."""
        predictions_sequence = []
        current_sequence = graph_sequence
        
        for step in range(steps):
            pred = self.forward(current_sequence, edge_index, edge_attr)
            predictions_sequence.append(pred)
            
            # Update sequence with predictions (for next iteration)
            # This is a simplified version - in practice, you'd update graph features
            # based on predictions
        
        return predictions_sequence


def create_model(config=None):
    """
    Create model from configuration.
    
    Args:
        config: Dictionary with model configuration
    
    Returns:
        Initialized model
    """
    if config is None:
        config = {
            'fire_feature_dim': 10,
            'weather_feature_dim': 8,
            'topo_feature_dim': 9,
            'modality_embed_dim': 64,
            'graph_hidden_dim': 128,
            'num_graph_layers': 3,
            'sequence_length': 7,
            'prediction_horizon': 1,
            'conv_type': 'GCN',
            'fusion_type': 'attention'
        }
    
    model = WildfirePredictionModel(**config)
    return model


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """Main function to test model architecture."""
    print("="*60)
    print("STEP 4: MODEL ARCHITECTURE")
    print("="*60)
    
    # Create model
    config = {
        'fire_feature_dim': 10,
        'weather_feature_dim': 8,
        'topo_feature_dim': 9,
        'modality_embed_dim': 64,
        'graph_hidden_dim': 128,
        'num_graph_layers': 3,
        'sequence_length': 7,
        'prediction_horizon': 1,
        'conv_type': 'GCN',
        'fusion_type': 'attention'
    }
    
    model = create_model(config)
    
    print("\nModel Architecture:")
    print(model)
    
    print(f"\nTotal Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    num_nodes = 100
    seq_length = 7
    feature_dim = 27  # fire + weather + topo
    
    # Create dummy graph sequence
    graph_sequence = []
    for _ in range(seq_length):
        x = torch.randn(num_nodes, feature_dim)
        graph_sequence.append(x)
    
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 8))
    
    # Forward pass
    with torch.no_grad():
        predictions = model(graph_sequence, edge_index)
    
    print(f"   Input sequence length: {seq_length}")
    print(f"   Number of nodes: {num_nodes}")
    print(f"   Output shapes:")
    for key, value in predictions.items():
        print(f"      {key}: {value.shape}")
    
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE COMPLETE!")
    print("="*60)
    
    return model


if __name__ == "__main__":
    model = main()

