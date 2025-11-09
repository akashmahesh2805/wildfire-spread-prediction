"""
Multi-Modal Graph Neural Network models for wildfire spread prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphSAGE, global_mean_pool, global_max_pool
from torch_geometric.nn import MessagePassing
from typing import Optional


class MultiModalGCN(nn.Module):
    """
    Multi-Modal Graph Convolutional Network for wildfire spread prediction.
    Uses separate encoders for different modalities and fuses them.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 output_dim: int = 1,
                 dropout: float = 0.2,
                 num_modalities: int = 3):
        """
        Initialize Multi-Modal GCN.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GCN layers
            output_dim: Output dimension (1 for binary classification/regression)
            dropout: Dropout rate
            num_modalities: Number of modalities (fire, weather, terrain)
        """
        super(MultiModalGCN, self).__init__()
        
        self.num_modalities = num_modalities
        self.hidden_dim = hidden_dim
        
        # Separate encoders for each modality (if features are separated)
        # For simplicity, we'll use a shared encoder here
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gcn_layers.append(
                GCNConv(hidden_dim if i == 0 else hidden_dim, hidden_dim)
            )
        
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass.
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge indices [2, E]
            batch: Batch vector for graph-level tasks (optional)
            
        Returns:
            Node embeddings or graph-level predictions
        """
        # Project input features
        x = self.input_proj(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Apply GCN layers
        for gcn in self.gcn_layers:
            x = gcn(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Output prediction
        out = self.output_layer(x)
        
        return out


class TemporalGCN(nn.Module):
    """
    Temporal Graph Convolutional Network with LSTM for time-series modeling.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_gcn_layers: int = 2,
                 lstm_hidden: int = 64,
                 num_lstm_layers: int = 2,
                 output_dim: int = 1,
                 dropout: float = 0.2):
        """
        Initialize Temporal GCN.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: GCN hidden dimension
            num_gcn_layers: Number of GCN layers
            lstm_hidden: LSTM hidden dimension
            num_lstm_layers: Number of LSTM layers
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super(TemporalGCN, self).__init__()
        
        # GCN layers for spatial encoding
        self.gcn_layers = nn.ModuleList()
        for i in range(num_gcn_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.gcn_layers.append(GCNConv(in_dim, hidden_dim))
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            hidden_dim,
            lstm_hidden,
            num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, output_dim)
        )
    
    def forward(self, x, edge_index, time_sequence=None):
        """
        Forward pass.
        
        Args:
            x: Node features [N, input_dim] or [T, N, input_dim] for sequences
            edge_index: Edge indices [2, E]
            time_sequence: Whether x is a sequence (T, N, F)
            
        Returns:
            Predictions
        """
        if time_sequence is None:
            time_sequence = len(x.shape) == 3
        
        if not time_sequence:
            # Single time step
            # Apply GCN
            for gcn in self.gcn_layers:
                x = gcn(x, edge_index)
                x = F.relu(x)
                x = self.dropout(x)
            
            # Output
            out = self.output_layer(x)
            return out
        else:
            # Time sequence: [T, N, F]
            T, N, F = x.shape
            outputs = []
            
            for t in range(T):
                x_t = x[t]  # [N, F]
                
                # Apply GCN
                for gcn in self.gcn_layers:
                    x_t = gcn(x_t, edge_index)
                    x_t = F.relu(x_t)
                    x_t = self.dropout(x_t)
                
                outputs.append(x_t)
            
            # Stack temporal outputs
            x_seq = torch.stack(outputs, dim=0)  # [T, N, hidden_dim]
            
            # Reshape for LSTM: [N, T, hidden_dim]
            x_seq = x_seq.transpose(0, 1)
            
            # Apply LSTM
            lstm_out, _ = self.lstm(x_seq)
            
            # Use last time step
            final_out = lstm_out[:, -1, :]  # [N, lstm_hidden]
            
            # Output
            out = self.output_layer(final_out)
            return out


class GraphAttentionWildfire(nn.Module):
    """
    Graph Attention Network (GAT) for wildfire spread prediction.
    Uses attention mechanism to weight neighbor contributions.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 output_dim: int = 1,
                 dropout: float = 0.2):
        """
        Initialize GAT model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super(GraphAttentionWildfire, self).__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # First layer
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True)
        )
        
        # Intermediate layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, 
                       dropout=dropout, concat=True)
            )
        
        # Last layer (single head)
        if num_layers > 1:
            self.gat_layers.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=1, 
                       dropout=dropout, concat=False)
            )
        
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge indices [2, E]
            
        Returns:
            Predictions [N, output_dim]
        """
        # Apply GAT layers
        for gat in self.gat_layers:
            x = gat(x, edge_index)
            x = F.elu(x)
            x = self.dropout(x)
        
        # Output
        out = self.output_layer(x)
        return out


class MultiModalFusionGNN(nn.Module):
    """
    Multi-Modal GNN with explicit modality fusion.
    Separates features by modality, processes them separately, then fuses.
    """
    
    def __init__(self,
                 fire_dim: int,
                 weather_dim: int,
                 terrain_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 output_dim: int = 1,
                 dropout: float = 0.2,
                 fusion_method: str = 'concat'):
        """
        Initialize Multi-Modal Fusion GNN.
        
        Args:
            fire_dim: Fire feature dimension
            weather_dim: Weather feature dimension
            terrain_dim: Terrain feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of GCN layers
            output_dim: Output dimension
            dropout: Dropout rate
            fusion_method: 'concat', 'add', or 'attention'
        """
        super(MultiModalFusionGNN, self).__init__()
        
        self.fusion_method = fusion_method
        
        # Modality-specific encoders
        self.fire_encoder = nn.Sequential(
            nn.Linear(fire_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.weather_encoder = nn.Sequential(
            nn.Linear(weather_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.terrain_encoder = nn.Sequential(
            nn.Linear(terrain_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion layer
        if fusion_method == 'concat':
            fusion_input_dim = hidden_dim * 3
        elif fusion_method == 'add':
            fusion_input_dim = hidden_dim
        elif fusion_method == 'attention':
            fusion_input_dim = hidden_dim
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # GCN layers after fusion
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = fusion_input_dim if i == 0 else hidden_dim
            self.gcn_layers.append(GCNConv(in_dim, hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x_fire, x_weather, x_terrain, edge_index):
        """
        Forward pass with separate modality inputs.
        
        Args:
            x_fire: Fire features [N, fire_dim]
            x_weather: Weather features [N, weather_dim]
            x_terrain: Terrain features [N, terrain_dim]
            edge_index: Edge indices [2, E]
            
        Returns:
            Predictions [N, output_dim]
        """
        # Encode each modality
        h_fire = self.fire_encoder(x_fire)
        h_weather = self.weather_encoder(x_weather)
        h_terrain = self.terrain_encoder(x_terrain)
        
        # Fuse modalities
        if self.fusion_method == 'concat':
            x = torch.cat([h_fire, h_weather, h_terrain], dim=1)
        elif self.fusion_method == 'add':
            x = h_fire + h_weather + h_terrain
        elif self.fusion_method == 'attention':
            # Stack modalities: [3, N, hidden_dim]
            modalities = torch.stack([h_fire, h_weather, h_terrain], dim=0)
            x, _ = self.attention(modalities, modalities, modalities)
            x = x.mean(dim=0)  # Average over modalities
        
        # Apply GCN layers
        for gcn in self.gcn_layers:
            x = gcn(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Output
        out = self.output_layer(x)
        return out

