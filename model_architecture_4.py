# Step 4: Model Architecture
# ===========================
# Multi-Modal Graph Neural Network for spatio-temporal wildfire spread prediction

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphSAGE
from torch_geometric.data import Data
from graph_construction_3 import GraphBuilder  # updated import

# -------------------- Modality Encoder --------------------
class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(ModalityEncoder, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(0.2)]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(0.2)]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.encoder = nn.Sequential(*layers)
    def forward(self, x):
        return self.encoder(x)

# -------------------- Multi-Modal Fusion --------------------
class MultiModalFusion(nn.Module):
    def __init__(self, fire_dim, weather_dim, topo_dim, output_dim, fusion_type='attention'):
        super(MultiModalFusion, self).__init__()
        self.fusion_type = fusion_type
        if fusion_type == 'concat':
            self.fusion = nn.Linear(fire_dim + weather_dim + topo_dim, output_dim)
        elif fusion_type == 'attention':
            self.attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=4, batch_first=True)
            self.fire_proj = nn.Linear(fire_dim, output_dim)
            self.weather_proj = nn.Linear(weather_dim, output_dim)
            self.topo_proj = nn.Linear(topo_dim, output_dim)
            self.fusion = nn.Linear(output_dim, output_dim)
        else:  # weighted
            self.fire_weight = nn.Parameter(torch.ones(1))
            self.weather_weight = nn.Parameter(torch.ones(1))
            self.topo_weight = nn.Parameter(torch.ones(1))
            self.fusion = nn.Linear(fire_dim + weather_dim + topo_dim, output_dim)
    def forward(self, fire_emb, weather_emb, topo_emb):
        if self.fusion_type == 'concat':
            return self.fusion(torch.cat([fire_emb, weather_emb, topo_emb], dim=-1))
        elif self.fusion_type == 'attention':
            mods = torch.cat([self.fire_proj(fire_emb).unsqueeze(1),
                              self.weather_proj(weather_emb).unsqueeze(1),
                              self.topo_proj(topo_emb).unsqueeze(1)], dim=1)
            pooled, _ = self.attention(mods, mods, mods)
            return self.fusion(pooled.mean(dim=1))
        else:  # weighted
            combined = torch.cat([self.fire_weight*fire_emb,
                                  self.weather_weight*weather_emb,
                                  self.topo_weight*topo_emb], dim=-1)
            return self.fusion(combined)

# -------------------- Graph Temporal Block --------------------
class GraphTemporalBlock(nn.Module):
    def __init__(self, in_dim, out_dim, conv_type='GAT', num_heads=4):
        super(GraphTemporalBlock, self).__init__()
        if conv_type == 'GCN':
            self.conv = GCNConv(in_dim, out_dim)
        elif conv_type == 'GAT':
            self.conv = GATConv(in_dim, out_dim, heads=num_heads, concat=False)
        else:
            self.conv = GraphSAGE(in_dim, out_dim, num_layers=2)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

# -------------------- Temporal GNN --------------------
class TemporalGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, conv_type='GAT'):
        super(TemporalGNN, self).__init__()
        self.layers = nn.ModuleList([GraphTemporalBlock(input_dim if i==0 else hidden_dim, hidden_dim, conv_type) 
                                     for i in range(num_layers)])
    def forward(self, x, edge_index, edge_attr=None):
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x

# -------------------- Wildfire Prediction Model --------------------
class WildfirePredictionModel(nn.Module):
    def __init__(self,
                 # Adjusted default feature dims to match processed data
                 # (fire: brightness/frp/type -> 3, weather: t2m/u10/v10 -> 3,
                 # topo: 7 interpolated topo columns)
                 fire_feature_dim=3,
                 weather_feature_dim=3,
                 topo_feature_dim=7,
                 modality_embed_dim=64,
                 graph_hidden_dim=128,
                 num_graph_layers=3,
                 sequence_length=7,
                 conv_type='GAT',
                 fusion_type='attention'):
        super(WildfirePredictionModel, self).__init__()
        self.fire_encoder = ModalityEncoder(fire_feature_dim, modality_embed_dim, modality_embed_dim)
        self.weather_encoder = ModalityEncoder(weather_feature_dim, modality_embed_dim, modality_embed_dim)
        self.topo_encoder = ModalityEncoder(topo_feature_dim, modality_embed_dim, modality_embed_dim)
        self.fusion = MultiModalFusion(modality_embed_dim, modality_embed_dim, modality_embed_dim,
                                       graph_hidden_dim, fusion_type)
        self.temporal_gnn = TemporalGNN(graph_hidden_dim, graph_hidden_dim, num_graph_layers, conv_type)
        self.lstm = nn.LSTM(graph_hidden_dim, graph_hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fire_occurrence_head = nn.Sequential(nn.Linear(graph_hidden_dim, graph_hidden_dim//2),
                                                  nn.ReLU(), nn.Dropout(0.3), nn.Linear(graph_hidden_dim//2,1),
                                                  nn.Sigmoid())
        self.fire_intensity_head = nn.Sequential(nn.Linear(graph_hidden_dim, graph_hidden_dim//2),
                                                 nn.ReLU(), nn.Dropout(0.3), nn.Linear(graph_hidden_dim//2,1),
                                                 nn.ReLU())
        self.spread_direction_head = nn.Sequential(nn.Linear(graph_hidden_dim, graph_hidden_dim//2),
                                                   nn.ReLU(), nn.Dropout(0.3), nn.Linear(graph_hidden_dim//2,2))
        self.fire_feature_dim = fire_feature_dim
        self.weather_feature_dim = weather_feature_dim
        self.topo_feature_dim = topo_feature_dim

    def forward(self, graph_sequence):
        graph_embeddings = []
        # Quick sanity check: ensure node feature dimensionality matches expected split
        expected_total = self.fire_feature_dim + self.weather_feature_dim + self.topo_feature_dim
        if len(graph_sequence) > 0:
            actual_dim = graph_sequence[0].x.shape[1]
            if actual_dim != expected_total:
                raise ValueError(f"Node feature dimension mismatch: got {actual_dim}, expected {expected_total} "
                                 f"(fire={self.fire_feature_dim}, weather={self.weather_feature_dim}, topo={self.topo_feature_dim})")
        for graph in graph_sequence:
            x = graph.x
            fire_features = x[:, :self.fire_feature_dim]
            weather_features = x[:, self.fire_feature_dim:self.fire_feature_dim+self.weather_feature_dim]
            topo_features = x[:, self.fire_feature_dim+self.weather_feature_dim:]
            fire_emb = self.fire_encoder(fire_features)
            weather_emb = self.weather_encoder(weather_features)
            topo_emb = self.topo_encoder(topo_features)
            fused = self.fusion(fire_emb, weather_emb, topo_emb)
            graph_emb = self.temporal_gnn(fused, graph.edge_index, graph.edge_attr)
            graph_embeddings.append(graph_emb)
        sequence_tensor = torch.stack(graph_embeddings, dim=1)
        lstm_out, _ = self.lstm(sequence_tensor)
        final_emb = lstm_out[:, -1, :]
        return {
            'fire_occurrence': self.fire_occurrence_head(final_emb),
            'fire_intensity': self.fire_intensity_head(final_emb),
            'spread_direction': self.spread_direction_head(final_emb)
        }

# -------------------- Create model --------------------
def create_model(config=None):
    if config is None:
        config = {'fire_feature_dim':3,'weather_feature_dim':3,'topo_feature_dim':7,
                  'modality_embed_dim':64,'graph_hidden_dim':128,'num_graph_layers':3,
                  'sequence_length':7,'conv_type':'GAT','fusion_type':'attention'}
    return WildfirePredictionModel(**config)

# -------------------- Count parameters --------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -------------------- Main --------------------
def main():
    print("="*60)
    print("STEP 4: MODEL ARCHITECTURE")
    print("="*60)
    
    # Load temporal graphs using GraphBuilder
    builder = GraphBuilder()
    builder.load_processed_data('processed_data.pkl')
    temporal_graphs = builder.build_temporal_graphs(sequence_length=7, method='knn', k=8)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    temporal_graphs = [g.to(device) for g in temporal_graphs]
    
    # Create model
    model = create_model().to(device)
    print("\nModel Architecture:\n", model)
    print(f"\nTotal Parameters: {count_parameters(model):,}")
    
    # Forward pass example
    with torch.no_grad():
        preds = model(temporal_graphs)
    
    print("\nPredictions shapes:")
    for k,v in preds.items():
        print(f"   {k}: {v.shape}")
    
    print("\nMODEL ARCHITECTURE COMPLETE!")
    print("="*60)
    return model

if __name__ == "__main__":
    model = main()
