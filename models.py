import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GATConv


class GNNEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers=2,
        edge_feature_dim=None,
        backbone="GCNConv",
        activation="relu",
        dropout_rate=0.0,
    ):
        super(GNNEncoder, self).__init__()
        conv_layer = {
            "GCNConv": GCNConv,
            "GraphConv": GraphConv,
            "SAGEConv": SAGEConv,
            "GATConv": GATConv,
        }.get(backbone, GCNConv)

        self.activation = getattr(F, activation, F.relu)
        self.edge_feature_dim = edge_feature_dim
        self.dropout_rate = dropout_rate

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Input layer
        self.convs.append(conv_layer(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(conv_layer(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
    def forward(self, x, edge_index, edge_feature=None):
        edge_weight = None
        if edge_feature is not None:
            edge_weight = edge_feature if self.edge_feature_dim == 1 else edge_feature.mean(dim=1)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = self.batch_norms[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.convs[-1](x, edge_index, edge_weight=edge_weight)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(LinkPredictor, self).__init__()
        self.proj = torch.nn.Linear(hidden_dim, output_dim)

    def decode(self, z, edge_index):
        # 预测边的存在性分数：使用投影后的表示
        z = self.proj(z)
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        z = self.proj(z)
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def forward(self, h, pos_edge_index, neg_edge_index):
        pos_scores = self.decode(h, pos_edge_index)
        neg_scores = self.decode(h, neg_edge_index)
        return pos_scores, neg_scores


class NodeClassifier(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(NodeClassifier, self).__init__()
        self.classifier = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, h):
        return self.classifier(h)


class NodeRegressor(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(NodeRegressor, self).__init__()
        self.regressor = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, h):
        return self.regressor(h).squeeze(-1)


class CombinedModel(nn.Module):
    def __init__(self, gnn_encoder, classifier):
        super(CombinedModel, self).__init__()
        self.gnn_encoder = gnn_encoder
        self.classifier = classifier

    def encode(self, *args, **kwargs):
        return self.gnn_encoder(*args, **kwargs)

    def classify(self, encoded):
        return self.classifier(encoded)

    def forward(self, *args, **kwargs):
        encoded = self.encode(*args, **kwargs)
        return self.classify(encoded)

class PromptGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)

