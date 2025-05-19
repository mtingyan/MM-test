import torch
import numpy as np
from sklearn.cluster import KMeans
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GATConv

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, path="checkpoint.pth", verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        self.best_model_state = model.state_dict()
        torch.save(self.best_model_state, self.path)
        if self.verbose:
            print(f"Validation loss decreased. Saving model to {self.path}")

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.path))


class GNNEncoder(torch.nn.Module):
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

        self.convs = torch.nn.ModuleList()
        self.convs.append(conv_layer(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(conv_layer(hidden_dim, hidden_dim))
            
    def forward(self, x, edge_index, edge_feature=None):
        edge_weight = None
        if edge_feature is not None:
            edge_weight = edge_feature if self.edge_feature_dim == 1 else edge_feature.mean(dim=1)

        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight=edge_weight)
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
