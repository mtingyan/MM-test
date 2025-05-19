import os
import torch
import argparse
import pickle
import random
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
import warnings
from torch_geometric.data import Data
from tgb.linkproppred.dataset import LinkPropPredDataset

from utils import  create_ba_random_graph, create_sbm_evolving_graph

### set max_edges 500_000
MAX_EDGES = 500_000

class DataSplitter:
    def __init__(self, args, data):
        self.data = data
        self.dataset = args.dataset
        self.end_time = data.end_time
        self.start_time = data.start_time
        self.save_path = args.save_path

        ### For link prediction task
        self.test_negative_sampling_ratio = getattr(args, "test_negative_sampling_ratio", None)

        ### For synthetic dataset
        self.initial_nodes = getattr(args, "initial_nodes", None)
        self.link_probability = getattr(args, "link_probability", None)
        self.nodes_per_step = getattr(args, "nodes_per_step", None)
        self.time_step = getattr(args, "time_step", None)
        
    def _split_edges(self):
        if self.dataset in ["tgbl-wiki", "tgbl-review", "tgbl-coin", "tgbl-flight", "tgbl-comment", "ogbl-collab"]:
            edge_time = self.data.edge_time.squeeze()
            test_edge_mask = (edge_time <= self.end_time) & (edge_time >= self.start_time)
            train_edge_mask = edge_time < self.start_time
            test_edges = self.data.edge_index[:, test_edge_mask]
        return train_edge_mask, test_edges

    def _split_nodes(self):
        if self.dataset in ["ogbn-arxiv", 'SBM', "BA-random", 'penn', 'Amherst41', 'Cornell5', 'Johns_Hopkins55', 'Reed98']:
            node_time = self.data.node_time.squeeze()
            test_node_mask = (node_time <= self.end_time) & (node_time >= self.start_time)
            edge_time = self.data.edge_time.squeeze()
            train_edge_mask = edge_time < self.start_time
            train_node_mask = node_time < self.start_time
        return train_node_mask, test_node_mask, train_edge_mask

    def _negative_sampling(self, negative_sampling_ratio, edge_index, test_edges):
        num_nodes = self.data.num_nodes
        num_samples = int(num_nodes * negative_sampling_ratio)

        time_mask = self.data.edge_time.squeeze() < self.start_time
        historical_edges = edge_index[:, time_mask]
        historical_edges_dict = {i: set() for i in range(num_nodes)}

        src, tgt = historical_edges
        for s, t in zip(src.tolist(), tgt.tolist()):
            historical_edges_dict[s].add(t)
        existing_edges = set(map(tuple, edge_index.t().tolist()))

        neg_edges = []

        for source, _ in test_edges.t():
            source = source.item()
            sampled_neg_edges = []

            historical_sampled = []
            if source in historical_edges_dict:
                historical_targets = list(historical_edges_dict[source])
                historical_sampled = random.sample(historical_targets, min(len(historical_targets), num_samples // 2))
                sampled_neg_edges.extend([[source, tgt] for tgt in historical_sampled])

            while len(sampled_neg_edges) < num_samples:
                target = random.randint(0, num_nodes - 1)
                if (target != source and target not in historical_edges_dict[source] and (source, target) not in existing_edges):
                    sampled_neg_edges.append([source, target])
            neg_edges.extend(sampled_neg_edges[:num_samples])

        return torch.tensor(neg_edges, dtype=torch.long).t()

    def _generate_save_path(self):
        if self.dataset in ["tgbl-wiki", "tgbl-review", "tgbl-coin", "tgbl-flight", "tgbl-comment", "ogbl-collab"]:
            file_name = f"start_time_{self.start_time}_end_time_{self.end_time}_neg_{self.test_negative_sampling_ratio}.pkl"
        elif self.dataset in ["ogbn-arxiv", 'penn', 'Amherst41', 'Cornell5', 'Johns_Hopkins55', 'Reed98']:
            file_name = f"start_time_{self.start_time}_end_time_{self.end_time}.pkl"
        elif self.dataset in ["BA-random", "SBM"]:
            file_name = f"start_time_{self.start_time}_end_time_{self.end_time}_node_{self.initial_nodes}_p_{self.link_probability}_newnode{self.nodes_per_step}_timestep_{self.time_step}.pkl"
        return os.path.join(self.save_path, file_name)

    def load_or_create_splits(self):
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message="You are using `torch.load` with `weights_only=False`",
        )
        if os.path.exists(self._generate_save_path()):
            with open(self._generate_save_path(), "rb") as f:
                splits = pickle.load(f)
            return splits
        if self.dataset in ["tgbl-wiki", "tgbl-review", "tgbl-coin", "tgbl-flight", "tgbl-comment", "ogbl-collab"]:
            train_mask, test_edges = self._split_edges()
            test_neg_edges = self._negative_sampling(self.test_negative_sampling_ratio, self.data.edge_index, test_edges)

            splits = {
                "train_mask": train_mask,
                "test_edges": test_edges,
                "test_neg_edges": test_neg_edges,
            }

        elif self.dataset in ["ogbn-arxiv", "BA-random", "SBM", 'penn', 'Amherst41', 'Cornell5', 'Johns_Hopkins55', 'Reed98']:
            train_mask, test_mask, train_edge_mask = self._split_nodes()
            splits = {
                "train_mask": train_mask,
                "test_mask": test_mask,
                "train_edge_mask": train_edge_mask,
            }
        os.makedirs(self.save_path, exist_ok=True)
        with open(self._generate_save_path(), "wb") as f:
            pickle.dump(splits, f)

        return splits


class EvolvingDataset:
    def __init__(self, args):
        self.dataset = args.dataset
        self.data_path = args.data_path
        self.feature_dim = args.input_dim

        ### load data
        if self.dataset == "BA-random":
            self.build_random_graph(args)
        elif self.dataset == 'SBM':
            self.bulid_sbm_graph(args)
        else:
            self.load_data()

    def build_random_graph(self, args):
        raw_data_path = os.path.join(
            args.save_path,
            "raw_data",
            f"{args.initial_nodes}_{args.link_probability}_{args.nodes_per_step}_{args.time_step}.pkl"
        )
        if os.path.exists(raw_data_path):
            print("Pickle file found. Loading existing graph data...")
            with open(raw_data_path, "rb") as f:
                data = pickle.load(f)
        else:
            raw_data_dir = os.path.dirname(raw_data_path)
            os.makedirs(raw_data_dir, exist_ok=True)
            data = create_ba_random_graph(args, raw_data_path)
        self.node_feature = data['node_feature']
        self.node_time = data['node_time']
        self.edges = data['edges']
        self.edge_time = data['edge_time']

        print("Graph data successfully loaded or created.")

    def bulid_sbm_graph(self, args):
        raw_data_path = os.path.join(
            args.save_path,
            "raw_data",
            f"{args.initial_nodes}_{args.link_probability}_{args.nodes_per_step}_{args.time_step}.pkl"
        )
        if os.path.exists(raw_data_path):
            print("Pickle file found. Loading existing graph data...")
            with open(raw_data_path, "rb") as f:
                data = pickle.load(f)
        else:
            raw_data_dir = os.path.dirname(raw_data_path)
            os.makedirs(raw_data_dir, exist_ok=True)
            data = create_sbm_evolving_graph(args, raw_data_path)
        self.node_feature = data['node_feature']
        self.node_time = data['node_time']
        self.edges = data['edges']
        self.edge_time = data['edge_time']
        self.node_label = data['node_label']
        print("Graph data successfully loaded or created.")

    def load_data(self):
        warnings.filterwarnings("ignore", category=FutureWarning)
        if self.dataset in ["tgbl-wiki", "tgbl-review", "tgbl-coin", "tgbl-flight", "tgbl-comment"]:
            dataset = LinkPropPredDataset(name=self.dataset, root="datasets", preprocess=True)
            data = dataset.full_data
            ### load edges, edge_feature and edge_time
            edge_feature = dataset.edge_feat
            self.edges = [[source, target] for source, target in zip(data["sources"], data["destinations"])]
            self.edge_time = data["timestamps"]
            ### sort edge 
            sorted_edges = sorted(zip(self.edge_time, self.edges, edge_feature), key=lambda x: x[0])
            if len(sorted_edges) > MAX_EDGES:
                sorted_edges = sorted_edges[:MAX_EDGES]
            self.edge_time, self.edges, self.edge_feature = zip(*sorted_edges)
            ### normalize edge and edge_time
            min_time = min(self.edge_time)
            self.edge_time = [t - min_time for t in self.edge_time]

        elif self.dataset in ["ogbn-arxiv"]:
            from ogb.nodeproppred import PygNodePropPredDataset
            dataset = PygNodePropPredDataset(name=self.dataset)[0]
            ### load node feature, edges and node_time
            self.node_feature = dataset.x
            self.node_label = dataset.y
            edge_index = dataset.edge_index
            edges = edge_index.t().tolist()
            self.node_time = dataset.node_year.squeeze().tolist()
            ### normalize node_time
            min_time = min(self.node_time)
            self.node_time = [t - min_time for t in self.node_time]
            edge_time = [max(self.node_time[source], self.node_time[target]) for source, target in edges]
            ### sort edge and edge_times
            sorted_edges = sorted(zip(edge_time, edges), key=lambda x: x[0])
            if len(sorted_edges) > MAX_EDGES:
                sorted_edges = sorted_edges[:MAX_EDGES]
            self.edge_time, self.edges = zip(*sorted_edges)
        
        elif self.dataset in ['penn', 'Amherst41', 'Cornell5', 'Johns_Hopkins55', 'Reed98']:
            file_dir = './dataset/social/'
            ### load node feature, edges and node_time
            y = np.load(os.path.join(file_dir, self.dataset + '_label.npy'))
            x = np.load(os.path.join(file_dir, self.dataset + '_node_feat.npy'))
            edge_index = np.load(os.path.join(file_dir, self.dataset + '_edge_index.npy'))
            t = np.load(os.path.join(file_dir, self.dataset + '_year.npy')) / 1.0
            ### Remove illegal data
            node_mask = (y != -1)
            self.node_label = torch.tensor(y[node_mask], dtype=torch.long).unsqueeze(1)
            self.node_feature = torch.tensor(x[node_mask, :], dtype=torch.float)
            self.node_time = t[node_mask]
            mask = np.isin(edge_index[0], np.where(node_mask)[0]) & np.isin(edge_index[1], np.where(node_mask)[0])
            edge_index = edge_index[:, mask]
            unique_nodes = np.unique(edge_index)
            node_mapping = {node: i for i, node in enumerate(unique_nodes)}
            edges = np.array([[node_mapping[edge_index[0, i]], node_mapping[edge_index[1, i]]] 
                  for i in range(edge_index.shape[1])])
            ### normalize node_time
            min_time = min(self.node_time)
            self.node_time = [t - min_time for t in self.node_time]
            edge_time = [max(self.node_time[source], self.node_time[target]) for source, target in edges]
            ### sort edge and edge_times
            sorted_edges = sorted(zip(edge_time, edges), key=lambda x: x[0])
            if len(sorted_edges) > MAX_EDGES:
                sorted_edges = sorted_edges[:MAX_EDGES]
            self.edge_time, self.edges = zip(*sorted_edges)

        elif self.dataset in ["ogbl-collab"]:
            from ogb.linkproppred import PygLinkPropPredDataset
            dataset = PygLinkPropPredDataset(name="ogbl-collab")[0]
            ### load node feature, edges, edge_feature and edge_time
            self.node_feature = dataset.x
            edge_index = dataset.edge_index
            edges = edge_index.t().tolist()
            edge_time = dataset.edge_year.squeeze().tolist()
            edge_feature = dataset.edge_weight
            ### sort edge and edge_time
            sorted_edges = sorted(zip(edge_time, edges, edge_feature), key=lambda x: x[0])
            if len(sorted_edges) > MAX_EDGES:
                sorted_edges = sorted_edges[:MAX_EDGES]
            self.edge_time, self.edges, self.edge_feature = zip(*sorted_edges)
            ### normalize edge and edge_time
            min_time = min(self.edge_time)
            self.edge_time = [t - min_time for t in self.edge_time]
        else:
            raise ValueError("Dataset not found: " + self.dataset)

    def build_graph(self, start_time, end_time):
        if self.dataset in ["tgbl-wiki", "tgbl-review", "tgbl-coin", "tgbl-flight", "tgbl-comment"]:
            edges = np.array(self.edges)
            edge_time = np.array(self.edge_time)

            mask = edge_time <= end_time
            valid_edges = edges[mask]
            valid_time = edge_time[mask]
            edge_feature = torch.tensor(np.array(self.edge_feature)[mask], dtype=torch.float32)
            unique_nodes, node_mapping = np.unique(valid_edges, return_inverse=True)
            num_nodes = len(unique_nodes)
            mapped_edges = node_mapping.reshape(-1, 2)
            edge_index = torch.tensor(mapped_edges.T, dtype=torch.long)
            edge_time = torch.tensor(valid_time, dtype=torch.float32).unsqueeze(1)

            x = torch.randn(num_nodes, self.feature_dim)
            self.graph = Data(x=x, edge_index=edge_index, edge_time=edge_time, end_time=end_time, start_time=start_time, edge_feature=edge_feature)

        elif self.dataset in ["ogbn-arxiv", 'penn', 'Amherst41', 'Cornell5', 'Johns_Hopkins55', 'Reed98']:
            edges = np.array(self.edges)
            edge_time = np.array(self.edge_time)
            node_time = np.array(self.node_time)

            node_mask = node_time <= end_time
            valid_nodes = np.where(node_mask)[0]
            node_mapping = {node: i for i, node in enumerate(valid_nodes)}
            num_nodes = len(valid_nodes)

            mask = edge_time <= end_time
            valid_edges = edges[mask]
            valid_time = edge_time[mask]
            mapped_edges = np.array([[node_mapping[src], node_mapping[dst]] for src, dst in valid_edges])
            edge_index = torch.tensor(mapped_edges.T, dtype=torch.long)
            edge_time = torch.tensor(valid_time, dtype=torch.float32).unsqueeze(1)
            node_time = torch.tensor(node_time[valid_nodes], dtype=torch.float32).unsqueeze(1)
            x = self.node_feature[valid_nodes].clone().detach().float()

            node_label = self.node_label[valid_nodes].clone().detach().long()
            self.graph = Data(x=x, edge_index=edge_index, node_time=node_time, edge_time=edge_time, end_time=end_time, start_time=start_time, y=node_label)
        
        elif self.dataset in ["ogbl-collab"]:
            edges = np.array(self.edges)
            edge_time = np.array(self.edge_time)

            mask = edge_time <= end_time
            valid_edges = edges[mask]
            valid_time = edge_time[mask]
            edge_feature = torch.tensor(np.array(self.edge_feature)[mask], dtype=torch.float32)

            unique_nodes, node_mapping = np.unique(valid_edges, return_inverse=True)
            num_nodes = len(unique_nodes)
            mapped_edges = node_mapping.reshape(-1, 2)
            edge_index = torch.tensor(mapped_edges.T, dtype=torch.long)
            edge_time = torch.tensor(valid_time, dtype=torch.float32).unsqueeze(1)

            x = self.node_feature[unique_nodes].clone().detach().float()

            self.graph = Data(x=x, edge_index=edge_index, edge_time=edge_time, end_time=end_time, start_time=start_time, edge_feature=edge_feature)
        
        elif self.dataset in ["BA-random"]:
            edges = self.edges
            edge_time = self.edge_time
            node_time = self.node_time

            node_mask = node_time <= end_time
            valid_nodes = np.where(node_mask)[0]
            node_mapping = {node: i for i, node in enumerate(valid_nodes)}
            num_nodes = len(valid_nodes)
            mask = edge_time <= end_time
            valid_edges = edges[mask]
            valid_time = edge_time[mask]

            mapped_edges = np.array([[node_mapping[src], node_mapping[dst]] for src, dst in valid_edges])
            edge_index = torch.tensor(mapped_edges.T, dtype=torch.long)
            edge_time = torch.tensor(valid_time, dtype=torch.float32).unsqueeze(1)
            node_time = torch.tensor(node_time[valid_nodes], dtype=torch.float32).unsqueeze(1)
            x = torch.tensor(self.node_feature[valid_nodes], dtype=torch.float32)

            self.graph = Data(x=x, edge_index=edge_index, node_time=node_time, edge_time=edge_time, end_time=end_time, start_time=start_time)
        
        elif self.dataset in ["SBM"]:
            edges = self.edges
            edge_time = self.edge_time
            node_time = self.node_time

            node_mask = node_time <= end_time
            valid_nodes = np.where(node_mask)[0]
            node_mapping = {node: i for i, node in enumerate(valid_nodes)}
            num_nodes = len(valid_nodes)
            mask = edge_time <= end_time
            valid_edges = edges[mask]
            valid_time = edge_time[mask]

            mapped_edges = np.array([[node_mapping[src], node_mapping[dst]] for src, dst in valid_edges])
            edge_index = torch.tensor(mapped_edges.T, dtype=torch.long)
            edge_time = torch.tensor(valid_time, dtype=torch.float32).unsqueeze(1)
            node_time = torch.tensor(node_time[valid_nodes], dtype=torch.float32).unsqueeze(1)
            x = torch.tensor(self.node_feature[valid_nodes], dtype=torch.float32)
            y = torch.tensor(self.node_label[valid_nodes], dtype=torch.long).unsqueeze(1)

            self.graph = Data(x=x, edge_index=edge_index, node_time=node_time, edge_time=edge_time, end_time=end_time, start_time=start_time, y=y)
               
        return self.graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training a Simple Model with Cached Data Splitting")
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv", help="Dataset name")
    parser.add_argument("--data_path", type=str, default="../dataset/arxiv_HepTh", help="Data path")
    parser.add_argument( "--save_path", type=str, default="split_data/ogbn-arxiv", help="split path")
    parser.add_argument("--input_dim", type=int, default=16, help="Number of hidden units in the model")
    parser.add_argument("--test_negative_sampling_ratio", type=float, default=0.01, help="Ratio of negative samples to positive samples")

    args = parser.parse_args()
    dataset = EvolvingDataset(args)
    data = dataset.build_graph(40, 47)
    print(data)
    print(max(data.edge_time), min(data.edge_time))
    print(max(data.y), min(data.y))
    print(max(data.node_time), min(data.node_time))
    splitter = DataSplitter(args, data)
    splits = splitter.load_or_create_splits()
    print(f"Train nodes shape is {sum(splits['train_mask'].tolist())}.")
    print(f"Test nodes shape is {sum(splits['test_mask'].tolist())}.")
    print(f"Train edges shape is {sum(splits['train_edge_mask'].tolist())}.")
    # print(f"Test edges shape is {splits['test_edges'].shape}.")
    # print(f"Test negative edges shape is {splits['test_neg_edges'].shape}.")
