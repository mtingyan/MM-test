import wandb
import argparse
import torch
import math
import os
import pandas as pd
import torch.nn.functional as F

from torch_geometric.utils import negative_sampling

from data_preprocess import EvolvingDataset, DataSplitter
from models import LinkPredictor, PromptGenerator, GNNEncoder
from utils import EarlyStopping, load_config, TemporalDataSplitter, get_link_labels, set_seed, add_time_features, add_degree_features, add_temporal_structure_features

TEST_K = 10
TEST_BATCH_SIZE = 1024


def compute_feature_prompt(h_prev, h_curr):
    # 计算softmax
    softmax_prev = F.softmax(h_prev, dim=-1)
    softmax_curr = F.softmax(h_curr, dim=-1)
    
    # 计算KL散度
    kl_div = F.kl_div(softmax_curr.log(), softmax_prev, reduction='none')

    return kl_div

def compute_degree_prompt(prev_degrees, curr_degrees, num_buckets=10):
    # 计算度的变化
    degree_changes = curr_degrees - prev_degrees
    
    # 将度的变化分桶
    min_change = degree_changes.min()
    max_change = degree_changes.max()
    bucket_size = (max_change - min_change) / num_buckets
    
    # 将每个节点的度变化映射到相应的桶
    bucketed_changes = torch.floor((degree_changes - min_change) / bucket_size).long()
    
    # 将桶索引转换为one-hot向量
    one_hot_changes = F.one_hot(bucketed_changes, num_classes=num_buckets)
    
    return one_hot_changes.float()

@torch.no_grad()
def evaluate(args, model, dataset, test_time):
    model.eval()
    score_list = {}
    for i, period in enumerate(test_time):
        test_data = dataset.build_graph(period[0], period[1])
        splitter = DataSplitter(args, test_data)
        splits = splitter.load_or_create_splits()
        edge_index = test_data.edge_index[:,splits["train_mask"]]
        x = test_data.x
        num_nodes = test_data.x.size(0)
        ### Input Augmentation 
        if args.model == 'TimeEncoding':
            x = add_time_features(x, edge_index, test_data.edge_time, num_nodes, args.time_dim)
        elif args.model == 'DegreeEncoding':
            x = add_degree_features(x, edge_index, num_nodes, args.degree_dim)
        elif args.model == 'DTEncoding':
            x = add_time_features(x, edge_index, test_data.edge_time, num_nodes, args.time_dim)
            x = add_degree_features(x, edge_index, num_nodes, args.degree_dim)
        elif args.model == 'TSS':
            x = add_temporal_structure_features(x, edge_index, test_data.edge_time, num_nodes, args.tss_dim)

        if args.dataset in [
            "tgbl-review",
            "tgbl-coin",
            "tgbl-flight",
            "tgbl-comment",
            "ogbl-collab",
        ]:
            edge_feature = test_data.edge_feature[splits["train_mask"]].to(args.device)
            z = model.encode(x.to(args.device), edge_index.to(args.device), edge_feature)
        elif args.dataset in ["tgbl-wiki",]:
            z = model.encode(x.to(args.device), edge_index.to(args.device))

        test_edge_index = splits["test_edges"].to(args.device)
        test_neg_edge_index = splits["test_neg_edges"].to(args.device)
        pos_scores = []
        neg_scores = []
        num_neg_samples = int(num_nodes * args.test_negative_sampling_ratio)
        
        for start in range(0, test_edge_index.size(1), TEST_BATCH_SIZE):
            end = start + TEST_BATCH_SIZE
            pos_scores.append(model.decode(z, test_edge_index[:, start:end]))

        for start in range(0, test_neg_edge_index.size(1), TEST_BATCH_SIZE * num_neg_samples):
            end = start + TEST_BATCH_SIZE * num_neg_samples
            neg_scores.append(model.decode(z, test_neg_edge_index[:, start:end]))

        positive_scores = torch.cat(pos_scores).view(-1, 1)
        negative_scores = torch.cat(neg_scores).view(-1, num_neg_samples)
        scores = torch.cat([positive_scores, negative_scores], dim=1)
        ranks = torch.argsort(torch.argsort(scores, dim=1, descending=True), dim=1)

        reciprocal_ranks = 1.0 / (ranks[:, 0] + 1).float()
        mrr = reciprocal_ranks.mean().item()
        score_list[f"test_period_{i}_mrr"] = mrr

        top_k_hits = (ranks[:, :TEST_K] == 0).float().sum(dim=1)  
        top_k_accuracy = top_k_hits.mean().item()
        score_list[f"test_period_{i}_hit@k"] = top_k_accuracy
        scores_at_k = scores[:, :TEST_K]
        dcg_at_k = torch.sum((2.0 ** scores_at_k - 1) / torch.log2(ranks[:, :TEST_K].float() + 2), dim=1)
        sorted_scores = torch.sort(scores, dim=1, descending=True)[0]  
        ideal_scores_at_k = sorted_scores[:, :TEST_K]  
        ideal_dcg_at_k = torch.sum((2.0 ** ideal_scores_at_k - 1) / torch.log2(torch.arange(1, TEST_K + 1, device=scores.device).float() + 1), dim=1)
        ndcg_at_k = (dcg_at_k / ideal_dcg_at_k).mean().item()
        score_list[f"test_period_{i}_ndcg@k"] = ndcg_at_k
        torch.cuda.empty_cache()

    return score_list

def predict_links(args):
    set_seed(args.seed)

    dataset = EvolvingDataset(args)
    train_time, test_time_list = TemporalDataSplitter(args, dataset).split_by_time()
    print(f"Training data ends in {train_time[0]-1}")
    print(f"Validation data starts from {train_time[0]}, ends in {train_time[1]}")
    print("Test data:")
    for i, test_time in enumerate(test_time_list):
        print(f"Test time {i + 1}: {test_time}")

    input_dim = args.input_dim + args.pt_dim

    model = LinkPredictor(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        edge_feature_dim=args.edge_dim,
        backbone=args.backbone,
        activation="tanh",
        dropout_rate=args.dropout
    ).to(args.device)

    prompt_generator = PromptGenerator(
        input_dim=args.feature_dim + args.degree_dim,
        output_dim=args.pt_dim
    ).to(args.device)

    model_name = f"{args.dataset}_{args.model}_{args.backbone}_{args.seed}_best_model.pth"
    early_stopping = EarlyStopping(
        patience=args.patience,
        path=f"{args.model_save_path}/{model_name}",
        verbose=True,
    )
    prompt_generator_name = f"{args.dataset}_prompt_generator_{args.backbone}_{args.seed}_best_model.pth"

    gnn_optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    prompt_optimizer = torch.optim.Adam(prompt_generator.parameters(), lr=args.learning_rate)
    
    criterion = torch.nn.BCEWithLogitsLoss()

    train_data = dataset.build_graph(train_time[0], train_time[1])
    for epoch in range(args.epochs):
        # 1. 训练模型
        model.train()
        prompt_generator.train()
        gnn_optimizer.zero_grad()
        prompt_optimizer.zero_grad()

        # 2. 获取训练数据
        splitter = DataSplitter(args, train_data)
        splits = splitter.load_or_create_splits()
        train_edge_index = train_data.edge_index[:,splits["train_mask"]]
        x = train_data.x
        edge_times = train_data.edge_time[splits["train_mask"]]

        # 3. 划分训练数据
        sorted_indices = torch.argsort(edge_times)
        train_edge_index = train_edge_index[:, sorted_indices]
        edge_times = edge_times[sorted_indices]
        k = args.k_split  
        num_edges = train_edge_index.size(1)
        split_size = num_edges // k
        splits_indices = [i * split_size for i in range(k)] + [num_edges]

