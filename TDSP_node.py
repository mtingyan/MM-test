import wandb
import argparse
import torch
import math
import os
import pandas as pd
import torch.nn.functional as F

from torch_geometric.utils import negative_sampling

from data_preprocess import EvolvingDataset, DataSplitter
from models import CombinedModel, NodeClassifier,  NodeRegressor, PromptGenerator, GNNEncoder
from utils import EarlyStopping, load_config, TemporalDataSplitter, get_link_labels, set_seed, compute_node_degrees

MRE_EPSILON = 1e-7

def compute_feature_prompt(h_prev, h_curr, bins=5, eps=1e-8, clip_max=10.0):
    # 计算 softmax 并添加数值稳定性
    softmax_prev = F.softmax(h_prev, dim=-1).clamp(min=eps, max=1 - eps)
    softmax_curr = F.softmax(h_curr, dim=-1).clamp(min=eps, max=1 - eps)
    
    # 计算 KL 散度（方向为 softmax_prev || softmax_curr）
    kl_div = F.kl_div(softmax_curr.log(), softmax_prev, reduction='none')
    
    # 裁剪 KL 散度值，防止极端值影响分桶
    kl_div_clipped = kl_div.clamp(max=clip_max)
    
    # 生成分桶边界（等宽分桶）
    min_kl, max_kl = kl_div.min(), kl_div.max()
    if min_kl == max_kl:
        # 所有值相等，避免除以零
        bins_tensor = torch.tensor([0.0], device=h_prev.device)
    else:
        bins_tensor = torch.linspace(min_kl.item(), max_kl.item(), steps=bins + 1, device=h_prev.device)
    
    # 分桶映射
    bucket_indices = torch.bucketize(kl_div_clipped.view(-1), bins_tensor).long()
    
    # 生成 one-hot 向量
    one_hot = torch.nn.functional.one_hot(bucket_indices, num_classes=bins).float()
    one_hot = one_hot.view(*kl_div.shape, bins)
    
    return one_hot


def compute_degree_prompt(prev_degrees, curr_degrees, num_buckets=10):
    # 计算度的变化
    degree_changes = curr_degrees - prev_degrees
    
    # 将当前度数分桶
    min_degree = curr_degrees.min()
    max_degree = curr_degrees.max()
    degree_bucket_size = (max_degree - min_degree) / num_buckets
    
    # 将度数变化分桶
    min_change = degree_changes.min()
    max_change = degree_changes.max()
    change_bucket_size = (max_change - min_change) / num_buckets
    
    # 将每个节点的当前度数映射到相应的桶
    bucketed_degrees = torch.floor((curr_degrees - min_degree) / degree_bucket_size).long()
    
    # 将每个节点的度变化映射到相应的桶
    bucketed_changes = torch.floor((degree_changes - min_change) / change_bucket_size).long()
    
    # 将桶索引转换为one-hot向量
    one_hot_degrees = F.one_hot(bucketed_degrees, num_classes=num_buckets)
    one_hot_changes = F.one_hot(bucketed_changes, num_classes=num_buckets)
    
    # 连接当前度数和度数变化的one-hot向量
    degree_prompt = torch.cat([one_hot_degrees, one_hot_changes], dim=1)
    
    return degree_prompt.float()

@torch.no_grad()
def evaluate(args, model, prompt_generator, dataset, time_periods):
    scores_list = {}
    model.eval()
    prompt_generator.eval()
    for i, time_period in enumerate(time_periods):
        
        test_data = dataset.build_graph(time_period[0], time_period[1])
        test_splitter = DataSplitter(args, test_data)
        test_splits = test_splitter.load_or_create_splits()
        edge_index = test_data.edge_index[:,test_splits["train_mask"]]
        x = test_data.x.to(args.device)
        num_nodes = x.size(0)
        edge_times = test_data.edge_time[test_splits["train_mask"]]
        test_edge_feature = test_data.edge_feature[test_splits["train_mask"]]

        if i == 0:
            prev_edge_num = edge_index.size(1)
            continue
        # 1. 划分测试数据
        sorted_indices = torch.argsort(edge_times)
        test_edge_index = edge_index[:, sorted_indices]
        edge_times = edge_times[sorted_indices]
        test_edge_feature = test_edge_feature[:, sorted_indices]
        
        # 2. 获取测试数据
        prompt = torch.zeros((num_nodes, args.pt_dim))
        x_aug = torch.cat([x, prompt], dim=-1)
        
        h_prev = model.encode(x_aug.to(args.device), test_edge_index[:, :prev_edge_num].to(args.device), test_edge_feature[:, :prev_edge_num].to(args.device))
        h_curr = model.encode(x_aug.to(args.device), test_edge_index.to(args.device), test_edge_feature.to(args.device))
        
        f_prompt = compute_feature_prompt(h_prev, h_curr, bins=args.feature_dim)
        d_prev = compute_node_degrees(test_edge_index[:, :prev_edge_num], num_nodes)
        d_curr = compute_node_degrees(test_edge_index, num_nodes)
        d_prompt = compute_degree_prompt(d_prev, d_curr, num_buckets=args.degree_dim)

        prompt = torch.cat([f_prompt, d_prompt], dim=-1).to(args.device)
        
        prompt = prompt_generator(prompt)

        x_aug = torch.cat([x, prompt], dim=-1)
        
        test_z = model.encode(x_aug.to(args.device), test_edge_index.to(args.device), test_edge_feature.to(args.device))

        test_edge_index = test_splits["test_edges"].to(args.device)
        test_neg_edge_index = test_splits["test_neg_edges"].to(args.device)
        pos_scores = []
        neg_scores = []
        num_neg_samples = int(num_nodes * args.test_negative_sampling_ratio)
        
        for start in range(0, test_edge_index.size(1), TEST_BATCH_SIZE):
            end = start + TEST_BATCH_SIZE
            pos_scores.append(model.decode(test_z, test_edge_index[:, start:end]))

        for start in range(0, test_neg_edge_index.size(1), TEST_BATCH_SIZE * num_neg_samples):
            end = start + TEST_BATCH_SIZE * num_neg_samples
            neg_scores.append(model.decode(test_z, test_neg_edge_index[:, start:end]))

        positive_scores = torch.cat(pos_scores).view(-1, 1)
        negative_scores = torch.cat(neg_scores).view(-1, num_neg_samples)
        scores = torch.cat([positive_scores, negative_scores], dim=1)
        ranks = torch.argsort(torch.argsort(scores, dim=1, descending=True), dim=1)

        reciprocal_ranks = 1.0 / (ranks[:, 0] + 1).float()
        mrr = reciprocal_ranks.mean().item()
        scores_list[f"test_period_{i}_mrr"] = mrr

        top_k_hits = (ranks[:, :TEST_K] == 0).float().sum(dim=1)  
        top_k_accuracy = top_k_hits.mean().item()
        scores_list[f"test_period_{i}_hit@k"] = top_k_accuracy
        scores_at_k = scores[:, :TEST_K]
        dcg_at_k = torch.sum((2.0 ** scores_at_k - 1) / torch.log2(ranks[:, :TEST_K].float() + 2), dim=1)
        sorted_scores = torch.sort(scores, dim=1, descending=True)[0]  
        ideal_scores_at_k = sorted_scores[:, :TEST_K]  
        ideal_dcg_at_k = torch.sum((2.0 ** ideal_scores_at_k - 1) / torch.log2(torch.arange(1, TEST_K + 1, device=scores.device).float() + 1), dim=1)
        ndcg_at_k = (dcg_at_k / ideal_dcg_at_k).mean().item()
        scores_list[f"test_period_{i}_ndcg@k"] = ndcg_at_k
        torch.cuda.empty_cache()
        
    return scores_list

def predict_node_labels(args):
    set_seed(args.seed)

    dataset = EvolvingDataset(args)
    train_time, test_time_list = TemporalDataSplitter(args, dataset).split_by_time()
    print(f"Training data ends in {train_time[0]-1}")
    print(f"Validation data starts from {train_time[0]}, ends in {train_time[1]}")
    print("Test data:")
    for i, test_time in enumerate(test_time_list):
        print(f"Test time {i + 1}: {test_time}")
    if args.dataset in ["ogbn-arxiv", "SBM", 'penn', 'Amherst41', 'Cornell5', 'Johns_Hopkins55', 'Reed98']:
        classifier = NodeClassifier(
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim
        ).to(args.device)
    elif args.dataset in ["BA-random"]:
        classifier = NodeRegressor(
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim
        ).to(args.device)       

    gnn_encoder = GNNEncoder(
        input_dim=args.input_dim + args.pt_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        edge_feature_dim=args.edge_dim,
        backbone=args.backbone,
        activation="tanh",
        dropout_rate=args.dropout
    ).to(args.device)

    model = CombinedModel(gnn_encoder, classifier).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model_name = f"{args.dataset}_{args.model}_{args.backbone}_{args.seed}_best_model.pth"
    early_stopping = EarlyStopping(
        patience=args.patience,
        path=f"{args.model_save_path}/{model_name}",
        verbose=True,
    )

    prompt_generator = PromptGenerator(
        input_dim=args.feature_dim + args.degree_dim,
        output_dim=args.pt_dim
    ).to(args.device)
    prompt_generator_name = f"{args.dataset}_prompt_generator_{args.backbone}_{args.seed}_best_model.pth"
    prompt_optimizer = torch.optim.Adam(prompt_generator.parameters(), lr=args.learning_rate)
    
    criterion = torch.nn.L1Loss()

    train_data = dataset.build_graph(train_time[0], train_time[1])
    for epoch in range(args.epochs):
        # 1. 训练模型
        model.train()
        prompt_generator.train()
        optimizer.zero_grad()
        prompt_optimizer.zero_grad()

        # 2. 获取训练数据
        splitter = DataSplitter(args, train_data)
        splits = splitter.load_or_create_splits()
        train_edge_index = train_data.edge_index[:,splits["train_edge_mask"]]
        x = train_data.x.to(args.device)
        num_nodes = x.size(0)
        edge_times = train_data.edge_time[splits["train_mask"]]
        train_edge_feature = train_data.edge_feature[splits["train_mask"]]

        # 3. 划分训练数据
        sorted_indices = torch.argsort(edge_times)
        train_edge_index = train_edge_index[:, sorted_indices]
        edge_times = edge_times[sorted_indices]
        train_edge_feature = train_edge_feature[:, sorted_indices]
        k = args.k_split  
        num_edges = train_edge_index.size(1)
        split_size = num_edges // k
        splits_indices = [i * split_size for i in range(k)] + [num_edges]
        
        tr_loss = 0
        consistency_loss = 0
        contrastive_loss = 0
        for i in range(1, k):

            prompt = torch.zeros((num_nodes, args.pt_dim))
            x_aug = torch.cat([x, prompt], dim=-1)
            
            h_prev = model.encode(x_aug.to(args.device), train_edge_index[:, :splits_indices[i-1]].to(args.device), train_edge_feature[:, :splits_indices[i-1]].to(args.device))
            h_curr = model.encode(x_aug.to(args.device), train_edge_index[:, :splits_indices[i]].to(args.device), train_edge_feature[:, :splits_indices[i]].to(args.device))

            f_prompt = compute_feature_prompt(h_prev, h_curr, bins=args.feature_dim)
            d_prev = compute_node_degrees(train_edge_index[:, :splits_indices[i-1]], num_nodes)
            d_curr = compute_node_degrees(train_edge_index[:, :splits_indices[i]], num_nodes)
            d_prompt = compute_degree_prompt(d_prev, d_curr, num_buckets=args.degree_dim)

            consistency_loss += F.mse_loss()
            prompt = torch.cat([f_prompt, d_prompt], dim=-1).to(args.device)
            prompt = prompt_generator(prompt)
            if i != 1:
                consistency_loss += F.mse_loss(prompt, last_prompt)
            last_prompt = prompt
            neg_edge_index = negative_sampling(
                edge_index=train_edge_index[:, :splits_indices[i]],
                num_nodes=num_nodes,
                num_neg_samples=int(
                    train_edge_index[:, :splits_indices[i]].size(1) * args.train_negative_sampling_ratio
                ),
            )
            x_aug = torch.cat([x, prompt], dim=-1)
            z = model.encode(x_aug.to(args.device), train_edge_index[:, :splits_indices[i]].to(args.device), train_edge_feature[:, :splits_indices[i]].to(args.device))

            link_labels = get_link_labels(train_edge_index[:, :splits_indices[i]], neg_edge_index).to(args.device)
            edge_index = torch.cat([train_edge_index[:, :splits_indices[i]].to(args.device), neg_edge_index.to(args.device)], dim=-1).long()
            link_logits = model.decode(z, edge_index)
            tr_loss += criterion(link_logits, link_labels)
            pos_score = F.cosine_similarity(prompt[train_edge_index[:, :splits_indices[i]][0]], prompt[train_edge_index[:, :splits_indices[i]][1]])
            neg_score = F.cosine_similarity(prompt[neg_edge_index[0]], prompt[neg_edge_index[1]])
            contrastive_loss += torch.mean(F.relu(neg_score - pos_score + args.margin))

        tr_loss /= k - 1
        consistency_loss /= k - 2
        contrastive_loss /= k - 1
        total_loss = tr_loss + args.consistency_weight * consistency_loss + args.contrastive_weight * contrastive_loss
        total_loss.backward()
        optimizer.step()
        prompt_optimizer.step()


        model.eval()
        prompt_generator.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            val_edge_index = splits["test_edges"]
            prompt = torch.zeros((num_nodes, args.pt_dim))
            x = train_data.x
            x_aug = torch.cat([x, prompt], dim=-1)
            h_prev = model.encode(x_aug.to(args.device), train_edge_index[:, :splits_indices[-2]].to(args.device), train_edge_feature[:, :splits_indices[-2]].to(args.device))
            h_curr = model.encode(x_aug.to(args.device), train_edge_index[:, :splits_indices[-1]].to(args.device), train_edge_feature[:, :splits_indices[-1]].to(args.device))

            f_prompt = compute_feature_prompt(h_prev, h_curr, bins=args.feature_dim)
            d_prev = compute_node_degrees(train_edge_index[:, :splits_indices[-2]], num_nodes)
            d_curr = compute_node_degrees(train_edge_index[:, :splits_indices[-1]], num_nodes)
            d_prompt = compute_degree_prompt(d_prev, d_curr, num_buckets=args.degree_dim)

            prompt = torch.cat([f_prompt, d_prompt], dim=-1).to(args.device)
            prompt = prompt_generator(prompt)

            x_aug = torch.cat([x, prompt], dim=-1)
            val_z = model.encode(x_aug.to(args.device), train_edge_index[:, :splits_indices[-1]].to(args.device), train_edge_feature[:, :splits_indices[-1]].to(args.device))

            val_neg_edge_index = splits["test_neg_edges"]
            val_link_labels = get_link_labels(val_edge_index, val_neg_edge_index).to(args.device)

            edge_index = torch.cat([val_edge_index.to(args.device), val_neg_edge_index.to(args.device)], dim=-1).long()
            val_link_logits = model.decode(val_z, edge_index)
            val_loss = criterion(val_link_logits, val_link_labels)
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "training_loss": tr_loss.item(),
                    "validation_loss": val_loss.item(),
                }
            )
            early_stopping(val_loss, model)
            print(
                f"Epoch {epoch+1}/{args.epochs}, Training Loss: {tr_loss.item()}, Validation Loss: {val_loss.item()}"
            )

        if early_stopping.early_stop:
            print("Early stopping triggered")
            torch.save(model.state_dict(), f"{args.model_save_path}/{model_name}")
            torch.save(prompt_generator.state_dict(), f"{args.model_save_path}/{prompt_generator_name}")
            break
        
    if args.dataset in ["tgbl-review"]:
        scores_list = evaluate(args, model, dataset, [train_time] + test_time_list[:-1])
    else:
        scores_list = evaluate(args, model, dataset, [train_time] + test_time_list)
    grouped_scores = {}
    for key, value in scores_list.items():
        period = key.split("_")[2]
        metric = key.split("_")[-1]
        if period not in grouped_scores:
            grouped_scores[period] = {}
        grouped_scores[period][metric] = value 

    for i, metrics in grouped_scores.items():
        wandb_log_data = {}
        metric_strings = []

        for metric, value in metrics.items():
            wandb_log_data[f"test_period_{i}_{metric}"] = value
            metric_strings.append(
                f"{metric.upper()}: {value:.4f}"
                if isinstance(value, (int, float))
                else f"{metric.upper()}: {value}"
            )

        wandb.log(wandb_log_data)
        print(f"Test period {i}: " + ", ".join(metric_strings))     
    return scores_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph Neural Network for Link Prediction")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument( "--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--seed_list", type=int, nargs="+", default=[42, 123, 456, 66], help="List of random seeds for experiments")
    parser.add_argument("--result_save_path", type=str, default="./result_tables", help="Path to save results.")
    parser.add_argument("--model_save_path", type=str, default="./model", help="Path to save model")
    parser.add_argument("--model", type=str, default="ERM", help="Model name")
    parser.add_argument("--project_name", type=str, default="TDG", help="Wandb project name")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training (e.g., 'cpu' or '0')")
    args = parser.parse_args()
    config = load_config(args.config)

    for key, value in config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
    args.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if args.device.type == "cuda":
        print(f"Using device: {args.device}, GPU: {torch.cuda.get_device_name(args.device.index)}")
    else:
        print("Using device: CPU")
    wandb_run_id = f"{args.project_name}_{args.dataset}_{args.model}_{args.backbone}"
    wandb.init(project=args.project_name, config=args, resume=None, id=wandb_run_id)
    all_results = []
    for seed in args.seed_list:
        print(f"Running experiment with seed {seed}")
        args.seed = seed
        result = predict_links(args)
        all_results.append(result)

    final_scores = {}
    for period_metric in all_results[0].keys():
        period = "_".join(period_metric.split("_")[0:3])
        metric = period_metric.split("_")[-1]
        if period not in final_scores:
            final_scores[period] = {}

        values = [result[period_metric] for result in all_results]
        mean_val = sum(values) / len(values)
        variance_val = sum((x - mean_val) ** 2 for x in values) / len(values)
        std_val = math.sqrt(variance_val)
        final_scores[period][metric] = {"mean": mean_val, "std": std_val}

    print("Final Results:")
    for period, metrics in final_scores.items():
        metric_strings = []
        for metric, stats in metrics.items():
            metric_strings.append(f"{metric.upper()}: Mean = {stats['mean']:.4f}, Std = {stats['std']:.4f}")
            wandb.log(
                {
                    f"{period}_{metric}_mean": stats["mean"],
                    f"{period}_{metric}_std": stats["std"],
                }
            )
        print(f"{period}: {', '.join(metric_strings)}")
    wandb.finish()

    ###
save_model = args.model

periods = list(final_scores.keys())
metrics = list(next(iter(final_scores.values())).keys())
columns = ["Model"]
for metric in metrics:
    for period in periods:
        columns.append(f"{period}_{metric}")

rows = []
row = [save_model]  
for metric in metrics: 
    for period in periods:
        stats = final_scores[period].get(metric, {"mean": "N/A", "std": "N/A"})
        if "mean" in stats and "std" in stats:
            value = f"{stats['mean'] * 100:.4f}%±{stats['std'] * 100:.4f}%"
        else:
            value = "N/A"
        row.append(value)

rows.append(row)
new_data_df = pd.DataFrame(rows, columns=columns)
result_path = os.path.join(args.result_save_path, f'{args.dataset}.xlsx')
print(result_path)

if os.path.exists(result_path):
    existing_df = pd.read_excel(result_path)
    updated_df = pd.concat([existing_df, new_data_df], ignore_index=True)
else:
    updated_df = new_data_df

updated_df.to_excel(result_path, index=False)
