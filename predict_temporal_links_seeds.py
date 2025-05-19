import wandb
import argparse
import torch
import math
import os
import pandas as pd

from torch_geometric.utils import negative_sampling

from data_preprocess import EvolvingDataset, DataSplitter
from models import LinkPredictor, EarlyStopping
from utils import load_config, TemporalDataSplitter, get_link_labels, set_seed, add_time_features, add_degree_features, add_temporal_structure_features

TEST_K = 10
TEST_BATCH_SIZE = 1024

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

    if args.model == 'TimeEncoding':
        input_dim =  args.input_dim + args.time_dim
    elif args.model == 'DegreeEncoding':
        input_dim =  args.input_dim + args.degree_dim
    elif args.model == 'DTEncoding':
        input_dim =  args.input_dim + args.degree_dim + args.time_dim
    elif args.model == 'TDSP':
        input_dim =  args.input_dim + args.feature_dim + args.degree_dim
    else:
        input_dim = args.input_dim
    if args.dataset in [
        "tgbl-review",
        "tgbl-coin",
        "tgbl-flight",
        "tgbl-comment",
        "ogbl-collab",
    ]:
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

    model_name = f"{args.dataset}_{args.model}_{args.backbone}_{args.seed}_best_model.pth"
    early_stopping = EarlyStopping(
        patience=args.patience,
        path=f"{args.model_save_path}/{model_name}",
        verbose=True,
    )
    prompt_encoder_name = f"{args.dataset}_prompt_encoder_{args.backbone}_{args.seed}_best_model.pth"

    gnn_optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    pt_optimizer = torch.optim.Adam(prompt_encoder_name.parameters(), lr=args.learning_rate)
    
    criterion = torch.nn.BCEWithLogitsLoss()

    train_data = dataset.build_graph(train_time[0], train_time[1])
    for epoch in range(args.epochs):
        model.train()
        gnn_optimizer.zero_grad()
        splitter = DataSplitter(args, train_data)
        splits = splitter.load_or_create_splits()
        train_edge_index = train_data.edge_index[:,splits["train_mask"]]
        x = train_data.x
        num_nodes = x.size(0)

        ### Input Augmentation 
        if args.model == 'TimeEncoding':
            x = add_time_features(x, train_edge_index, train_data.edge_time, num_nodes, args.time_dim)
        elif args.model == 'DegreeEncoding':
            x = add_degree_features(x, train_edge_index, num_nodes, args.degree_dim)
        elif args.model == 'DTEncoding':
            x = add_time_features(x, train_edge_index, train_data.edge_time, num_nodes, args.time_dim)
            x = add_degree_features(x, train_edge_index, num_nodes, args.degree_dim)
        elif args.model == 'TSS':
            x = add_temporal_structure_features(x, train_edge_index, train_data.edge_time, num_nodes, args.tss_dim)
     
        neg_edge_index = negative_sampling(
            edge_index=train_edge_index,
            num_nodes=num_nodes,
            num_neg_samples=int(
                train_edge_index.size(1) * args.train_negative_sampling_ratio
            ),
        )
        link_labels = get_link_labels(train_edge_index, neg_edge_index).to(args.device)
        if args.dataset in [
            "tgbl-review",
            "tgbl-coin",
            "tgbl-flight",
            "tgbl-comment",
            "ogbl-collab",
        ]:
            train_edge_feature = train_data.edge_feature[splits["train_mask"]].to(args.device)
            z = model.encode(x.to(args.device), train_edge_index.to(args.device), train_edge_feature)
        elif args.dataset in ["tgbl-wiki"]:
            z = model.encode(x.to(args.device), train_edge_index.to(args.device))
        edge_index = torch.cat([train_edge_index.to(args.device), neg_edge_index.to(args.device)], dim=-1).long()
        link_logits = model.decode(z, edge_index)
        loss = criterion(link_logits, link_labels)
        loss.backward()
        gnn_optimizer.step()

        model.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            val_edge_index = splits["test_edges"]
            x = train_data.x
            if args.model == 'TimeEncoding':
                x = add_time_features(x, val_edge_index, train_data.edge_time, num_nodes, args.time_dim)
            elif args.model == 'DegreeEncoding':
                x = add_degree_features(x, val_edge_index, num_nodes, args.degree_dim)
            elif args.model == 'DTEncoding':
                x = add_time_features(x, val_edge_index, train_data.edge_time, num_nodes, args.time_dim)
                x = add_degree_features(x, val_edge_index, num_nodes, args.degree_dim)
            elif args.model == 'TSS':
                x = add_temporal_structure_features(x, val_edge_index, train_data.edge_time, num_nodes, args.tss_dim)

            val_neg_edge_index = splits["test_neg_edges"]
            val_link_labels = get_link_labels(val_edge_index, val_neg_edge_index).to(args.device)
            if args.dataset in [
                "tgbl-review",
                "tgbl-coin",
                "tgbl-flight",
                "tgbl-comment",
                "ogbl-collab",
            ]:
                train_edge_feature = train_data.edge_feature[splits["train_mask"]].to(args.device)
                val_z = model.encode(x.to(args.device), train_edge_index.to(args.device), train_edge_feature)
            elif args.dataset in ["tgbl-wiki"]:
                val_z = model.encode(x.to(args.device), train_edge_index.to(args.device))
            edge_index = torch.cat([val_edge_index.to(args.device), val_neg_edge_index.to(args.device)], dim=-1).long()
            val_link_logits = model.decode(val_z, edge_index)
            val_loss = criterion(val_link_logits, val_link_labels)
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "training_loss": loss.item(),
                    "validation_loss": val_loss.item(),
                }
            )
            early_stopping(val_loss, model)
            print(
                f"Epoch {epoch+1}/{args.epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}"
            )

        if early_stopping.early_stop:
            print("Early stopping triggered")
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