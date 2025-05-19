import wandb
import argparse
import torch
import math
import os
import pickle
import pandas as pd
import numpy as np

from data_preprocess import EvolvingDataset, DataSplitter
from models import NodeClassifier, EarlyStopping, NodeRegressor, TSSNodeRegressor
from utils import load_config, TemporalDataSplitter, get_node_labels, set_seed, add_time_features, add_degree_features, add_temporal_structure_features

MRE_EPSILON = 1e-7   # Add epsilon to avoid division by zero

@torch.no_grad()
def evaluate(args, model, dataset, test_time):
    model.eval()
    score_list = {}
    for i, period in enumerate(test_time):
        test_data = dataset.build_graph(period[0], period[1])
        splitter = DataSplitter(args, test_data)
        splits = splitter.load_or_create_splits()
        edge_index = test_data.edge_index 
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
        elif args.model in ['TSS', 'TSSM']:
            x = add_temporal_structure_features(x, edge_index, test_data.edge_time, num_nodes, args.tss_dim)

        logits, h = model(x.to(args.device), edge_index.to(args.device))

        if args.dataset in ["ogbn-arxiv", "SBM", 'penn', 'Amherst41', 'Cornell5', 'Johns_Hopkins55', 'Reed98']:
            labels = test_data.y.to(args.device)[splits['test_mask']].squeeze(1)
            preds = logits.argmax(dim=1)[splits["test_mask"]]
            acc = (preds == labels).sum().item() / labels.size(0)
            score_list[f"test_period_{i}_MicroACC"] = acc
            unique_classes = torch.unique(labels)
            class_accuracies = {}
            for cls in unique_classes:
                cls_mask = (labels == cls)
                cls_acc = (preds[cls_mask] == labels[cls_mask]).sum().item() / labels[cls_mask].size(0)
                class_accuracies[cls] = cls_acc
            score_list[f"test_period_{i}_MacroACC"] = np.mean(list(class_accuracies.values()))
            # output_data = {
            #     "hidden_embedding": h[splits['test_mask']].cpu().detach().numpy() ,
            #     "predicted_labels": preds.cpu().detach().numpy() ,
            #     "true_labels": labels.cpu().detach().numpy(),
            # }
            # output_file_path = f"./emb_pkl/{args.dataset}last_layer_output_{i}.pkl"
            # with open(output_file_path, "wb") as f:
            #     pickle.dump(output_data, f)

        elif args.dataset in ["BA-random"]:
            labels = get_node_labels(args.node_label_method, edge_index.T, num_nodes)[splits["test_mask"]].to(args.device)
            preds = logits
            numerator = torch.sum(torch.abs(labels - preds[splits["test_mask"]]))
            denominator = torch.sum(torch.abs(labels))

            mae = torch.mean(torch.abs(labels - preds[splits["test_mask"]]))
            rmse = torch.sqrt(torch.mean((labels - preds[splits["test_mask"]]) ** 2))
            mre = numerator / (denominator + MRE_EPSILON)  

            score_list[f"test_period_{i}_mae"] = mae.item()
            score_list[f"test_period_{i}_rmse"] = rmse.item()
            score_list[f"test_period_{i}_mre"] = mre.item()

        torch.cuda.empty_cache()
    return score_list

def predict_node_labels(args):
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
    elif args.model in ['TSS', 'TSSM']:
        input_dim =  args.input_dim + args.tss_dim
    else:
        input_dim = args.input_dim
    if args.dataset in ["ogbn-arxiv", "SBM", 'penn', 'Amherst41', 'Cornell5', 'Johns_Hopkins55', 'Reed98']:
        model = NodeClassifier(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_layers=args.num_layers,
            backbone=args.backbone,
            activation=args.activation,
            dropout_rate=args.dropout,
        ).to(args.device)
        criterion = torch.nn.CrossEntropyLoss()
    elif args.dataset in ["BA-random"]:
        if args.model in ['TSSM', 'DTEncoding']:
            model = TSSNodeRegressor(
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                output_dim=args.output_dim,
                num_layers=args.num_layers,
                backbone=args.backbone,
                activation="tanh",
                dropout_rate=args.dropout,
            ).to(args.device)
        else:
            model = NodeRegressor(
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                output_dim=args.output_dim,
                num_layers=args.num_layers,
                backbone=args.backbone,
                activation="tanh",
                dropout_rate=args.dropout,
            ).to(args.device)
        criterion = torch.nn.L1Loss()

    model_name = (
        f"{args.dataset}_{args.model}_{args.backbone}_{args.seed}_best_model.pth"
    )
    early_stopping = EarlyStopping(
        patience=args.patience,
        path=f"{args.model_save_path}/{model_name}",
        verbose=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    train_data = dataset.build_graph(train_time[0], train_time[1])
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        splitter = DataSplitter(args, train_data)
        splits = splitter.load_or_create_splits()
        train_edge_index = train_data.edge_index[:, splits["train_edge_mask"]]
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
        elif args.model in ['TSS', 'TSSM']:
            x = add_temporal_structure_features(x, train_edge_index, train_data.edge_time, num_nodes, args.tss_dim)

        logits, _ = model(x.to(args.device), train_edge_index.to(args.device))
        if args.dataset in ["ogbn-arxiv", "SBM", 'penn', 'Amherst41', 'Cornell5', 'Johns_Hopkins55', 'Reed98']:
            labels = train_data.y.squeeze(1).to(args.device)
        elif args.dataset in ["BA-random"]:  
            labels = get_node_labels(args.node_label_method, train_edge_index.T, num_nodes).to(args.device)

        loss = criterion(logits[splits["train_mask"]], labels[splits["train_mask"]])
        loss.backward()
        optimizer.step()

        model.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            val_edge_index = train_data.edge_index
            x = train_data.x
            if args.model == 'TimeEncoding':
                x = add_time_features(x, val_edge_index, train_data.edge_time, num_nodes, args.time_dim)
            elif args.model == 'DegreeEncoding':
                x = add_degree_features(x, val_edge_index, num_nodes, args.degree_dim)
            elif args.model == 'DTEncoding':
                x = add_time_features(x, val_edge_index, train_data.edge_time, num_nodes, args.time_dim)
                x = add_degree_features(x, val_edge_index, num_nodes, args.degree_dim)
            elif args.model in ['TSS', 'TSSM']:
                x = add_temporal_structure_features(x, val_edge_index, train_data.edge_time, num_nodes, args.tss_dim)

            logits, _ = model(x.to(args.device), val_edge_index.to(args.device))
            if args.dataset in ["ogbn-arxiv", "SBM",'penn', 'Amherst41', 'Cornell5', 'Johns_Hopkins55', 'Reed98']:
                labels = train_data.y.squeeze(1).to(args.device)
            elif args.dataset in ["BA-random"]:  
                labels = get_node_labels(args.node_label_method, val_edge_index.T, num_nodes).to(args.device)
            val_loss = criterion(logits[splits["test_mask"]], labels[splits["test_mask"]])
 
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "training_loss": loss.item(),
                    "validation_loss": val_loss.item(),
                }
            )
            early_stopping(val_loss, model)
            print(f"Epoch {epoch+1}/{args.epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
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
    parser = argparse.ArgumentParser(description="Graph Neural Network for Node Classification and Regression")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--seed_list", type=int, nargs="+", default=[42, 123, 456, 66], help="List of random seeds for experiments")
    parser.add_argument("--model_save_path", type=str, default="./model", help="Path to save model")
    parser.add_argument("--result_save_path", type=str, default="./result_tables", help="Path to save results.")
    parser.add_argument("--model", type=str, default="ERM", help="Model name")
    parser.add_argument("--activation", type=str, default="tanh", help="Activation method.")
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
        result = predict_node_labels(args)
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
            value = f"{stats['mean']:.4f}±{stats['std']:.4f}" if "mean" in stats and "std" in stats else "N/A"
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



