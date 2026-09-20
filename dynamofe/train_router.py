"""Train and evaluate DynaMoFE Dynamic Gating Router.

Uses source-identity cluster cross-validation to ensure leak-free evaluation.
Optimizes binary classification margin while maintaining expert entropy/diversity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from dynamofe.router import DynamicGatingRouter, DynaMoFEDetector


# Flexible root resolution: works standalone in DynaMoFE repo or inside deepfake-research
REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT_WORKSPACE = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else REPO_ROOT

if (REPO_ROOT / "data/predictions").exists() or (REPO_ROOT / "outputs/degradations").exists():
    PROJECT_ROOT = REPO_ROOT
    OUTPUT_ROOT = REPO_ROOT / "outputs"
    PRED_DIR = REPO_ROOT / "data/predictions" if (REPO_ROOT / "data/predictions").exists() else REPO_ROOT / "outputs"
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    OUTPUT_ROOT = PROJECT_ROOT / "outputs/dynamofe"
    PRED_DIR = PROJECT_ROOT / "outputs/external_codec_robustness_benchmark_v1/predictions/external"


def load_benchmark_data():
    """Load matched prediction tables and degradation features."""
    pred_dir = PRED_DIR
    deg_file = OUTPUT_ROOT / "degradations/ffpp_test_degradations.pt"

    if not deg_file.exists():
        raise FileNotFoundError(f"Degradations file not found: {deg_file}")

    deg_data = torch.load(deg_file, map_location="cpu", weights_only=False)
    environments = deg_data["environments"]
    paths = deg_data["paths"]
    labels = np.array(deg_data["labels"])

    # Load predictions for all experts
    fcg_df = pd.read_csv(pred_dir / "fcg_official.csv")
    tall_df = pd.read_csv(pred_dir / "tall_local_seed42.csv")
    f3net_df = pd.read_csv(pred_dir / "f3net_deepfakebench.csv")
    xcep_df = pd.read_csv(pred_dir / "xception_deepfakebench.csv")
    fa_df = pd.read_csv(pred_dir / "forensics_adapter_official.csv")

    source_groups = fcg_df[fcg_df["codec_environment"] == "canonical"].sort_values("video_id")["source_identity"].values

    expert_dfs = {
        "fcg": fcg_df,
        "tall": tall_df,
        "f3net": f3net_df,
        "xception": xcep_df,
        "forensics_adapter": fa_df,
    }

    return deg_data, expert_dfs, environments, labels, source_groups


def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    auc = roc_auc_score(y_true, scores) * 100.0
    ap = average_precision_score(y_true, scores) * 100.0
    # Equal error rate
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = 0.5 * (fpr[eer_idx] + fnr[eer_idx]) * 100.0
    return {"auc": float(auc), "ap": float(ap), "eer": float(eer)}


class RouterDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        degradations: np.ndarray,
        expert_scores: np.ndarray,
        labels: np.ndarray,
        risks: np.ndarray,
    ) -> None:
        self.degradations = torch.tensor(degradations, dtype=torch.float32)
        self.expert_scores = torch.tensor(expert_scores, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.risks = torch.tensor(risks, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.degradations[idx], self.expert_scores[idx], self.labels[idx], self.risks[idx]


def train_router_epoch(
    model: DynaMoFEDetector,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha_fused: float = 0.5,
) -> float:
    model.train()
    total_loss = 0.0
    mse_criterion = nn.MSELoss()

    for deg, scores, labels, risks in loader:
        deg = deg.to(device)
        scores = scores.to(device)
        labels = labels.to(device)
        risks = risks.to(device)

        optimizer.zero_grad()
        delta = model.router.net(deg)  # (B, M)
        # Condition-dependent expert risk MSE loss (delta approximates negative relative risk)
        loss_risk = mse_criterion(delta, -risks)

        # Fused decision margin loss
        fused_scores, weights = model(deg, scores)
        y_sign = 2.0 * labels - 1.0
        loss_fused = F.relu(1.0 - y_sign * fused_scores).mean()

        loss = loss_risk + alpha_fused * loss_fused
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * len(labels)

    return total_loss / len(loader.dataset)


def evaluate_model(
    model: DynaMoFEDetector,
    deg: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        deg_t = torch.tensor(deg, dtype=torch.float32).to(device)
        scores_t = torch.tensor(scores, dtype=torch.float32).to(device)
        fused, weights = model(deg_t, scores_t)
        fused_np = fused.cpu().numpy()
        weights_np = weights.cpu().numpy()

    metrics = compute_metrics(labels, fused_np)
    return metrics, fused_np, weights_np


def run_cross_validation(
    experts: Sequence[str] = ("fcg", "tall", "f3net", "xception"),
    num_folds: int = 5,
    num_epochs: int = 20,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    seed: int = 42,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str)

    deg_data, expert_dfs, environments, labels, source_groups = load_benchmark_data()
    unique_groups = np.unique(source_groups)
    np.random.shuffle(unique_groups)
    fold_groups = np.array_split(unique_groups, num_folds)

    N_videos = len(labels)
    M = len(experts)
    K = deg_data["degradations"]["canonical"].shape[-1]

    # Pre-extract scores into unified arrays per environment: env -> (700, M)
    env_scores = {}
    for env in environments:
        cols = []
        for exp in experts:
            df = expert_dfs[exp]
            sub = df[df["codec_environment"] == env].sort_values("video_id")
            cols.append(sub["score"].values)
        env_scores[env] = np.stack(cols, axis=1)  # (700, M)

    # Out-of-fold prediction storage: env -> (700,)
    oof_predictions = {env: np.zeros(N_videos, dtype=np.float32) for env in environments}
    oof_weights = {env: np.zeros((N_videos, M), dtype=np.float32) for env in environments}

    print(f"\n--- Starting {num_folds}-Fold Cluster Cross-Validation for DynaMoFE (Experts: {experts}) ---")

    for fold_idx in range(num_folds):
        val_clusters = set(fold_groups[fold_idx])
        train_mask = np.array([g not in val_clusters for g in source_groups])
        val_mask = ~train_mask

        # Clean fold-internal canonical stats for leak-free score standardization
        train_can_scs = env_scores["canonical"][train_mask]
        fold_means = train_can_scs.mean(axis=0).tolist()
        fold_stds = train_can_scs.std(axis=0).tolist()

        # Collect training data across all environments for the training clusters
        train_degs = []
        train_scs = []
        train_lbs = []
        train_risks = []
        y_train_sign = 2.0 * labels[train_mask] - 1.0

        for env in environments:
            d_env = deg_data["degradations"][env][train_mask]
            s_env = env_scores[env][train_mask]
            # Standardize expert scores using training fold canonical stats
            s_std = (s_env - np.array(fold_means)) / (np.array(fold_stds) + 1e-7)
            # Sample-level margin error per expert: max(0, 1 - y_sign * s_std)
            r_env = np.maximum(0.0, 1.0 - y_train_sign[:, None] * s_std)
            # Relative risk centered around mean expert risk per sample
            r_rel = r_env - r_env.mean(axis=1, keepdims=True)

            train_degs.append(d_env)
            train_scs.append(s_env)
            train_lbs.append(labels[train_mask])
            train_risks.append(r_rel)

        train_degs = np.concatenate(train_degs, axis=0)
        train_scs = np.concatenate(train_scs, axis=0)
        train_lbs = np.concatenate(train_lbs, axis=0)
        train_risks = np.concatenate(train_risks, axis=0)

        train_ds = RouterDataset(train_degs, train_scs, train_lbs, train_risks)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

        if M == 4:
            base_weights = [0.4, 0.2, 0.2, 0.2]
        elif M == 3:
            base_weights = [0.5, 0.25, 0.25]
        else:
            base_weights = [1.0 / M] * M

        router = DynamicGatingRouter(in_dim=K, num_experts=M, hidden_dim=hidden_dim, base_weights=base_weights).to(device)
        detector = DynaMoFEDetector(router, expert_names=experts, means=fold_means, stds=fold_stds).to(device)
        optimizer = torch.optim.AdamW(detector.parameters(), lr=lr, weight_decay=1e-3)

        for epoch in range(num_epochs):
            train_loss = train_router_epoch(detector, train_loader, optimizer, device)

        # Evaluate out-of-fold for this fold on all environments
        for env in environments:
            val_deg = deg_data["degradations"][env][val_mask]
            val_sc = env_scores[env][val_mask]
            val_lb = labels[val_mask]

            metrics, preds, wts = evaluate_model(detector, val_deg, val_sc, val_lb, device)
            oof_predictions[env][val_mask] = preds
            oof_weights[env][val_mask] = wts

        print(f"Fold {fold_idx+1}/{num_folds} complete.")

    # Compute overall Out-Of-Fold metrics across all environments
    summary_results = []
    for env in environments:
        preds = oof_predictions[env]
        wts = oof_weights[env]
        m = compute_metrics(labels, preds)
        mean_w = wts.mean(axis=0).tolist()
        summary_results.append({
            "environment": env,
            "auc": m["auc"],
            "ap": m["ap"],
            "eer": m["eer"],
            "mean_weights": {exp: round(mean_w[i], 4) for i, exp in enumerate(experts)},
        })

    df_summary = pd.DataFrame(summary_results)
    sealed_sub = df_summary[df_summary["environment"] != "canonical"]
    sealed_mean_auc = float(sealed_sub["auc"].mean())
    sealed_worst_auc = float(sealed_sub["auc"].min())
    canonical_auc = float(df_summary[df_summary["environment"] == "canonical"]["auc"].iloc[0])

    print("\n=======================================================")
    print("DynaMoFE Out-Of-Fold Cross-Validation Summary Results:")
    print("=======================================================")
    for row in summary_results:
        print(f"  {row['environment']:<28}: AUC = {row['auc']:.4f}%, AP = {row['ap']:.4f}%, Weights = {row['mean_weights']}")
    print(f"\nCanonical AUC:    {canonical_auc:.4f}%")
    print(f"Sealed Mean AUC:  {sealed_mean_auc:.4f}%")
    print(f"Worst Codec AUC:  {sealed_worst_auc:.4f}%")
    print("=======================================================")

    results = {
        "experts": list(experts),
        "canonical_auc": canonical_auc,
        "sealed_mean_auc": sealed_mean_auc,
        "sealed_worst_auc": sealed_worst_auc,
        "per_environment": summary_results,
        "oof_predictions": {env: oof_predictions[env].tolist() for env in environments},
        "oof_weights": {env: oof_weights[env].tolist() for env in environments},
    }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", nargs="+", default=["fcg", "tall", "f3net", "xception"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    res = run_cross_validation(
        experts=args.experts,
        num_folds=args.folds,
        num_epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )

    out_file = OUTPUT_ROOT / "dynamofe_results.json"
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved results to {out_file}")
