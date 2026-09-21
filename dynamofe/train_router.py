"""Train the DynaMoFE Dynamic Neural Router (v2 Reliability-Supervised).

Trains a condition-dependent expert risk predictor conditioned on the 16-D physical
degradation signature vector with direct Huber risk supervision and confidence-gated
static prior fallback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from dynamofe.router import DynamicGatingRouter, DynaMoFEDetector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def load_benchmark_data() -> tuple[dict[str, Any], dict[str, pd.DataFrame], list[str], np.ndarray, list[str]]:
    """Load pre-extracted degradations and expert predictions."""
    deg_file = OUTPUT_ROOT / "degradations" / "ffpp_test_degradations_expanded28.pt"
    if not deg_file.exists():
        deg_file = OUTPUT_ROOT / "degradations" / "ffpp_test_degradations.pt"
    if not deg_file.exists():
        deg_file = DATA_ROOT / "degradations" / "ffpp_test_degradations.pt"
    if not deg_file.exists():
        raise FileNotFoundError(f"Degradation file not found: {deg_file}")

    deg_data = torch.load(deg_file, weights_only=False)
    environments = list(deg_data["degradations"].keys())
    labels = np.array(deg_data["labels"], dtype=np.float32)
    source_groups = [p.split("/")[-1].split("_")[0] for p in deg_data["paths"]]

    pred_files = {
        "fcg": DATA_ROOT / "predictions" / "fcg_official.csv",
        "tall": DATA_ROOT / "predictions" / "tall_local_seed42.csv",
        "f3net": DATA_ROOT / "predictions" / "f3net_deepfakebench.csv",
        "xception": DATA_ROOT / "predictions" / "xception_deepfakebench.csv",
    }
    expert_dfs = {k: pd.read_csv(p) for k, p in pred_files.items() if p.exists()}

    return deg_data, expert_dfs, environments, labels, source_groups


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Compute ROC-AUC, AP, and EER."""
    auc = roc_auc_score(y_true, y_score) * 100.0
    ap = average_precision_score(y_true, y_score) * 100.0
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0) * 100.0
    return {"auc": float(auc), "ap": float(ap), "eer": float(eer)}


def confidence_weighted_ranking_loss(
    pred_risks: Tensor,
    true_losses: Tensor,
    c_max: float = 2.0,
) -> Tensor:
    """Confidence-weighted pairwise ranking loss:
    c_{ij} = min(|true_loss_i - true_loss_j|, c_max)
    When true_loss_i < true_loss_j (expert i has lower loss / better performance),
    we want pred_risks_i < pred_risks_j (expert i has lower predicted risk).
    diff_true = true_loss_i - true_loss_j (< 0)
    diff_pred = pred_risk_i - pred_risk_j (< 0)
    sign_target = sign(diff_true) = -1
    sign_target * diff_pred = (-1) * (< 0) > 0.
    Misranking penalty: softplus(-sign_target * diff_pred).
    """
    diff_true = true_losses.unsqueeze(2) - true_losses.unsqueeze(1)
    diff_pred = pred_risks.unsqueeze(2) - pred_risks.unsqueeze(1)
    sign_target = torch.sign(diff_true)
    c_ij = torch.clamp(torch.abs(diff_true), min=0.0, max=c_max)
    mask = torch.triu(torch.ones(pred_risks.shape[1], pred_risks.shape[1], device=pred_risks.device), diagonal=1).bool()
    loss_matrix = c_ij * F.softplus(-sign_target * diff_pred)
    denom = c_ij[:, mask].sum() + 1e-7
    return loss_matrix[:, mask].sum() / denom


def pairwise_ranking_loss(pred_risks: Tensor, true_losses: Tensor) -> Tensor:
    """Pairwise ranking loss encouraging pred_risks_i < pred_risks_j when true_loss_i < true_loss_j."""
    diff_true = true_losses.unsqueeze(2) - true_losses.unsqueeze(1)
    diff_pred = pred_risks.unsqueeze(2) - pred_risks.unsqueeze(1)
    sign_target = torch.sign(diff_true)
    mask = torch.triu(torch.ones(pred_risks.shape[1], pred_risks.shape[1], device=pred_risks.device), diagonal=1).bool()
    loss_matrix = F.softplus(-sign_target * diff_pred)
    return loss_matrix[:, mask].mean()


class RouterDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        degradations: np.ndarray,
        expert_scores: np.ndarray,
        labels: np.ndarray,
        risks: np.ndarray,
        raw_losses: np.ndarray,
    ) -> None:
        self.degradations = torch.tensor(degradations, dtype=torch.float32)
        self.expert_scores = torch.tensor(expert_scores, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.risks = torch.tensor(risks, dtype=torch.float32)
        self.raw_losses = torch.tensor(raw_losses, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            self.degradations[idx],
            self.expert_scores[idx],
            self.labels[idx],
            self.risks[idx],
            self.raw_losses[idx],
        )


def train_router_epoch(
    model: DynaMoFEDetector,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta_risk: float = 0.5,
    beta_rank: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    huber_criterion = nn.HuberLoss(delta=0.5)

    for deg, scores, labels, risks, raw_losses in loader:
        deg = deg.to(device)
        scores = scores.to(device)
        labels = labels.to(device)
        risks = risks.to(device)
        raw_losses = raw_losses.to(device)

        optimizer.zero_grad()
        fused_scores, weights, pred_risk = model(deg, scores, return_risk=True)

        # Condition-dependent expert risk supervision
        loss_risk = huber_criterion(pred_risk, risks)

        # Pairwise expert ranking loss (confidence-weighted)
        loss_rank = confidence_weighted_ranking_loss(pred_risk, raw_losses)

        # Fused decision margin cross-entropy loss
        loss_bce = F.binary_cross_entropy_with_logits(fused_scores, labels)

        # Regularization on risk deviation
        loss_reg = 0.01 * torch.mean(pred_risk ** 2)

        loss = loss_bce + beta_risk * loss_risk + beta_rank * loss_rank + loss_reg
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
    num_epochs: int = 25,
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

    print(f"\n--- Starting {num_folds}-Fold Cluster Cross-Validation for DynaMoFE v2 (Experts: {experts}) ---")

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
        train_raw_losses = []
        y_train_tensor = torch.tensor(labels[train_mask], dtype=torch.float32)

        for env in environments:
            d_env = deg_data["degradations"][env][train_mask]
            s_env = env_scores[env][train_mask]
            # Standardize expert scores using training fold canonical stats
            s_std = (s_env - np.array(fold_means)) / (np.array(fold_stds) + 1e-7)

            # Per-expert BCE loss
            s_std_t = torch.tensor(s_std, dtype=torch.float32)
            exp_losses = torch.stack([
                F.binary_cross_entropy_with_logits(s_std_t[:, m], y_train_tensor, reduction='none')
                for m in range(M)
            ], dim=1).numpy()

            # Target relative risk centered around mean expert loss per sample
            r_rel = exp_losses - exp_losses.mean(axis=1, keepdims=True)

            train_degs.append(d_env)
            train_scs.append(s_env)
            train_lbs.append(labels[train_mask])
            train_risks.append(r_rel)
            train_raw_losses.append(exp_losses)

        train_degs = np.concatenate(train_degs, axis=0)
        train_scs = np.concatenate(train_scs, axis=0)
        train_lbs = np.concatenate(train_lbs, axis=0)
        train_risks = np.concatenate(train_risks, axis=0)
        train_raw_losses = np.concatenate(train_raw_losses, axis=0)

        # Per-fold degradation feature normalization strictly on training fold
        deg_mu = train_degs.mean(axis=0, keepdims=True)
        deg_sigma = train_degs.std(axis=0, keepdims=True) + 1e-7
        train_degs_norm = (train_degs - deg_mu) / deg_sigma

        train_ds = RouterDataset(train_degs_norm, train_scs, train_lbs, train_risks, train_raw_losses)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

        if M == 4:
            base_weights = [0.4, 0.2, 0.2, 0.2]
        elif M == 3:
            base_weights = [0.5, 0.25, 0.25]
        else:
            base_weights = [1.0 / M] * M

        router = DynamicGatingRouter(
            in_dim=K,
            num_experts=M,
            hidden_dim=hidden_dim,
            base_weights=base_weights,
            routing_mode="residual",
            delta_scale=0.25,
            use_tanh=True,
            use_confidence=False,
        ).to(device)
        detector = DynaMoFEDetector(router, expert_names=experts, means=fold_means, stds=fold_stds).to(device)
        optimizer = torch.optim.AdamW(detector.parameters(), lr=lr, weight_decay=1e-4)

        for epoch in range(num_epochs):
            train_loss = train_router_epoch(detector, train_loader, optimizer, device, beta_risk=0.5, beta_rank=0.5)

        # Evaluate out-of-fold for this fold on all environments
        for env in environments:
            val_deg_raw = deg_data["degradations"][env][val_mask]
            val_deg = (val_deg_raw - deg_mu) / deg_sigma
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


def run_leave_one_degradation_out(
    experts: Sequence[str] = ("fcg", "tall", "f3net", "xception"),
    num_epochs: int = 20,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    seed: int = 42,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> pd.DataFrame:
    """Evaluate zero-shot unseen codec generalization by holding out each codec in turn."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str)

    deg_data, expert_dfs, environments, labels, _ = load_benchmark_data()
    M = len(experts)
    K = deg_data["degradations"]["canonical"].shape[-1]
    w_static = np.array([0.40, 0.20, 0.20, 0.20] if M == 4 else [1.0 / M] * M)

    # Reference canonical stats
    can_means = [float(expert_dfs[exp][expert_dfs[exp]["codec_environment"] == "canonical"]["score"].mean()) for exp in experts]
    can_stds = [float(expert_dfs[exp][expert_dfs[exp]["codec_environment"] == "canonical"]["score"].std()) for exp in experts]

    env_scores = {}
    for env in environments:
        cols = []
        for exp in experts:
            df = expert_dfs[exp]
            sub = df[df["codec_environment"] == env].sort_values("video_id")
            cols.append(sub["score"].values)
        env_scores[env] = np.stack(cols, axis=1)

    held_out_candidates = [e for e in environments if e != "canonical"]
    rows = []

    print("\n=======================================================")
    print("Running Leave-One-Degradation-Out Evaluation:")
    print("=======================================================")

    for held_out in held_out_candidates:
        train_envs = [e for e in environments if e != held_out]

        train_degs = []
        train_scs = []
        train_lbs = []
        train_risks = []
        train_raw_losses = []
        y_tensor = torch.tensor(labels, dtype=torch.float32)

        for env in train_envs:
            d_env = deg_data["degradations"][env]
            s_env = env_scores[env]
            s_std = (s_env - np.array(can_means)) / (np.array(can_stds) + 1e-7)

            s_std_t = torch.tensor(s_std, dtype=torch.float32)
            exp_losses = torch.stack([
                F.binary_cross_entropy_with_logits(s_std_t[:, m], y_tensor, reduction='none')
                for m in range(M)
            ], dim=1).numpy()
            r_rel = exp_losses - exp_losses.mean(axis=1, keepdims=True)

            train_degs.append(d_env)
            train_scs.append(s_env)
            train_lbs.append(labels)
            train_risks.append(r_rel)
            train_raw_losses.append(exp_losses)

        train_degs = np.concatenate(train_degs, axis=0)
        train_scs = np.concatenate(train_scs, axis=0)
        train_lbs = np.concatenate(train_lbs, axis=0)
        train_risks = np.concatenate(train_risks, axis=0)
        train_raw_losses = np.concatenate(train_raw_losses, axis=0)

        train_ds = RouterDataset(train_degs, train_scs, train_lbs, train_risks, train_raw_losses)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

        router = DynamicGatingRouter(
            in_dim=K,
            num_experts=M,
            hidden_dim=hidden_dim,
            base_weights=w_static.tolist(),
            routing_mode="residual",
            delta_scale=0.5,
            use_confidence=False,
        ).to(device)
        detector = DynaMoFEDetector(router, expert_names=experts, means=can_means, stds=can_stds).to(device)
        optimizer = torch.optim.AdamW(detector.parameters(), lr=lr, weight_decay=1e-4)

        for epoch in range(num_epochs):
            train_router_epoch(detector, train_loader, optimizer, device, beta_risk=0.5, beta_rank=1.0)

        # Evaluate on the unseen held-out codec
        test_deg = deg_data["degradations"][held_out]
        test_sc = env_scores[held_out]
        metrics_dyn, preds_dyn, wts_dyn = evaluate_model(detector, test_deg, test_sc, labels, device)

        # Baseline: Static Quad Fusion
        s_std_test = (test_sc - np.array(can_means)) / (np.array(can_stds) + 1e-7)
        s_static = (w_static.reshape(1, -1) * s_std_test).sum(axis=-1)
        metrics_static = compute_metrics(labels, s_static)

        # Baseline: FCG Alone
        metrics_fcg = compute_metrics(labels, test_sc[:, 0])

        delta_static = metrics_dyn["auc"] - metrics_static["auc"]
        delta_fcg = metrics_dyn["auc"] - metrics_fcg["auc"]
        mean_w = wts_dyn.mean(axis=0).tolist()

        row = {
            "held_out_codec": held_out,
            "fcg_auc": metrics_fcg["auc"],
            "static_quad_auc": metrics_static["auc"],
            "dynamofe_zero_shot_auc": metrics_dyn["auc"],
            "delta_vs_static": delta_static,
            "delta_vs_fcg": delta_fcg,
            "mean_fcg_weight": mean_w[0],
            "mean_tall_weight": mean_w[1],
            "mean_f3net_weight": mean_w[2],
            "mean_xception_weight": mean_w[3],
        }
        rows.append(row)
        print(f"Held-Out Codec: {held_out:<28} | FCG: {metrics_fcg['auc']:.2f}% | Static: {metrics_static['auc']:.2f}% | DynaMoFE (Zero-Shot): {metrics_dyn['auc']:.2f}% (Delta: {delta_static:+.2f} pp)")

    df_lodo = pd.DataFrame(rows)
    return df_lodo


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

    # Run Leave-One-Degradation-Out
    df_lodo = run_leave_one_degradation_out(
        experts=args.experts,
        num_epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )
    lodo_file = OUTPUT_ROOT / "dynamofe_leave_one_out.csv"
    df_lodo.to_csv(lodo_file, index=False)
    print(f"Saved leave-one-out results to {lodo_file}")
