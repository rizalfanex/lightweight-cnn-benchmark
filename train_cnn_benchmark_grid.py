import os
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T
from torchvision.models import (
    mobilenet_v2, MobileNet_V2_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights,
    squeezenet1_1, SqueezeNet1_1_Weights,
    resnet18, ResNet18_Weights,
)

import matplotlib.pyplot as plt


# -------------------------
# Repro / IO
# -------------------------
def set_seed(seed: int = 42) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_run_dir(root: str = "runs") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(root, ts)
    os.makedirs(out, exist_ok=True)
    return out


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, topk=(1, 5)) -> List[float]:
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # [B, maxk]
    pred = pred.t()  # [maxk, B]
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append((correct_k / targets.size(0)).item())
    return res


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum, n = 0.0, 0
    top1_sum, top5_sum = 0.0, 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        b = y.size(0)
        loss_sum += loss.item() * b
        t1, t5 = accuracy_topk(logits, y, topk=(1, 5))
        top1_sum += t1 * b
        top5_sum += t5 * b
        n += b

    return {
        "val_loss": loss_sum / max(n, 1),
        "val_top1": top1_sum / max(n, 1),
        "val_top5": top5_sum / max(n, 1),
    }


@torch.no_grad()
def measure_latency_ms(
    model: nn.Module,
    device: torch.device,
    img_size: int,
    iters: int = 200,
    warmup: int = 30,
) -> float:
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)

    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t1 = time.time()
    return ((t1 - t0) / max(iters, 1)) * 1000.0


# -------------------------
# Transforms / Data
# -------------------------
def imagenet_norm():
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_transforms(img_size: int, aug_mode: str) -> Tuple[T.Compose, T.Compose]:
    mean, std = imagenet_norm()

    # NOTE: CIFAR100 is 32x32; resize then crop.
    if aug_mode == "basic":
        train_tf = T.Compose([
            T.Resize(img_size),
            T.RandomResizedCrop(img_size, scale=(0.80, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    elif aug_mode == "advanced":
        train_tf = T.Compose([
            T.Resize(img_size),
            T.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.06),
            T.RandomRotation(degrees=10),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        raise ValueError(f"Unknown aug_mode: {aug_mode}")

    test_tf = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_tf, test_tf


def build_loaders(
    data_dir: str,
    img_size: int,
    aug_mode: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_tf, test_tf = build_transforms(img_size, aug_mode)

    train_set = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_tf
    )
    test_set = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_tf
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, persistent_workers=(num_workers > 0)
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=(num_workers > 0)
    )
    return train_loader, test_loader


# -------------------------
# Models
# -------------------------
def make_model(model_name: str, num_classes: int = 100, pretrained: bool = True) -> nn.Module:
    mn = model_name.lower().strip()

    if mn == "mobilenet_v2":
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        m = mobilenet_v2(weights=weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m

    if mn == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = efficientnet_b0(weights=weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m

    if mn == "shufflenet_v2_x1_0":
        weights = ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        m = shufflenet_v2_x1_0(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if mn == "squeezenet1_1":
        weights = SqueezeNet1_1_Weights.DEFAULT if pretrained else None
        m = squeezenet1_1(weights=weights)
        m.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        return m

    if mn == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        m = resnet18(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    raise ValueError(f"Unknown model_name: {model_name}")


# -------------------------
# Train core
# -------------------------
def train_one_setting(
    model_name: str,
    img_size: int,
    aug_mode: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    amp: bool,
    grad_clip: float,
    pretrained: bool,
) -> Dict[str, Any]:
    setting_id = f"{model_name}__sz{img_size}__{aug_mode}"
    print(f"\n=== START {setting_id} ===")

    model = make_model(model_name, num_classes=100, pretrained=pretrained).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    best_top1 = -1.0
    history = {"epoch": [], "train_loss": [], "val_top1": [], "val_top5": [], "val_loss": []}

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            b = y.size(0)
            loss_sum += loss.item() * b
            n += b

        scheduler.step()

        train_loss = loss_sum / max(n, 1)
        val = evaluate(model, val_loader, device)

        history["epoch"].append(ep)
        history["train_loss"].append(train_loss)
        history["val_top1"].append(val["val_top1"])
        history["val_top5"].append(val["val_top5"])
        history["val_loss"].append(val["val_loss"])

        if val["val_top1"] > best_top1:
            best_top1 = val["val_top1"]
            ckpt_path = os.path.join(out_dir, f"{setting_id}_best.pt")
            torch.save(
                {"setting_id": setting_id, "state_dict": model.state_dict(), "best_val_top1": best_top1, "epoch": ep},
                ckpt_path,
            )

        print(
            f"[{setting_id}] ep {ep:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_top1={val['val_top1']:.4f} | val_top5={val['val_top5']:.4f}"
        )

    train_time_sec = time.time() - t0

    # load best for latency & final eval consistency
    best_ckpt = torch.load(os.path.join(out_dir, f"{setting_id}_best.pt"), map_location=device)
    model.load_state_dict(best_ckpt["state_dict"])
    model.eval()

    params = count_params(model)
    latency_ms = measure_latency_ms(model, device=device, img_size=img_size, iters=200, warmup=30)
    final_val = evaluate(model, val_loader, device)

    return {
        "setting_id": setting_id,
        "model": model_name,
        "img_size": img_size,
        "aug_mode": aug_mode,
        "params": int(params),
        "best_val_top1": float(best_top1),
        "final_val_top1": float(final_val["val_top1"]),
        "final_val_top5": float(final_val["val_top5"]),
        "final_val_loss": float(final_val["val_loss"]),
        "latency_ms": float(latency_ms),
        "train_time_sec": float(train_time_sec),
        "history": history,
    }


# -------------------------
# Aggregations (no pandas)
# -------------------------
def mean(xs: List[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def group_mean(rows: List[Dict[str, Any]], key: str, value_fields: List[str]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r[key]), []).append(r)

    out = []
    for g, items in sorted(groups.items(), key=lambda x: x[0]):
        agg = {key: g}
        for vf in value_fields:
            agg[vf] = mean([float(it[vf]) for it in items])
        out.append(agg)
    return out


def save_csv(rows: List[Dict[str, Any]], path: str, cols: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")


# -------------------------
# Plots
# -------------------------
def plot_tradeoff_by_setting(rows: List[Dict[str, Any]], out_path: str) -> None:
    plt.figure()
    for r in rows:
        x = r["params"] / 1e6
        y = r["best_val_top1"] * 100.0
        plt.scatter([x], [y])
    plt.xlabel("Trainable Params (Millions)")
    plt.ylabel("Best Val Top-1 (%)")
    plt.title("Accuracy–Efficiency Trade-off (All Settings)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_effect_bar(agg_rows: List[Dict[str, Any]], key: str, field: str, title: str, out_path: str) -> None:
    labels = [str(r[key]) for r in agg_rows]
    vals = [float(r[field]) * 100.0 for r in agg_rows]
    plt.figure()
    plt.bar(labels, vals)
    plt.xlabel(key)
    plt.ylabel(f"{field} (%)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_convergence_grid(
    histories: Dict[str, Dict[str, List[float]]],
    out_path: str,
    max_panels: int = 20
) -> None:
    # Show up to 20 settings; 20 experiments exactly matches default grid.
    keys = list(histories.keys())[:max_panels]
    n = len(keys)
    cols = 4
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols * 4.2, rows * 3.2))

    for i, k in enumerate(keys, 1):
        h = histories[k]
        plt.subplot(rows, cols, i)
        plt.plot(h["epoch"], h["val_top1"])
        plt.title(k, fontsize=8)
        plt.xlabel("Epoch")
        plt.ylabel("Val Top-1")
        plt.ylim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--epochs", type=int, default=40, help="40 recommended for overnight; 50 if you want extra")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)

    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--no-pretrained", action="store_true")

    ap.add_argument("--models", nargs="+",
                    default=["mobilenet_v2", "efficientnet_b0", "shufflenet_v2_x1_0", "squeezenet1_1", "resnet18"])
    ap.add_argument("--img-sizes", nargs="+", type=int, default=[128, 224])
    ap.add_argument("--aug-modes", nargs="+", default=["basic", "advanced"])
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(force_cpu=args.cpu)
    out_dir = now_run_dir(args.out_root)

    print(f"Device: {device}")
    print(f"Out: {out_dir}")

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    all_results: List[Dict[str, Any]] = []
    histories: Dict[str, Dict[str, List[float]]] = {}

    # Grid loop
    for img_size in args.img_sizes:
        for aug_mode in args.aug_modes:
            train_loader, val_loader = build_loaders(
                data_dir=args.data_dir,
                img_size=img_size,
                aug_mode=aug_mode,
                batch_size=args.batch,
                num_workers=args.num_workers,
            )

            for model_name in args.models:
                r = train_one_setting(
                    model_name=model_name,
                    img_size=img_size,
                    aug_mode=aug_mode,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    out_dir=out_dir,
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    amp=(not args.no_amp),
                    grad_clip=args.grad_clip,
                    pretrained=(not args.no_pretrained),
                )
                histories[r["setting_id"]] = r["history"]
                all_results.append({k: v for k, v in r.items() if k != "history"})

    # Save full JSON
    with open(os.path.join(out_dir, "metrics_full.json"), "w", encoding="utf-8") as f:
        json.dump({"results": all_results, "histories": histories}, f, indent=2)

    # Save full CSV
    full_cols = [
        "setting_id", "model", "img_size", "aug_mode",
        "params", "best_val_top1", "final_val_top1", "final_val_top5",
        "final_val_loss", "latency_ms", "train_time_sec"
    ]
    save_csv(all_results, os.path.join(out_dir, "results_full.csv"), full_cols)

    # Aggregations for paper tables
    agg_fields = ["best_val_top1", "final_val_top1", "final_val_top5", "latency_ms", "train_time_sec"]
    agg_by_model = group_mean(all_results, "model", agg_fields)
    agg_by_res = group_mean(all_results, "img_size", ["best_val_top1", "final_val_top1", "final_val_top5"])
    agg_by_aug = group_mean(all_results, "aug_mode", ["best_val_top1", "final_val_top1", "final_val_top5"])

    save_csv(agg_by_model, os.path.join(out_dir, "results_agg_by_model.csv"),
             ["model"] + agg_fields)
    save_csv(agg_by_res, os.path.join(out_dir, "results_agg_by_resolution.csv"),
             ["img_size", "best_val_top1", "final_val_top1", "final_val_top5"])
    save_csv(agg_by_aug, os.path.join(out_dir, "results_agg_by_aug.csv"),
             ["aug_mode", "best_val_top1", "final_val_top1", "final_val_top5"])

    # Plots
    plot_convergence_grid(histories, os.path.join(out_dir, "convergence_grid.png"))
    plot_tradeoff_by_setting(all_results, os.path.join(out_dir, "tradeoff_acc_vs_params.png"))
    plot_effect_bar(agg_by_res, "img_size", "final_val_top1",
                    "Resolution Effect (Avg Final Val Top-1)", os.path.join(out_dir, "resolution_effect.png"))
    plot_effect_bar(agg_by_aug, "aug_mode", "final_val_top1",
                    "Augmentation Effect (Avg Final Val Top-1)", os.path.join(out_dir, "augmentation_effect.png"))

    print("\n=== DONE ===")
    print("Key outputs:")
    print(f"- {os.path.join(out_dir, 'results_full.csv')}")
    print(f"- {os.path.join(out_dir, 'results_agg_by_model.csv')}")
    print(f"- {os.path.join(out_dir, 'results_agg_by_resolution.csv')}")
    print(f"- {os.path.join(out_dir, 'results_agg_by_aug.csv')}")
    print(f"- {os.path.join(out_dir, 'convergence_grid.png')}")
    print(f"- {os.path.join(out_dir, 'tradeoff_acc_vs_params.png')}")
    print(f"- {os.path.join(out_dir, 'resolution_effect.png')}")
    print(f"- {os.path.join(out_dir, 'augmentation_effect.png')}")
    print(f"- Checkpoints: *_best.pt in {out_dir}")


if __name__ == "__main__":
    main()