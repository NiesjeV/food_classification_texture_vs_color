"""
analyze.py — Per-category color vs texture analysis

Loads training history JSONs from all streams, computes the
colour-dominance score per class, and generates the key figures
for your poster.

Usage:
  python analyze.py --out_dir ./outputs

Outputs (all in --out_dir):
  color_vs_texture.csv     — per-class acc_C, acc_T, acc_F, score
  fig_ranked_bar.png       — ranked bar chart for the poster
  fig_stream_curves.png    — val accuracy curves across streams
  fig_scatter.png          — acc_C vs acc_T scatter per class
"""

import argparse
import json
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader

from dataset import FoodDataset, get_transform
from model import get_model


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir",      default="./outputs")
    p.add_argument("--ckpt_dir",     default="./checkpoints")
    p.add_argument("--data_dir",     default="./data")
    p.add_argument("--num_classes",  type=int, default=80)
    p.add_argument("--embed_dim",    type=int, default=512)
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--num_workers",  type=int, default=4)
    # Optional: class name mapping file (one name per line, index = class id)
    p.add_argument("--class_names",  default=None,
                   help="Text file with one class name per line")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Load class names (optional)
# ---------------------------------------------------------------------------

def load_class_names(path, num_classes):
    if path and Path(path).exists():
        with open(path) as f:
            names = [l.strip() for l in f.readlines()]
        return names[:num_classes]
    return [str(i) for i in range(num_classes)]


# ---------------------------------------------------------------------------
# Evaluate per-class accuracy from a checkpoint on the val set
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_per_class(stream, ckpt_path, data_dir, num_classes, embed_dim,
                   batch_size, num_workers, device):
    model = get_model(stream, num_classes, embed_dim).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    # We use the val split of the labeled data
    dataset = FoodDataset(data_dir, split="train", stream=stream, augment=False)

    # Recreate the same val split used during training
    import torch
    from torch.utils.data import random_split
    n = len(dataset)
    n_val = int(n * 0.15)
    generator = torch.Generator().manual_seed(42)
    _, val_ds = random_split(dataset, [n - n_val, n_val], generator=generator)

    loader = DataLoader(val_ds, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)

    class_correct = [0] * num_classes
    class_total   = [0] * num_classes

    for x, labels, _ in loader:
        x, labels = x.to(device), labels.to(device)
        logits = model(x)
        preds  = logits.argmax(dim=1)

        for c in range(num_classes):
            mask = (labels == c)
            if mask.sum() == 0:
                continue
            class_correct[c] += (preds[mask] == c).sum().item()
            class_total[c]   += mask.sum().item()

    per_class = {c: class_correct[c] / class_total[c]
                 if class_total[c] > 0 else None
                 for c in range(num_classes)}
    return per_class


# ---------------------------------------------------------------------------
# Load training history JSONs
# ---------------------------------------------------------------------------

def load_history(out_dir, stream):
    path = Path(out_dir) / f"{stream}_history.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_stream_curves(histories, out_path):
    """Val accuracy learning curves for all streams."""
    COLORS = {"color": "#EF9F27", "texture": "#1D9E75",
              "full": "#7F77DD", "fused": "#D85A30"}

    fig, ax = plt.subplots(figsize=(8, 4))
    for stream, hist in histories.items():
        if hist is None:
            continue
        epochs = [r["epoch"] for r in hist]
        vals   = [r["val_acc"] for r in hist]
        ax.plot(epochs, vals, label=stream, color=COLORS.get(stream, "gray"), lw=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Stream comparison — validation accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def fig_ranked_bar(per_class_color, per_class_texture, class_names,
                   num_classes, out_path, top_n=40):
    """
    Ranked bar chart of (acc_C - acc_T) per class.
    Positive = color-dominant, negative = texture-dominant.
    Shows the top_n most extreme classes.
    """
    scores = {}
    for c in range(num_classes):
        ac = per_class_color.get(c)
        at = per_class_texture.get(c)
        if ac is not None and at is not None:
            scores[c] = ac - at

    # Sort by score
    sorted_items = sorted(scores.items(), key=lambda x: x[1])
    # Take top_n/2 from each end
    half = top_n // 2
    display = sorted_items[:half] + sorted_items[-half:]

    labels = [class_names[c] for c, _ in display]
    values = [s for _, s in display]
    colors = ["#1D9E75" if v < 0 else "#EF9F27" for v in values]

    fig, ax = plt.subplots(figsize=(10, max(6, len(display) * 0.28)))
    bars = ax.barh(range(len(display)), values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(display)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Accuracy difference (color acc − texture acc)")
    ax.set_title("Which food categories rely more on color vs texture?")

    color_patch   = mpatches.Patch(color="#EF9F27", label="Color-dominant")
    texture_patch = mpatches.Patch(color="#1D9E75", label="Texture-dominant")
    ax.legend(handles=[color_patch, texture_patch], loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def fig_scatter(per_class_color, per_class_texture, per_class_full,
                class_names, num_classes, out_path):
    """Scatter: acc_C on x, acc_T on y, colored by acc_F."""
    xs, ys, cs, labels = [], [], [], []
    for c in range(num_classes):
        ac = per_class_color.get(c)
        at = per_class_texture.get(c)
        af = per_class_full.get(c)
        if ac is not None and at is not None:
            xs.append(ac)
            ys.append(at)
            cs.append(af if af is not None else 0)
            labels.append(class_names[c])

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(xs, ys, c=cs, cmap="viridis", alpha=0.8, s=60,
                    vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="Full-stream accuracy")

    # Diagonal: equal reliance
    lim = max(max(xs), max(ys)) + 0.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
    ax.text(lim * 0.6, lim * 0.65, "equal", fontsize=8,
            color="gray", rotation=45)

    ax.set_xlabel("Color-stream accuracy (acc_C)")
    ax.set_ylabel("Texture-stream accuracy (acc_T)")
    ax.set_title("Color vs texture accuracy per food category")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    # Annotate top 5 most color-dominant and texture-dominant
    diffs = [x - y for x, y in zip(xs, ys)]
    idx_color   = sorted(range(len(diffs)), key=lambda i: -diffs[i])[:5]
    idx_texture = sorted(range(len(diffs)), key=lambda i:  diffs[i])[:5]
    for i in idx_color + idx_texture:
        ax.annotate(labels[i], (xs[i], ys[i]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

def grad_cam(model, x, target_class, layer):
    """Returns a heatmap for a single image tensor x."""
    gradients = []
    activations = []

    def fwd_hook(m, inp, out):
        activations.append(out)
    def bwd_hook(m, gin, gout):
        gradients.append(gout[0])

    fh = layer.register_forward_hook(fwd_hook)
    bh = layer.register_full_backward_hook(bwd_hook)

    out = model(x.unsqueeze(0))
    model.zero_grad()
    out[0, target_class].backward()

    fh.remove(); bh.remove()

    grad = gradients[0].mean(dim=[2, 3], keepdim=True)
    cam  = F.relu((grad * activations[0]).sum(dim=1).squeeze())
    cam  = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(args.class_names, args.num_classes)

    # --- Learning curves ---
    histories = {s: load_history(args.out_dir, s)
                 for s in ("color", "texture", "full", "fused")}
    if any(h is not None for h in histories.values()):
        fig_stream_curves(histories,
                          Path(args.out_dir) / "fig_stream_curves.png")

    # --- Per-class accuracy per stream ---
    streams_to_eval = []
    for stream in ("color", "texture", "full"):
        ckpt = Path(args.ckpt_dir) / f"{stream}_best.pt"
        if ckpt.exists():
            streams_to_eval.append(stream)

    if len(streams_to_eval) < 2:
        print("Need at least color and texture checkpoints for per-class analysis.")
        print("Run train.py for each stream first.")
        return

    print("Evaluating per-class accuracy...")
    per_class = {}
    for stream in streams_to_eval:
        ckpt = Path(args.ckpt_dir) / f"{stream}_best.pt"
        print(f"  {stream}...")
        per_class[stream] = eval_per_class(
            stream, str(ckpt), args.data_dir, args.num_classes,
            args.embed_dim, args.batch_size, args.num_workers, device)

    # --- Save CSV ---
    csv_path = Path(args.out_dir) / "color_vs_texture.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["class_id", "class_name"] + \
                 [f"acc_{s}" for s in streams_to_eval] + \
                 ["color_minus_texture"]
        writer.writerow(header)
        for c in range(args.num_classes):
            row = [c, class_names[c]]
            for s in streams_to_eval:
                v = per_class[s].get(c)
                row.append(f"{v:.4f}" if v is not None else "")
            ac = per_class.get("color", {}).get(c)
            at = per_class.get("texture", {}).get(c)
            score = (ac - at) if (ac is not None and at is not None) else ""
            row.append(f"{score:.4f}" if isinstance(score, float) else "")
            writer.writerow(row)
    print(f"Saved: {csv_path}")

    # --- Figures ---
    if "color" in per_class and "texture" in per_class:
        fig_ranked_bar(per_class["color"], per_class["texture"],
                       class_names, args.num_classes,
                       Path(args.out_dir) / "fig_ranked_bar.png")

        full_cls = per_class.get("full", {})
        fig_scatter(per_class["color"], per_class["texture"], full_cls,
                    class_names, args.num_classes,
                    Path(args.out_dir) / "fig_scatter.png")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()