"""
train.py — Training loop for all streams

Usage examples:
  python train.py --stream full   --data_dir ./data --epochs 40
  python train.py --stream color  --data_dir ./data --epochs 40
  python train.py --stream texture --data_dir ./data --epochs 40
  python train.py --stream fused  --data_dir ./data --epochs 40 \
                  --color_ckpt checkpoints/color_best.pt \
                  --texture_ckpt checkpoints/texture_best.pt
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import make_loaders, make_multistream_loaders
from model import get_model

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stream",       default="full",
                   choices=["color", "texture", "full", "fused"])
    p.add_argument("--data_dir",     default="./data")
    p.add_argument("--epochs",       type=int, default=40)
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--embed_dim",    type=int, default=512)
    p.add_argument("--num_classes",  type=int, default=80)
    p.add_argument("--ckpt_dir",     default="./checkpoints")
    p.add_argument("--out_dir",      default="./outputs")
    p.add_argument("--val_frac",     type=float, default=0.15)
    # For fused model: optionally bootstrap from single-stream checkpoints
    p.add_argument("--color_ckpt",   default=None)
    p.add_argument("--texture_ckpt", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def per_class_accuracy(logits, labels, num_classes):
    """Returns dict {class_id: acc} for classes present in this batch."""
    preds = logits.argmax(dim=1)
    correct = {}
    total   = {}
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() == 0:
            continue
        correct[c] = (preds[mask] == c).sum().item()
        total[c]   = mask.sum().item()
    return correct, total


# ---------------------------------------------------------------------------
# One epoch of training
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device, stream, num_classes):
    model.train()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    class_correct = {c: 0 for c in range(num_classes)}
    class_total   = {c: 0 for c in range(num_classes)}

    for batch in loader:
        if stream == "fused":
            x_c, x_t, x_f, labels, _ = batch
            x_c, x_t = x_c.to(device), x_t.to(device)
            labels = labels.to(device)
            logits = model(x_c, x_t)
        else:
            x, labels, _ = batch
            x, labels = x.to(device), labels.to(device)
            logits = model(x)

        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_acc  += accuracy(logits, labels)
        n_batches  += 1

        c_corr, c_tot = per_class_accuracy(logits, labels, num_classes)
        for c in c_corr:
            class_correct[c] += c_corr[c]
            class_total[c]   += c_tot[c]

    per_class = {c: class_correct[c] / class_total[c]
                 for c in class_total if class_total[c] > 0}
    return total_loss / n_batches, total_acc / n_batches, per_class


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def val_epoch(model, loader, criterion, device, stream, num_classes):
    model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    class_correct = {c: 0 for c in range(num_classes)}
    class_total   = {c: 0 for c in range(num_classes)}

    for batch in loader:
        if stream == "fused":
            x_c, x_t, x_f, labels, _ = batch
            x_c, x_t = x_c.to(device), x_t.to(device)
            labels = labels.to(device)
            logits = model(x_c, x_t)
        else:
            x, labels, _ = batch
            x, labels = x.to(device), labels.to(device)
            logits = model(x)

        loss = criterion(logits, labels)
        total_loss += loss.item()
        total_acc  += accuracy(logits, labels)
        n_batches  += 1

        c_corr, c_tot = per_class_accuracy(logits, labels, num_classes)
        for c in c_corr:
            class_correct[c] += c_corr[c]
            class_total[c]   += c_tot[c]

    per_class = {c: class_correct[c] / class_total[c]
                 for c in class_total if class_total[c] > 0}
    return total_loss / n_batches, total_acc / n_batches, per_class


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Stream: {args.stream}")

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Data
    if args.stream == "fused":
        train_loader, val_loader = make_multistream_loaders(
            args.data_dir, batch_size=args.batch_size,
            num_workers=args.num_workers, val_frac=args.val_frac)
    else:
        train_loader, val_loader = make_loaders(
            args.data_dir, stream=args.stream,
            batch_size=args.batch_size, num_workers=args.num_workers,
            val_frac=args.val_frac)

    # Model
    model = get_model(args.stream, args.num_classes, args.embed_dim).to(device)

    if args.stream == "fused" and args.color_ckpt and args.texture_ckpt:
        model.load_pretrained_branches(args.color_ckpt, args.texture_ckpt, device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {total_params:.2f}M")

    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc, tr_cls = train_epoch(
            model, train_loader, optimizer, criterion,
            device, args.stream, args.num_classes)

        val_loss, val_acc, val_cls = val_epoch(
            model, val_loader, criterion,
            device, args.stream, args.num_classes)

        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"{elapsed:.1f}s")

        record = {
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": val_loss,  "val_acc": val_acc,
            "per_class_val": val_cls,
        }
        history.append(record)

        # Checkpoint: best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = Path(args.ckpt_dir) / f"{args.stream}_best.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_acc": val_acc,
                "args": vars(args),
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint (val_acc={val_acc:.4f})")

    # Save training history
    hist_path = Path(args.out_dir) / f"{args.stream}_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
    print(f"History saved to {hist_path}")


if __name__ == "__main__":
    main()