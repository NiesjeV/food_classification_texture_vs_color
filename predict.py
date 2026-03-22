"""
predict.py — Generate Kaggle submission CSV

Usage:
  python predict.py --stream full    --ckpt checkpoints/full_best.pt    --data_dir ./data
  python predict.py --stream color   --ckpt checkpoints/color_best.pt   --data_dir ./data
  python predict.py --stream texture --ckpt checkpoints/texture_best.pt --data_dir ./data
  python predict.py --stream fused   --ckpt checkpoints/fused_best.pt   --data_dir ./data
  
  # Ensemble all three single-stream models:
  python predict.py --ensemble \
    --color_ckpt   checkpoints/color_best.pt \
    --texture_ckpt checkpoints/texture_best.pt \
    --full_ckpt    checkpoints/full_best.pt \
    --data_dir ./data
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import FoodDataset, MultiStreamFoodDataset, get_transform
from model import get_model


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stream",       default="full",
                   choices=["color", "texture", "full", "fused"])
    p.add_argument("--ckpt",         default=None, help="Checkpoint for single stream")
    p.add_argument("--data_dir",     default="./data")
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--embed_dim",    type=int, default=512)
    p.add_argument("--num_classes",  type=int, default=80)
    p.add_argument("--out_dir",      default="./outputs")
    # Ensemble mode
    p.add_argument("--ensemble",     action="store_true")
    p.add_argument("--color_ckpt",   default=None)
    p.add_argument("--texture_ckpt", default=None)
    p.add_argument("--full_ckpt",    default=None)
    # TTA (test-time augmentation)
    p.add_argument("--tta",          action="store_true",
                   help="Average predictions over 5 random crops")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------

def load_model(stream, ckpt_path, num_classes, embed_dim, device):
    model = get_model(stream, num_classes, embed_dim).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"Loaded {stream} model from {ckpt_path}")
    return model


# ---------------------------------------------------------------------------
# Single-stream inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_single(model, stream, data_dir, batch_size, num_workers,
                   device, tta=False, num_classes=80):
    test_ds = FoodDataset(data_dir, split="test", stream=stream, augment=False)
    loader  = DataLoader(test_ds, batch_size=batch_size,
                         shuffle=False, num_workers=num_workers)

    all_names, all_probs = [], []

    for x, _, names in loader:
        x = x.to(device)

        if tta:
            # 5-crop TTA: original + 4 corners → average softmax
            probs = F.softmax(model(x), dim=1)
            for _ in range(4):
                aug = get_transform(stream, augment=True)
                # re-apply augmentation from scratch is complex;
                # simple TTA: horizontal flip
                x_flip = torch.flip(x, dims=[-1])
                probs = probs + F.softmax(model(x_flip), dim=1)
            probs /= 5
        else:
            probs = F.softmax(model(x), dim=1)

        all_names.extend(names)
        all_probs.append(probs.cpu())

    return all_names, torch.cat(all_probs, dim=0)


# ---------------------------------------------------------------------------
# Fused model inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_fused(model, data_dir, batch_size, num_workers, device):
    test_ds = MultiStreamFoodDataset(data_dir, split="test", augment=False)
    loader  = DataLoader(test_ds, batch_size=batch_size,
                         shuffle=False, num_workers=num_workers)

    all_names, all_probs = [], []

    for x_c, x_t, x_f, _, names in loader:
        x_c, x_t = x_c.to(device), x_t.to(device)
        probs = F.softmax(model(x_c, x_t), dim=1)
        all_names.extend(names)
        all_probs.append(probs.cpu())

    return all_names, torch.cat(all_probs, dim=0)


# ---------------------------------------------------------------------------
# Write submission CSV
# ---------------------------------------------------------------------------

def write_submission(names, probs, out_path):
    labels = probs.argmax(dim=1).tolist()
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img_name", "label"])
        for name, label in zip(names, labels):
            writer.writerow([name, label])
    print(f"Submission saved to {out_path}  ({len(labels)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.ensemble:
        # Load all three single-stream models
        assert all([args.color_ckpt, args.texture_ckpt, args.full_ckpt]), \
            "Ensemble requires --color_ckpt, --texture_ckpt, and --full_ckpt"

        models = {
            "color":   load_model("color",   args.color_ckpt,   args.num_classes, args.embed_dim, device),
            "texture": load_model("texture", args.texture_ckpt, args.num_classes, args.embed_dim, device),
            "full":    load_model("full",    args.full_ckpt,    args.num_classes, args.embed_dim, device),
        }

        # Collect probs from each stream
        ensemble_probs = None
        names = None
        for stream, model in models.items():
            n, p = predict_single(model, stream, args.data_dir,
                                  args.batch_size, args.num_workers,
                                  device, tta=args.tta,
                                  num_classes=args.num_classes)
            if ensemble_probs is None:
                ensemble_probs = p
                names = n
            else:
                ensemble_probs += p

        ensemble_probs /= len(models)
        out_path = Path(args.out_dir) / "submission_ensemble.csv"
        write_submission(names, ensemble_probs, out_path)

    elif args.stream == "fused":
        model = load_model("fused", args.ckpt, args.num_classes,
                           args.embed_dim, device)
        names, probs = predict_fused(model, args.data_dir,
                                     args.batch_size, args.num_workers, device)
        out_path = Path(args.out_dir) / "submission_fused.csv"
        write_submission(names, probs, out_path)

    else:
        model = load_model(args.stream, args.ckpt, args.num_classes,
                           args.embed_dim, device)
        names, probs = predict_single(model, args.stream, args.data_dir,
                                      args.batch_size, args.num_workers,
                                      device, tta=args.tta,
                                      num_classes=args.num_classes)
        out_path = Path(args.out_dir) / f"submission_{args.stream}.csv"
        write_submission(names, probs, out_path)


if __name__ == "__main__":
    main()
