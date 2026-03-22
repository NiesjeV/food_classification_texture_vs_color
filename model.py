"""
model.py — CNN architectures for the color vs texture study

Models:
  FoodCNN       — shared backbone, works for any single stream
  FusedFoodCNN  — dual-stream (color + texture) with late fusion
  get_model()   — factory function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv → BN → ReLU with optional downsampling."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Backbone encoder (shared across all single-stream models)
# ---------------------------------------------------------------------------

class FoodEncoder(nn.Module):
    """
    4-stage CNN encoder.
    Input:  (B, in_channels, 128, 128)
    Output: (B, embed_dim)             — L2-normalised embedding
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim

        self.stage1 = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2),          # → 64×64
        )
        self.stage2 = nn.Sequential(
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),          # → 32×32
        )
        self.stage3 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2),          # → 16×16
        )
        self.stage4 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.AdaptiveAvgPool2d(4),  # → 4×4  (4096 flat)
        )

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Single-stream classifier (CNN-C, CNN-T, CNN-F)
# ---------------------------------------------------------------------------

class FoodCNN(nn.Module):
    """
    Single-stream food classifier.
      stream='color'   → in_channels=3  (blurred RGB)
      stream='texture' → in_channels=1  (grayscale)
      stream='full'    → in_channels=3  (normal RGB, baseline)
    """

    def __init__(self, stream: str = "full", num_classes: int = 80,
                 embed_dim: int = 512):
        super().__init__()
        self.stream = stream
        in_ch = 1 if stream == "texture" else 3

        self.encoder = FoodEncoder(in_channels=in_ch, embed_dim=embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        emb = self.encoder(x)
        return self.classifier(emb)

    def get_embedding(self, x):
        """Return the pre-classification embedding (for analysis)."""
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Dual-stream fused model (CNN-C + CNN-T → fused → classifier)
# ---------------------------------------------------------------------------

class FusedFoodCNN(nn.Module):
    """
    Late-fusion model:
      Color branch  → 512-d embedding
      Texture branch → 512-d embedding
      Concat (1024-d) → Linear(512) → classifier(80)

    Optionally load pre-trained single-stream weights for each branch.
    """

    def __init__(self, num_classes: int = 80, embed_dim: int = 512):
        super().__init__()
        self.color_enc   = FoodEncoder(in_channels=3, embed_dim=embed_dim)
        self.texture_enc = FoodEncoder(in_channels=1, embed_dim=embed_dim)

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x_color, x_texture):
        e_c = self.color_enc(x_color)
        e_t = self.texture_enc(x_texture)
        fused = self.fusion(torch.cat([e_c, e_t], dim=1))
        return self.classifier(fused)

    def load_pretrained_branches(self, color_ckpt: str, texture_ckpt: str,
                                  device: str = "cpu"):
        """Bootstrap fusion model from already-trained single-stream checkpoints."""
        c_state = torch.load(color_ckpt,   map_location=device)["model"]
        t_state = torch.load(texture_ckpt, map_location=device)["model"]

        # Strip 'encoder.' prefix and load
        c_enc = {k.replace("encoder.", ""): v
                 for k, v in c_state.items() if k.startswith("encoder.")}
        t_enc = {k.replace("encoder.", ""): v
                 for k, v in t_state.items() if k.startswith("encoder.")}

        self.color_enc.load_state_dict(c_enc, strict=True)
        self.texture_enc.load_state_dict(t_enc, strict=True)
        print("Loaded pre-trained encoder weights into fusion model.")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_model(stream: str, num_classes: int = 80,
              embed_dim: int = 512) -> nn.Module:
    """
    stream: 'color' | 'texture' | 'full' | 'fused'
    Returns the appropriate model.
    """
    if stream == "fused":
        return FusedFoodCNN(num_classes=num_classes, embed_dim=embed_dim)
    return FoodCNN(stream=stream, num_classes=num_classes, embed_dim=embed_dim)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for s, ch in [("color", 3), ("texture", 1), ("full", 3)]:
        m = get_model(s)
        x = torch.randn(4, ch, 128, 128)
        out = m(x)
        print(f"[{s:8s}] output shape: {out.shape}  params: "
              f"{sum(p.numel() for p in m.parameters()) / 1e6:.2f}M")

    fused = get_model("fused")
    xc = torch.randn(4, 3, 128, 128)
    xt = torch.randn(4, 1, 128, 128)
    out = fused(xc, xt)
    print(f"[fused   ] output shape: {out.shape}  params: "
          f"{sum(p.numel() for p in fused.parameters()) / 1e6:.2f}M")