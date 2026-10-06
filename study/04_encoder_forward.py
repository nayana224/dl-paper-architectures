"""Pre-LN 단일 Encoder: LN -> MHA -> residual -> LN -> MLP -> residual."""
from pathlib import Path
import torch
from torch import nn
from PIL import Image
from torchvision.transforms import functional as TF


class EncoderBlock(nn.Module):
    def __init__(self, dim=128, heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        normalized = self.norm1(x)
        attended, weights = self.attention(
            normalized, normalized, normalized,
            need_weights=True, average_attn_weights=False,
        )
        x = x + attended
        x = x + self.mlp(self.norm2(x))
        return x, weights


def main():
    torch.manual_seed(0)
    assets = Path(__file__).resolve().parent / "assets"
    paths = sorted(assets.glob("pet_*.jpg"))
    if not paths:
        raise SystemExit("먼저 python study/00_prepare_dataset.py를 실행하세요.")
    image = Image.open(paths[0]).convert("RGB")
    x = TF.to_tensor(image.resize((224, 224), Image.Resampling.BILINEAR))
    patches = x.unfold(1, 16, 16).unfold(2, 16, 16)
    patches = patches.permute(1, 2, 0, 3, 4).reshape(196, 3, 16, 16)
    flat = patches.reshape(196, 768)
    D = 128
    tokens = nn.Linear(768, D)(flat).unsqueeze(0)
    cls_token = nn.Parameter(torch.zeros(1, 1, D))
    position = nn.Parameter(torch.randn(1, 197, D) * 0.02)
    x = torch.cat([cls_token, tokens], dim=1) + position
    encoder = EncoderBlock(D, heads=4).eval()
    with torch.no_grad():
        hidden, attention = encoder(x)
    print("Encoder input/output:", list(x.shape), list(hidden.shape))
    print("Attention [B, heads, query, key]:", list(attention.shape))
    print("CLS representation:", list(hidden[:, 0].shape))
    print("교육용 무작위 파라미터이며 pretrained encoder가 아닙니다.")


if __name__ == "__main__":
    main()
