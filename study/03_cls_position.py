"""CLS token과 Position Embedding은 학습 가능한 nn.Parameter이다."""
from pathlib import Path
import torch
from torch import nn
from PIL import Image
from torchvision.transforms import functional as TF


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
    tokens = nn.Linear(768, D)(flat)
    cls_token = nn.Parameter(torch.zeros(1, D))
    position = nn.Parameter(torch.randn(197, D) * 0.02)
    with_cls = torch.cat([cls_token, tokens], dim=0)
    positioned = with_cls + position
    print("patch tokens:", list(tokens.shape))
    print("CLS 추가:", list(with_cls.shape))
    print("Position 추가:", list(positioned.shape))
    print("CLS / Position requires_grad:", cls_token.requires_grad, position.requires_grad)
    print("이 파라미터는 학습 가능하지만 이 실습에서는 학습하지 않습니다.")


if __name__ == "__main__":
    main()
