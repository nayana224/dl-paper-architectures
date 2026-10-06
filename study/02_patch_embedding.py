"""패치의 픽셀 벡터를 학습 가능한 Linear로 D차원에 투영한다."""
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
    projection = nn.Linear(16 * 16 * 3, D)
    tokens = projection(flat)
    print("raw patch dimension P²C =", flat.shape[-1])
    print("교육용 hidden size D =", D)
    print("Patch Embedding:", list(flat.shape), "->", list(tokens.shape))
    # ViT-B/16의 D=768은 픽셀 차원 768과 숫자만 같고 의미는 다르다.
    print("ViT-B/16: raw patch dimension=768, hidden size D=768 (서로 다른 개념)")
    print("현재 projection은 학습하지 않은 무작위 초기화입니다.")


if __name__ == "__main__":
    main()
