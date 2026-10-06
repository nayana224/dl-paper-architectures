"""torchvision ViT-B/16으로 선택한 모든 이미지의 ImageNet Top-5를 확인한다."""
from pathlib import Path
import torch
from PIL import Image
from torchvision.models import vit_b_16, ViT_B_16_Weights


def main():
    paths = sorted((Path(__file__).resolve().parent / "assets").glob("pet_*.jpg"))
    if not paths:
        raise SystemExit("먼저 python study/00_prepare_dataset.py를 실행하세요.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights).to(device).eval()
    preprocess = weights.transforms()
    # Oxford-IIIT Pet 품종 정답과 ImageNet 클래스는 다른 label space이다.
    print("Pet label과 ImageNet label은 다릅니다. 품종 정확도로 해석하지 마세요.")
    for path in paths:
        image = Image.open(path).convert("RGB")
        inputs = preprocess(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            probabilities = model(inputs).softmax(dim=-1)[0]
        print(f"\n{path.name}")
        values, indices = probabilities.topk(5)
        for rank, (value, index) in enumerate(zip(values, indices), 1):
            print(f"{rank}: {weights.meta['categories'][index.item()]} {value.item():.4f}")


if __name__ == "__main__":
    main()
