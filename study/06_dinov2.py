"""DINOv2: 사진 한 장에서 Pretrained Patch Feature를 추출·시각화하는 실습.

DINOv2는 SAM처럼 Segmentation Mask를 직접 예측하지 않는다.
PCA 색/유사도 지도는 learned feature를 관찰하기 위한 것이며 GT Mask가 아니다.
"""
import argparse
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


MODEL_ID = "facebook/dinov2-small"
ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "assets" / "dinov2_dog.jpg"
OUTPUTS = ROOT / "outputs"
# PyTorch Hub에서 제공하는 실제 강아지 사진. 최초 실행에만 다운로드한다.
SAMPLE_URL = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"


def load_image(path_arg):
    path = Path(path_arg).expanduser() if path_arg else SAMPLE
    if not path.is_file():
        if path_arg:
            raise FileNotFoundError(f"이미지가 없습니다: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        print("샘플 사진 다운로드:", SAMPLE_URL)
        try:
            urlretrieve(SAMPLE_URL, path)
        except Exception as exc:
            raise RuntimeError("사진 다운로드 실패. --image 경로로 사진을 지정하세요.") from exc
    return Image.open(path).convert("RGB"), path


def normalize01(values):
    values = np.asarray(values, dtype=np.float32)
    lo, hi = float(values.min()), float(values.max())
    return (values - lo) / (hi - lo + 1e-8)


def main():
    parser = argparse.ArgumentParser(description="DINOv2 patch feature PCA / cosine similarity")
    parser.add_argument("--image", type=str, default=None, help="직접 지정하는 RGB 사진 경로")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image, path = load_image(args.image)
    print("Device:", device, "| Image:", path, "| Original size:", image.size)
    print("Checkpoint:", MODEL_ID)

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixels = inputs["pixel_values"]
    print("Pixel values:", list(pixels.shape))

    with torch.inference_mode():
        hidden = model(**inputs).last_hidden_state  # [B, 1+patch_count, D]
    print("Last hidden state (CLS + patches):", list(hidden.shape))

    patch_size = model.config.patch_size
    grid_h = pixels.shape[-2] // patch_size
    grid_w = pixels.shape[-1] // patch_size
    count = grid_h * grid_w
    # dinov2-small은 Register Token이 없으며 CLS가 index 0이다.
    if hidden.shape[1] != count + 1:
        raise RuntimeError("CLS + patch token 수가 예상과 다릅니다. 모델 구성 확인이 필요합니다.")

    cls = hidden[:, 0]
    patch_tokens = hidden[:, 1:]
    print("CLS feature:", list(cls.shape))
    print("Patch features:", list(patch_tokens.shape))
    print("Patch grid:", [grid_h, grid_w, hidden.shape[-1]])

    # 공간적으로 배열한 Patch Feature [N,D]를 PCA로 3차원에 투영.
    features = patch_tokens[0].float().cpu()
    features = features - features.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(features, full_matrices=False)
    pca = (features @ vh[:3].T).numpy().reshape(grid_h, grid_w, 3)
    pca_rgb = np.stack([normalize01(pca[:, :, c]) for c in range(3)], axis=-1)

    # 가운데 patch를 기준으로 코사인 유사도를 계산한다.
    # 가운데 patch가 물체가 아닌 경우 --image와 함께 결과를 다시 확인해야 한다.
    raw = patch_tokens[0].float().cpu()
    unit = torch.nn.functional.normalize(raw, dim=-1)
    center_index = (grid_h // 2) * grid_w + grid_w // 2
    similarity = (unit @ unit[center_index]).numpy().reshape(grid_h, grid_w)
    print("Center patch index:", center_index)
    print("Cosine similarity range:", float(similarity.min()), float(similarity.max()))

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    # Processor가 실제 모델에 제공한 정규화/크롭 이미지를 원본 사진 대신 참고용으로 그리지 않는다.
    # PCA와 유사도 맵은 processor가 만든 grid이며 원본 사진의 비율과 달라질 수 있다.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_title("Input photo")
    axes[1].imshow(pca_rgb)
    axes[1].set_title("Patch features: PCA → RGB")
    heat = axes[2].imshow(similarity, cmap="magma", vmin=-1, vmax=1)
    axes[2].scatter([grid_w // 2], [grid_h // 2], c="cyan", marker="x", s=65)
    axes[2].set_title("Cosine similarity to center patch")
    fig.colorbar(heat, ax=axes[2], label="cosine similarity")
    for ax in axes:
        ax.axis("off")
    out = OUTPUTS / "06_dinov2_features.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("Saved:", out)
    print("PCA는 색상에 클래스 의미가 없고, similarity는 Segmentation Mask가 아닙니다.")


if __name__ == "__main__":
    main()
