"""unfold로 이미지가 196개의 패치로 바뀌는 과정을 관찰한다."""
from pathlib import Path
import torch
from PIL import Image
from torchvision.transforms import functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    assets = Path(__file__).resolve().parent / "assets"
    paths = sorted(assets.glob("pet_*.jpg"))
    if not paths:
        raise SystemExit("먼저 python study/00_prepare_dataset.py를 실행하세요.")
    image = Image.open(paths[0]).convert("RGB")
    x = TF.to_tensor(image.resize((224, 224), Image.Resampling.BILINEAR))
    print("Image:", list(x.shape))
    patches = x.unfold(1, 16, 16).unfold(2, 16, 16)
    print("unfold:", list(patches.shape))
    patches = patches.permute(1, 2, 0, 3, 4)
    print("permute:", list(patches.shape))
    patches = patches.reshape(196, 3, 16, 16)
    print("patches:", list(patches.shape))
    print("flatten:", list(patches.reshape(196, 768).shape))
    fig, axes = plt.subplots(14, 14, figsize=(9, 9))
    for patch, ax in zip(patches, axes.flat):
        ax.imshow(patch.permute(1, 2, 0).numpy())
        ax.axis("off")
    fig.subplots_adjust(wspace=0.08, hspace=0.08)
    output = Path(__file__).resolve().parent / "outputs"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "01_patch_grid.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
