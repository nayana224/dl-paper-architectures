"""하나의 layer/head에서 CLS query가 어느 key token을 보는지 직접 확인한다.
Single-head attention: 이 layer/head의 CLS query는 어디에 주목하는가?
Attention Rollout: 여러 head/layer의 attention 흐름을 누적하면 어떻게 되는가?
이 실습은 Rollout이 아니며 attention은 인과적 설명이 아니다.
"""
import argparse
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification

MODEL_ID = "google/vit-base-patch16-224"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=-1, help="0부터 시작하는 layer index; -1은 마지막 layer")
    parser.add_argument("--head", type=int, default=0, help="0부터 시작하는 head index")
    args = parser.parse_args()
    assets = Path(__file__).resolve().parent / "assets"
    paths = sorted(assets.glob("pet_*.jpg"))
    if not paths:
        raise SystemExit("먼저 python study/00_prepare_dataset.py를 실행하세요.")
    image = Image.open(paths[0]).convert("RGB")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    # SDPA 대신 eager를 사용해야 attention 행렬을 직접 반환받을 수 있다.
    model = ViTForImageClassification.from_pretrained(
        MODEL_ID, attn_implementation="eager",
    ).to(device).eval()
    inputs = processor(images=image, return_tensors="pt").to(device)
    layer = args.layer if args.layer >= 0 else model.config.num_hidden_layers + args.layer
    if not 0 <= layer < model.config.num_hidden_layers:
        parser.error("layer 범위를 벗어났습니다.")
    if not 0 <= args.head < model.config.num_attention_heads:
        parser.error("head 범위를 벗어났습니다.")
    with torch.inference_mode():
        result = model(**inputs, output_attentions=True, return_dict=True)
    if result.attentions is None or result.attentions[layer] is None:
        raise RuntimeError("Attention이 없습니다. eager 설정과 Transformers 버전을 확인하세요.")
    attention = result.attentions[layer].cpu()
    row = attention[0, args.head, 0, :]
    print("Attention shape:", list(attention.shape))
    print(f"선택 layer={layer}, head={args.head} (0-based)")
    print("CLS query attention vector:", row.tolist())
    print("CLS -> CLS:", row[0].item())
    print("CLS -> 196 patches:", row[1:].tolist())
    print("전체 CLS attention row 합:", row.sum().item())
    assert torch.allclose(row.sum(), torch.tensor(1.0), atol=1e-5), "Attention row 합이 1이 아닙니다."
    grid = row[1:].reshape(14, 14)
    values, indices = row[1:].topk(10)
    print("rank / patch row / patch column / raw attention weight (0-based)")
    for rank, (value, index) in enumerate(zip(values, indices), 1):
        print(rank, index.item() // 14, index.item() % 14, f"{value.item():.8f}")
    # 모델에 전달된 pixel_values를 역정규화하므로 processor의 실제 geometry와 일치한다.
    pixels = inputs["pixel_values"][0].cpu()
    if processor.do_normalize:
        mean = torch.tensor(processor.image_mean).view(3, 1, 1)
        std = torch.tensor(processor.image_std).view(3, 1, 1)
        pixels = pixels * std + mean
    if not processor.do_rescale:
        pixels = pixels / 255.0
    crop = pixels.clamp(0, 1).permute(1, 2, 0).numpy()
    upsampled = torch.nn.functional.interpolate(
        grid[None, None], size=pixels.shape[-2:], mode="bilinear", align_corners=False,
    )[0, 0]
    output = Path(__file__).resolve().parent / "outputs"
    output.mkdir(exist_ok=True)
    for overlay in [False, True]:
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        if overlay:
            ax.imshow(crop)
        data = upsampled if overlay else grid
        plot = ax.imshow(data.numpy(), cmap="magma", alpha=0.55 if overlay else 1.0,
                         vmin=grid.min().item(), vmax=grid.max().item())
        ax.set_title(f"CLS attention: layer {layer}, head {args.head}")
        fig.colorbar(plot, ax=ax, label="Raw attention weight")
        if overlay:
            ax.axis("off")
        else:
            ax.set_xlabel("Patch column (0-based)")
            ax.set_ylabel("Patch row (0-based)")
        suffix = "overlay" if overlay else "heatmap"
        fig.savefig(output / f"08_layer{layer}_head{args.head}_{suffix}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
