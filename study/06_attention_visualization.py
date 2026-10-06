"""torchvision의 head 평균 + residual을 레이어별로 누적한 Attention Rollout."""
from pathlib import Path
import torch
from PIL import Image
from torchvision.models import vit_b_16, ViT_B_16_Weights
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    assets = Path(__file__).resolve().parent / "assets"
    paths = sorted(assets.glob("pet_*.jpg"))
    if not paths:
        raise SystemExit("먼저 python study/00_prepare_dataset.py를 실행하세요.")
    image = Image.open(paths[0]).convert("RGB")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    model = vit_b_16(weights=weights).to(device).eval()
    inputs = preprocess(image).unsqueeze(0).to(device)
    attentions = []

    def request_weights(module, args, kwargs):
        # torchvision encoder는 기본적으로 need_weights=False를 사용한다.
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return args, kwargs

    def collect_weights(module, args, result):
        attentions.append(result[1].detach().cpu())

    handles = []
    for block in model.encoder.layers:
        handles.append(block.self_attention.register_forward_pre_hook(request_weights, with_kwargs=True))
        handles.append(block.self_attention.register_forward_hook(collect_weights))
    try:
        with torch.no_grad():
            model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    print("레이어 수 / Attention shape:", len(attentions), list(attentions[0].shape))
    rollout = torch.eye(197)
    for attention in attentions:
        matrix = attention[0].mean(dim=0) + torch.eye(197)
        matrix = matrix / matrix.sum(dim=-1, keepdim=True)
        rollout = matrix @ rollout
    grid = rollout[0, 1:].reshape(14, 14)
    upsampled = torch.nn.functional.interpolate(
        grid[None, None], size=inputs.shape[-2:], mode="bilinear", align_corners=False,
    )[0, 0]
    # 실제 입력 tensor를 역정규화하여 resize + center crop 좌표와 정확히 맞춘다.
    mean = torch.tensor(preprocess.mean).view(3, 1, 1)
    std = torch.tensor(preprocess.std).view(3, 1, 1)
    crop = (inputs[0].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    fig, axes = plt.subplots(1, 4, figsize=(17, 4), constrained_layout=True)
    axes[0].imshow(crop)
    axes[0].set_title("Actual model input crop")
    for ax, data, title in zip(axes[1:3], [grid, upsampled], ["CLS rollout 14x14", "Upsampled rollout"]):
        plot = ax.imshow(data.numpy(), cmap="magma", vmin=grid.min().item(), vmax=grid.max().item())
        ax.set_title(title)
        fig.colorbar(plot, ax=ax, label="Relative rollout score")
    axes[3].imshow(crop)
    plot = axes[3].imshow(upsampled.numpy(), cmap="magma", alpha=0.55,
                          vmin=grid.min().item(), vmax=grid.max().item())
    axes[3].set_title("Overlay")
    fig.colorbar(plot, ax=axes[3], label="Relative rollout score")
    for ax in axes:
        ax.axis("off")
    # Rollout은 attention 경로의 상대적 점수이며 인과적 설명이 아니다.
    output = Path(__file__).resolve().parent / "outputs"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "06_attention_rollout.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
