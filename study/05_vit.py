"""ViT: 패치 분할 → CLS → 자기 어텐션 → 분류 → 한 번의 가중치 갱신을 확인한다.

쿼리·키·값 연산을 직접 구현하고 첫 번째 헤드의 CLS 어텐션을 시각화한다.
무작위 초기화 모델과 합성 이미지 한 장을 사용한다.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 화면 인터페이스가 없는 환경에서도 그림을 저장한다.
import matplotlib.pyplot as plt
import torch
from torch import nn
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF


IMAGE_SIZE = 224
PATCH_SIZE = 16
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE
DIM = 128
HEADS = 4
HEAD_DIM = DIM // HEADS
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


class MultiHeadAttention(nn.Module):
    """같은 입력으로 쿼리·키·값을 만들고 여러 헤드에서 자기 어텐션을 계산한다."""

    def __init__(self):
        super().__init__()
        # 입력 토큰 하나(특징 차원 128)에서 쿼리·키·값을 한 번에 계산한다.
        self.to_qkv = nn.Linear(DIM, 3 * DIM)
        self.to_out = nn.Linear(DIM, DIM)

    def forward(self, x):
        batch, tokens, _ = x.shape

        # [B,N,3D] → [B,N,3,H,d_head] → 축 순서를 바꾼 뒤 쿼리·키·값 각각 [B,H,N,d_head].
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch, tokens, 3, HEADS, HEAD_DIM)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        scores = (q @ k.transpose(-2, -1)) * (HEAD_DIM ** -0.5)
        weights = scores.softmax(dim=-1)

        context = weights @ v
        print("Attention @ V (head별):", list(context.shape))
        context = context.transpose(1, 2).reshape(batch, tokens, DIM)
        out = self.to_out(context)

        print("Q/K/V 각각:", list(q.shape))
        print("Attention score:", list(scores.shape))
        print("Attention weights:", list(weights.shape))
        print("Head 연결 + Output projection:", list(out.shape))
        print("CLS attention row 합:", weights[0, 0, 0].sum().item())

        return out, weights


class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(DIM)
        self.attention = MultiHeadAttention()
        self.norm2 = nn.LayerNorm(DIM)
        self.mlp = nn.Sequential(
            nn.Linear(DIM, DIM * 4),
            nn.GELU(),
            nn.Linear(DIM * 4, DIM),
        )

    def forward(self, x):
        # 사전 정규화: 층 정규화 → 어텐션 → 잔차 연결, 이어서 정규화 → 다층 퍼셉트론 → 잔차 연결.
        attended, weights = self.attention(self.norm1(x))
        x = x + attended
        x = x + self.mlp(self.norm2(x))
        return x, weights


class SmallViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Linear(PATCH_SIZE * PATCH_SIZE * 3, DIM)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, DIM))
        self.position_embedding = nn.Parameter(
            torch.randn(1, GRID_SIZE * GRID_SIZE + 1, DIM) * 0.02
        )
        self.encoder = EncoderBlock()
        self.norm = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, 3)

    def forward(self, image):
        print("Image:", list(image.shape))
        # [B,3,224,224] → [B,196,768]; 16×16 크기의 색상 패치 14×14개다.
        patches = image.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(
            3, PATCH_SIZE, PATCH_SIZE
        )
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(image.shape[0], GRID_SIZE**2, 3 * PATCH_SIZE**2)
        print("Patchify:", list(patches.shape))

        tokens = self.patch_embedding(patches)
        print("Patch Embedding:", list(tokens.shape))
        tokens = torch.cat(
            [self.cls_token.expand(image.shape[0], -1, -1), tokens], dim=1
        )
        print("CLS 추가:", list(tokens.shape))
        tokens = tokens + self.position_embedding
        print("Position Embedding:", list(tokens.shape))

        tokens, attention_weights = self.encoder(tokens)
        print("Transformer Encoder:", list(tokens.shape))
        cls = self.norm(tokens[:, 0])  # 맨 앞의 CLS 토큰만 사용한다.
        print("CLS representation:", list(cls.shape))
        logits = self.head(cls)
        print("Classification Head:", list(logits.shape))
        return logits, attention_weights


def save_visualizations(image_tensor, weights):
    """원본 이미지의 패치 격자와 첫 번째 헤드의 CLS→패치 어텐션을 저장한다."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image = image_tensor[0].detach().cpu().permute(1, 2, 0).numpy()

    # 원본 이미지 위에 14×14개 패치의 경계를 그려 분할 위치를 확인한다.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    ax.set_xticks(range(0, IMAGE_SIZE + 1, PATCH_SIZE))
    ax.set_yticks(range(0, IMAGE_SIZE + 1, PATCH_SIZE))
    ax.grid(color="cyan", linewidth=0.8)
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    ax.set_title("16x16 Patch Grid (14x14 = 196 patches)")
    fig.savefig(OUTPUT_DIR / "01_patch_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 어텐션 형태는 [배치,헤드,토큰,토큰]이다. 첫 헤드의 CLS 쿼리(0번)가
    # 이미지 패치의 키(1~196번)를 얼마나 참조하는지 확인한다.
    cls_row = weights[0, 0, 0].detach().cpu()
    grid = cls_row[1:].reshape(GRID_SIZE, GRID_SIZE)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    heatmap = axes[0].imshow(grid.numpy(), cmap="magma")
    axes[0].set_title("Head 0: CLS -> patches (14x14)")
    axes[0].set_xlabel("Patch column")
    axes[0].set_ylabel("Patch row")
    fig.colorbar(heatmap, ax=axes[0], label="Raw attention weight")
    axes[1].imshow(image)
    axes[1].imshow(
        torch.nn.functional.interpolate(
            grid[None, None], size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="nearest",
        )[0, 0].numpy(),
        cmap="magma",
        alpha=0.55,
        vmin=grid.min().item(),
        vmax=grid.max().item(),
    )
    axes[1].set_title("CLS attention overlay (random model)")
    axes[1].axis("off")
    fig.savefig(OUTPUT_DIR / "01_attention_heatmap.png", dpi=150)
    plt.close(fig)
    print("시각화 저장:", OUTPUT_DIR / "01_patch_grid.png")
    print("시각화 저장:", OUTPUT_DIR / "01_attention_heatmap.png")
    print("주의: 무작위 초기화 모델의 attention은 학습된 의미/인과적 설명이 아닙니다.")


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 가상 클래스: 0=새, 1=공, 2=자동차. 정답이 공인 이미지를 직접 만든다.
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white")
    ImageDraw.Draw(image).ellipse((48, 48, 176, 176), fill="red")
    inputs = TF.to_tensor(image).unsqueeze(0).to(device)
    names = ["Bird", "Ball", "Car"]
    gt = torch.tensor([1], dtype=torch.long, device=device)

    model = SmallViT().to(device).train()  # 사전학습 가중치를 사용하지 않는다.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    optimizer.zero_grad()
    logits, attention = model(inputs)  # 순전파
    save_visualizations(inputs, attention)

    print("logits:", logits.detach().cpu().tolist())  # 확률이 아닌 원시 예측 점수다.
    probabilities = logits.detach().softmax(dim=-1)  # 출력값을 확률로 해석하기 위한 계산이다.
    print("softmax:", probabilities.cpu().tolist())
    print("Prediction:", names[probabilities.argmax(dim=-1).item()])
    print("GT:", gt.item(), names[gt.item()])

    loss = loss_fn(logits, gt)  # 교차 엔트로피 손실에는 소프트맥스 적용 전 점수를 입력한다.
    print("CrossEntropyLoss:", loss.item())

    parameter = model.head.weight
    before = parameter.detach().clone()
    loss.backward()  # 기울기만 계산하며 이 시점에는 가중치가 바뀌지 않는다.
    print("head.weight gradient 일부:", parameter.grad[0, :5].detach().cpu().tolist())
    print("optimizer.step() 전 head.weight 일부:", before[0, :5].cpu().tolist())
    optimizer.step()  # 계산한 기울기로 실제 모델 파라미터를 갱신한다.
    after = parameter.detach()
    print("optimizer.step() 후 head.weight 일부:", after[0, :5].cpu().tolist())
    print("head.weight 최대 변경량:", (after - before).abs().max().item())
    print("정확히 1 step 완료. 이 결과는 분류 성능을 보여주는 실험이 아닙니다.")


if __name__ == "__main__":
    main()
