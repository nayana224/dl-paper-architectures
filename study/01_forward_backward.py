"""무작위 초기화한 작은 ViT로 forward부터 weight update까지 정확히 1 step 수행한다."""
import torch
from torch import nn
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF


class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(128)
        self.attention = nn.MultiheadAttention(128, 4, batch_first=True)
        self.norm2 = nn.LayerNorm(128)
        self.mlp = nn.Sequential(nn.Linear(128, 512), nn.GELU(), nn.Linear(512, 128))

    def forward(self, x):
        # Pre-LN: LN -> MHA -> residual, LN -> MLP -> residual.
        normalized = self.norm1(x)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        x = x + attended
        return x + self.mlp(self.norm2(x))


class SmallViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Linear(16 * 16 * 3, 128)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 128))
        self.position_embedding = nn.Parameter(torch.randn(1, 197, 128) * 0.02)
        self.encoder = EncoderBlock()
        self.norm = nn.LayerNorm(128)
        self.head = nn.Linear(128, 3)

    def forward(self, image):
        print("Image:", list(image.shape))
        # [B,C,H,W] -> [B,14,14,C,16,16] -> [B,196,768].
        patches = image.unfold(2, 16, 16).unfold(3, 16, 16)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(image.shape[0], 196, 768)
        print("Patchify:", list(patches.shape))
        tokens = self.patch_embedding(patches)
        print("Patch Embedding:", list(tokens.shape))
        tokens = torch.cat([self.cls_token.expand(image.shape[0], -1, -1), tokens], dim=1)
        print("CLS 추가:", list(tokens.shape))
        tokens = tokens + self.position_embedding
        print("Position Embedding:", list(tokens.shape))
        tokens = self.encoder(tokens)
        print("Transformer Encoder:", list(tokens.shape))
        cls = self.norm(tokens[:, 0])
        print("CLS representation:", list(cls.shape))
        logits = self.head(cls)
        print("Classification Head:", list(logits.shape))
        return logits


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    # 가상 class: 0=Bird, 1=Ball, 2=Car. Pet 품종을 가상 GT로 오해하지 않도록
    # Ball 이미지를 메모리에서 직접 그린다. 파일 저장이나 dataset 다운로드는 없다.
    image = Image.new("RGB", (224, 224), "white")
    ImageDraw.Draw(image).ellipse((48, 48, 176, 176), fill="red")
    inputs = TF.to_tensor(image).unsqueeze(0).to(device)
    names = ["Bird", "Ball", "Car"]
    gt = torch.tensor([1], dtype=torch.long, device=device)
    model = SmallViT().to(device).train()  # pretrained weight를 사용하지 않는다.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    optimizer.zero_grad()
    logits = model(inputs)  # logits는 확률이 아니라 raw class score이다.
    print("logits:", logits.detach().cpu().tolist())
    # softmax는 prediction을 보기 위한 것으로 loss 입력에는 사용하지 않는다.
    probabilities = logits.detach().softmax(dim=-1)
    print("softmax:", probabilities.cpu().tolist())
    print("Prediction:", names[probabilities.argmax(dim=-1).item()])
    print("GT:", gt.item(), names[gt.item()])
    loss = loss_fn(logits, gt)  # CrossEntropyLoss는 logits와 정수 GT를 사용한다.
    print("CrossEntropyLoss:", loss.item())

    parameter = model.head.weight
    before = parameter.detach().clone()
    loss.backward()  # gradient만 계산하며 아직 weight는 바뀌지 않는다.
    print("head.weight gradient 일부:", parameter.grad[0, :5].detach().cpu().tolist())
    print("optimizer.step() 전 head.weight 일부:", before[0, :5].cpu().tolist())
    optimizer.step()  # 여기서 gradient를 이용하여 실제 weight를 변경한다.
    after = parameter.detach()
    print("optimizer.step() 후 head.weight 일부:", after[0, :5].cpu().tolist())
    print("head.weight 최대 변경량:", (after - before).abs().max().item())
    print("정확히 1 step 완료. 이 결과는 분류 성능을 보여주는 실험이 아닙니다.")


if __name__ == "__main__":
    main()
