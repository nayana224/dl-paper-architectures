"""ResNet (2015): H(x)=F(x)+shortcut, Forward/Loss/Backward 학습용 축소판."""
import torch
from torch import nn
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF

class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                               nn.BatchNorm2d(cout), nn.ReLU(),
                               nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
                               nn.BatchNorm2d(cout))
        # 크기가 바뀌면 1x1 Projection으로 Shortcut 크기를 맞춘다.
        self.shortcut = (nn.Identity() if cin == cout and stride == 1 else
                         nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                       nn.BatchNorm2d(cout)))
    def forward(self, x):
        f, s = self.f(x), self.shortcut(x)
        print("F(x), shortcut:", tuple(f.shape), tuple(s.shape))
        return torch.relu(f + s)  # ResNet v1: Add 이후 ReLU

class MiniResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1)
        self.block1 = ResidualBlock(16, 16)
        self.block2 = ResidualBlock(16, 32, stride=2)
        self.head = nn.Linear(32, 3)
    def forward(self, x):
        print("Image:", tuple(x.shape))
        x = torch.relu(self.stem(x))
        x = self.block1(x)
        x = self.block2(x)
        print("Feature:", tuple(x.shape))
        x = x.mean(dim=(2, 3))  # Global Average Pooling
        print("Pooled:", tuple(x.shape))
        return self.head(x)

def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(image).rectangle((12, 12, 52, 52), fill="blue")
    x = TF.to_tensor(image).unsqueeze(0).to(device)
    gt = torch.tensor([1], device=device)  # 가상 GT, 분류 성능 평가 아님
    model = MiniResNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, gt)
    before = model.head.weight.detach().clone()
    loss.backward()
    print("Logits:", logits.detach().cpu().tolist(), "GT:", gt.tolist(), "Loss:", loss.item())
    print("Residual gradient norm:", model.block1.f[3].weight.grad.norm().item())
    optimizer.step()
    print("Head max weight update:", (model.head.weight-before).abs().max().item())

if __name__ == "__main__":
    main()
