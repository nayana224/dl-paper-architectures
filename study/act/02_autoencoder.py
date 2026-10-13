"""AE 입문: 가상 14-DoF 관절 궤적을 8차원으로 압축하고 복원한다."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

OUT = Path(__file__).resolve().parents[2] / "outputs"
STEPS, DOF, LATENT = 10, 14, 8


def make_trajectories(n=512):
    """로봇 실측치가 아닌 매끄러운 학습용 가상 관절 시퀀스 [N,10,14]."""
    g = torch.Generator().manual_seed(17)
    t = torch.linspace(0, 1, STEPS)[None, :, None]
    start = torch.randn(n, 1, DOF, generator=g) * 0.3
    velocity = torch.randn(n, 1, DOF, generator=g) * 0.5
    curve = torch.randn(n, 1, DOF, generator=g) * 0.2
    return start + velocity * t + curve * torch.sin(torch.pi * t)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(STEPS * DOF, 64), nn.ReLU(),
            nn.Linear(64, LATENT),
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT, 64), nn.ReLU(),
            nn.Linear(64, STEPS * DOF), nn.Unflatten(1, (STEPS, DOF)),
        )

    def forward(self, actions):
        z = self.encoder(actions)
        reconstructed = self.decoder(z)
        return reconstructed, z


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=300)
    args = p.parse_args()
    if args.epochs < 1:
        p.error("--epochs must be >= 1")
    torch.manual_seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = make_trajectories().to(device)
    model = Autoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    losses = []
    print("Input/GT:", tuple(train.shape), "| device:", device)
    for epoch in range(args.epochs):
        model.train()
        prediction, z = model(train)
        loss = nn.functional.mse_loss(prediction, train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch == 0 or (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d}: reconstruction MSE={loss.item():.6f}")
    model.eval()
    with torch.no_grad():
        reconstructed, z = model(train[:1])
    print("Latent z:", tuple(z.shape), "| Reconstructed:", tuple(reconstructed.shape))
    print("GT is the input itself; no separate action labels.")
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for j in range(3):
        ax[0].plot(train[0, :, j].cpu(), marker="o", label=f"GT joint {j}")
        ax[0].plot(reconstructed[0, :, j].cpu(), linestyle="--",
                   label=f"AE joint {j}")
    ax[0].set(xlabel="Timestep", ylabel="Joint value (synthetic)",
              title="Original vs AE reconstruction")
    ax[0].legend(fontsize=7, ncol=2)
    ax[1].plot(losses)
    ax[1].set(xlabel="Epoch", ylabel="MSE", title="AE reconstruction loss")
    target = OUT / "08_act_ae.png"
    fig.savefig(target, dpi=150)
    plt.close(fig)
    print("Saved:", target)


if __name__ == "__main__":
    main()
