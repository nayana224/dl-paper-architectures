"""VAE 입문: AE와 달리 mu/logvar 분포를 학습하고 z를 샘플링한다."""
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
    """AE와 동일한 시드/생성식으로 비교 가능한 합성 데이터."""
    g = torch.Generator().manual_seed(17)
    t = torch.linspace(0, 1, STEPS)[None, :, None]
    start = torch.randn(n, 1, DOF, generator=g) * 0.3
    velocity = torch.randn(n, 1, DOF, generator=g) * 0.5
    curve = torch.randn(n, 1, DOF, generator=g) * 0.2
    return start + velocity * t + curve * torch.sin(torch.pi * t)


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(STEPS*DOF, 64),
                                     nn.ReLU())
        self.mu = nn.Linear(64, LATENT)
        self.logvar = nn.Linear(64, LATENT)
        self.decoder = nn.Sequential(
            nn.Linear(LATENT, 64), nn.ReLU(),
            nn.Linear(64, STEPS*DOF), nn.Unflatten(1, (STEPS, DOF)),
        )

    def encode(self, actions):
        h = self.encoder(actions)
        return self.mu(h), self.logvar(h)  # logvar = log(variance)

    @staticmethod
    def reparameterize(mu, logvar):
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def decode(self, z):
        return self.decoder(z)

    def forward(self, actions):
        mu, logvar = self.encode(actions)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--beta", type=float, default=0.01,
                   help="KL loss weight: raise to regularize z more strongly")
    args = p.parse_args()
    if args.epochs < 1 or args.beta < 0:
        p.error("epochs >= 1 and beta >= 0 are required")
    torch.manual_seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = make_trajectories().to(device)
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    histories = {"total": [], "recon": [], "kl": []}
    print("Input/GT:", tuple(train.shape), "| device:", device)
    for epoch in range(args.epochs):
        model.train()
        prediction, mu, logvar, z = model(train)
        recon = nn.functional.mse_loss(prediction, train)
        # Diagonal Gaussian KL(q(z|x) || N(0,I)), 평균 batch당 합산 latent dim.
        kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1).mean()
        total = recon + args.beta * kl
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        for key, val in (("total", total), ("recon", recon), ("kl", kl)):
            histories[key].append(val.item())
        if epoch == 0 or (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d}: total={total.item():.6f} "
                  f"recon={recon.item():.6f} KL={kl.item():.6f}")
    model.eval()
    with torch.no_grad():
        # 무작위 z 대신 mu로 재구성: 노이즈가 없는 비교용 결과.
        mu, logvar = model.encode(train[:1])
        reconstructed = model.decode(mu)
        random_sample = model.decode(torch.randn(1, LATENT, device=device))
    print("mu / logvar:", tuple(mu.shape), tuple(logvar.shape))
    print("Reconstructed / random sample:", tuple(reconstructed.shape),
          tuple(random_sample.shape))
    print("KL encourages latent distributions near N(0,I); valid robot actions are NOT guaranteed.")
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for j in range(3):
        ax[0].plot(train[0, :, j].cpu(), label=f"GT joint {j}")
        ax[0].plot(reconstructed[0, :, j].cpu(), linestyle="--",
                   label=f"VAE joint {j}")
    ax[0].set(xlabel="Timestep", ylabel="Joint value (synthetic)",
              title="Original vs VAE mean reconstruction")
    ax[0].legend(fontsize=7, ncol=2)
    for key, values in histories.items():
        ax[1].plot(values, label=key)
    ax[1].set(xlabel="Epoch", ylabel="Loss", title=f"VAE loss (beta={args.beta})")
    ax[1].legend()
    target = OUT / "08_act_vae.png"
    fig.savefig(target, dpi=150)
    plt.close(fig)
    print("Saved:", target)


if __name__ == "__main__":
    main()
