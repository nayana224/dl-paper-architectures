"""ACT Figure 4 architecture deep dive (educational PyTorch implementation).

Run: python study/act/05_act_architecture.py --image-size 64 --chunk-size 8
No robot hardware/dataset; random tensors demonstrate shape, gradient and inference.
Differences from published ALOHA training: torchvision ResNet18 standard feature width,
small input resolution, standard PyTorch Transformer blocks, synthetic observations,
and a simplified training loop. This is NOT a released ACT checkpoint.
"""
import argparse
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


def shape(name, tensor):
    print(f"{name:34s} {tuple(tensor.shape)}")
    return tensor


def sinusoid(length, width, device):
    """1D sinusoidal positions, output [length, width]."""
    p = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    divisor = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / width)
    )
    result = torch.zeros(length, width, device=device)
    result[:, 0::2] = torch.sin(p * divisor)
    result[:, 1::2] = torch.cos(p * divisor[:result[:, 1::2].shape[1]])
    return result


class Position2D(nn.Module):
    """Learn-free 2D sinusoidal embedding for flattened [H,W] visual tokens."""
    def __init__(self, hidden):
        super().__init__()
        assert hidden % 4 == 0, "hidden must be divisible by 4"
        self.hidden = hidden

    def forward(self, h, w, device):
        yy = sinusoid(h, self.hidden // 2, device)
        xx = sinusoid(w, self.hidden // 2, device)
        return torch.cat([
            yy[:, None, :].expand(h, w, -1),
            xx[None, :, :].expand(h, w, -1)
        ], dim=-1).reshape(h * w, self.hidden)


class ActionLatentEncoder(nn.Module):
    """Posterior q(z | q_t, A_t:t+k). Uses GT actions only at training time."""
    def __init__(self, dof=14, hidden=512, latent=32, layers=4, heads=8):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        self.joints = nn.Linear(dof, hidden)
        self.actions = nn.Linear(dof, hidden)
        layer = nn.TransformerEncoderLayer(
            hidden, heads, dim_feedforward=hidden * 4,
            dropout=0.0, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)

    def forward(self, joints, gt_actions, trace=False):
        b, k, dof = gt_actions.shape
        cls = self.cls.expand(b, -1, -1)
        joint_tokens = self.joints(joints).unsqueeze(1)
        action_tokens = self.actions(gt_actions)
        tokens = torch.cat((cls, joint_tokens, action_tokens), dim=1)
        # Positions are assigned to action sequence, with zeros for CLS/joints.
        positions = torch.zeros_like(tokens)
        positions[:, 2:] = sinusoid(k, tokens.size(-1), tokens.device)
        tokens = tokens + positions
        encoded = self.encoder(tokens)
        mu, logvar = self.mu(encoded[:, 0]), self.logvar(encoded[:, 0])
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps
        if trace:
            shape("CVAE tokens [B,K+2,D]", tokens)
            shape("posterior mean [B,L]", mu)
            shape("posterior logvar [B,L]", logvar)
            shape("sampled z [B,L]", z)
        return z, mu, logvar


class MultiViewResNet(nn.Module):
    """One weight-shared ResNet18 feature extractor for every camera."""
    def __init__(self, hidden=512):
        super().__init__()
        backbone = resnet18(weights=None)
        # Keep conv feature map, not global pooling or 1000-class classifier.
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.project = nn.Conv2d(512, hidden, 1)
        self.positions = Position2D(hidden)

    def forward(self, images, trace=False):
        b, cams, rgb, h, w = images.shape
        assert rgb == 3
        x = images.reshape(b * cams, rgb, h, w)
        feat = self.project(self.features(x))
        _, d, fh, fw = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = tokens + self.positions(fh, fw, feat.device).unsqueeze(0)
        tokens = tokens.reshape(b, cams * fh * fw, d)
        if trace:
            shape("images [B,C,3,H,W]", images)
            shape("CNN maps [B*C,D,h,w]", feat)
            shape("visual tokens [B,C*h*w,D]", tokens)
        return tokens


class ACTPolicy(nn.Module):
    """ACT-inspired conditional sequence prediction with learned query positions."""
    def __init__(self, dof=14, cameras=4, chunk=8, hidden=512,
                 latent=32, heads=8, encoder_layers=4, decoder_layers=7):
        super().__init__()
        self.dof, self.cameras, self.chunk, self.latent = dof, cameras, chunk, latent
        self.latent_encoder = ActionLatentEncoder(
            dof, hidden, latent, layers=encoder_layers, heads=heads
        )
        self.visual = MultiViewResNet(hidden)
        self.joint_token = nn.Linear(dof, hidden)
        self.latent_token = nn.Linear(latent, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 4, dropout=0.0, batch_first=True, norm_first=True
        )
        dec_layer = nn.TransformerDecoderLayer(
            hidden, heads, hidden * 4, dropout=0.0, batch_first=True, norm_first=True
        )
        self.policy_encoder = nn.TransformerEncoder(
            enc_layer, encoder_layers, enable_nested_tensor=False
        )
        self.policy_decoder = nn.TransformerDecoder(dec_layer, decoder_layers)
        # Figure 4 uses fixed query positions (rather than GT action embeddings).
        self.register_buffer("query_positions", sinusoid(chunk, hidden, "cpu"))
        self.action_head = nn.Linear(hidden, dof)

    def forward(self, images, joints, gt_actions=None, trace=False):
        b = images.size(0)
        assert images.ndim == 5 and images.size(1) == self.cameras
        assert joints.shape == (b, self.dof)
        if gt_actions is not None:
            assert gt_actions.shape == (b, self.chunk, self.dof)
            z, mu, logvar = self.latent_encoder(joints, gt_actions, trace)
        else:
            # Paper's deterministic inference convention: prior mean z=0.
            z = joints.new_zeros(b, self.latent)
            mu = logvar = None
            if trace:
                shape("inference z = 0 [B,L]", z)

        visual = self.visual(images, trace)
        jt = self.joint_token(joints).unsqueeze(1)
        zt = self.latent_token(z).unsqueeze(1)
        condition = torch.cat((visual, jt, zt), dim=1)
        memory = self.policy_encoder(condition)
        query = self.query_positions.to(dtype=memory.dtype, device=memory.device)
        query = query.unsqueeze(0).expand(b, -1, -1)
        decoded = self.policy_decoder(query, memory)
        actions = self.action_head(decoded)
        if trace:
            shape("condition tokens [B,N+2,D]", condition)
            shape("policy memory [B,N+2,D]", memory)
            shape("action queries [B,K,D]", query)
            shape("decoder features [B,K,D]", decoded)
            shape("predicted actions [B,K,14]", actions)
        return {"actions": actions, "z": z, "mu": mu, "logvar": logvar}


def act_loss(prediction, target, mu, logvar, beta=10.0):
    """L1 reconstruction + beta * KL (diagonal posterior against N(0,I))."""
    reconstruction = F.l1_loss(prediction, target)
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp())
    kl = kl.sum(dim=-1).mean()
    return reconstruction + beta * kl, reconstruction, kl


def temporal_ensemble(chunks, now, decay=0.01):
    """Combine overlapping action chunks.

    chunks: list of (prediction_start_t, actions[K,DOF]).
    In ACT's weighting convention newer predictions receive higher weights.
    """
    relevant = [(t, a[now - t]) for t, a in chunks if t <= now < t + len(a)]
    if not relevant:
        raise ValueError("No chunk covers the requested execution timestep")
    # Newest first, weight exp(-decay * age).
    newest = max(t for t, _ in relevant)
    values = torch.stack([a for _, a in relevant])
    weights = values.new_tensor([math.exp(-decay * (newest - t)) for t, _ in relevant])
    weights = weights / weights.sum()
    return (weights[:, None] * values).sum(0), weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=128,
                        help="Use 512 to approach Figure 4 representation width")
    parser.add_argument("--layers", type=int, default=2,
                        help="Use 4 for encoder depth; decoder remains 7")
    parser.add_argument("--decoder-layers", type=int, default=2,
                        help="Use 7 for Figure 4 decoder depth")
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--train-step", action="store_true")
    args = parser.parse_args()
    if args.image_size < 32 or args.chunk_size < 1 or args.hidden % 8:
        parser.error("image-size >= 32, chunk-size >= 1, hidden divisible by 8")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ACTPolicy(chunk=args.chunk_size, hidden=args.hidden,
                      encoder_layers=args.layers, decoder_layers=args.decoder_layers).to(device)
    # Batch=1 is intentional: large models are memory heavy on laptops.
    images = torch.randn(1, 4, 3, args.image_size, args.image_size, device=device)
    joints = torch.randn(1, 14, device=device) * 0.1
    gt = joints.unsqueeze(1) + torch.linspace(
        0.0, 0.2, args.chunk_size, device=device)[None, :, None]
    print(f"Device: {device}; 4 synthetic cameras, 14 control axes")
    model.train()
    training = model(images, joints, gt_actions=gt, trace=True)
    total, recon, kl = act_loss(training["actions"], gt,
                                training["mu"], training["logvar"], args.beta)
    print(f"train forward: total={total.item():.5f}, L1={recon.item():.5f}, KL={kl.item():.5f}")
    if args.train_step:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        print("action head gradient norm:", model.action_head.weight.grad.norm().item())
        optimizer.step()
        print("One optimizer step completed (synthetic input, not a trained policy)")
    model.eval()
    with torch.no_grad():
        inference = model(images, joints, trace=True)
        assert inference["mu"] is None and torch.all(inference["z"] == 0)
        chunks = [(0, inference["actions"][0]),
                  (1, inference["actions"][0] + 0.1)]
        combined, weights = temporal_ensemble(chunks, now=1)
        shape("temporal ensemble action [14]", combined)
        print("overlap weights (oldest,newest):", weights.tolist())
    print("WARNING: Synthetic tensors + random weights: never send outputs to robot hardware.")


if __name__ == "__main__":
    main()
