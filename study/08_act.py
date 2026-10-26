"""ACT 그림 4: 조건부 변분 오토인코더 → 다중 시점 ResNet18 → Transformer → 행동 묶음.

논문 설정: RGB 카메라 4대(480×640), 관절·행동 14차원, 행동 묶음 100개, 은닉 차원 512.\n한 파일에서 학습, 추론(z=0), L1·KL 손실과 시간적 앙상블을 확인한다.
합성 입력/무작위 가중치이며 실제 ALOHA 정책을 재현하지 않는다.
"""
import argparse
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


def shape(name, tensor):
    print(f"{name:34s} {tuple(tensor.shape)}")


def sinusoid(length, width, device):
    """1차원 사인·코사인 위치 인코딩을 [길이,특징 차원]으로 반환한다."""
    p = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    divisor = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / width)
    )
    result = torch.zeros(length, width, device=device)
    result[:, 0::2] = torch.sin(p * divisor)
    result[:, 1::2] = torch.cos(p * divisor[:result[:, 1::2].shape[1]])
    return result


def position_2d(h, w, hidden, device):
    """높이×너비 특징 지도를 펼칠 때 사용할 2차원 사인·코사인 위치 인코딩을 만든다."""
    assert hidden % 4 == 0
    yy = sinusoid(h, hidden // 2, device)
    xx = sinusoid(w, hidden // 2, device)
    return torch.cat((
        yy[:, None, :].expand(h, w, -1),
        xx[None, :, :].expand(h, w, -1),
    ), dim=-1).reshape(h * w, hidden)


class ActionLatentEncoder(nn.Module):
    """관절과 정답 행동으로 사후분포를 추정하며, 정답 행동은 학습할 때만 사용한다."""
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
        # 행동 순서에는 위치 인코딩을 더하고 CLS·관절 토큰에는 0을 더한다.
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
    """카메라마다 동일한 가중치를 공유하는 ResNet18로 시각 특징을 추출한다."""
    def __init__(self, hidden=512):
        super().__init__()
        backbone = resnet18(weights=None)
        # 전역 풀링과 1000종 분류기는 제거하고 합성곱 특징 지도만 유지한다.
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.project = nn.Conv2d(512, hidden, 1)

    def forward(self, images, trace=False):
        b, cams, rgb, h, w = images.shape
        assert rgb == 3
        x = images.reshape(b * cams, rgb, h, w)
        feat = self.project(self.features(x))
        _, d, fh, fw = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = tokens + position_2d(fh, fw, d, feat.device).unsqueeze(0)
        tokens = tokens.reshape(b, cams * fh * fw, d)
        if trace:
            shape("images [B,C,3,H,W]", images)
            shape("CNN maps [B*C,D,h,w]", feat)
            shape("visual tokens [B,C*h*w,D]", tokens)
        return tokens


class ACTPolicy(nn.Module):
    """미래 행동 시점마다 고정된 위치 쿼리를 사용하는 ACT 정책이다."""
    def __init__(self, dof=14, cameras=4, chunk=100, hidden=512,
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
        # 그림 4에서는 정답 행동 임베딩이 아니라 고정된 위치 쿼리를 사용한다.
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
            # 논문의 결정론적 추론 방식에 따라 사전분포 평균인 z=0을 사용한다.
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
    """L1 재구성 손실과 베타 가중 KL 손실을 더해 정규분포 사전분포와의 차이를 제약한다."""
    reconstruction = F.l1_loss(prediction, target)
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp())
    kl = kl.sum(dim=-1).mean()
    return reconstruction + beta * kl, reconstruction, kl


def temporal_ensemble(chunks, now, decay=0.01):
    """같은 실행 시점에 겹치는 행동 묶음의 예측을 결합한다.

    입력 묶음은 (예측 시작 시각, 행동[묶음 길이,제어 차원]) 쌍의 목록이다.
    이 예제에서는 더 최근에 예측한 행동에 더 높은 가중치를 부여한다.
    """
    relevant = [(t, a[now - t]) for t, a in chunks if t <= now < t + len(a)]
    if not relevant:
        raise ValueError("No chunk covers the requested execution timestep")
    # 가장 최신 예측을 기준으로 예측 시점의 차이에 따라 지수 가중치를 계산한다.
    newest = max(t for t, _ in relevant)
    values = torch.stack([a for _, a in relevant])
    weights = values.new_tensor([math.exp(-decay * (newest - t)) for t, _ in relevant])
    weights = weights / weights.sum()
    return (weights[:, None] * values).sum(0), weights


def main():
    parser = argparse.ArgumentParser()
    # 기본값은 그림 4의 차원이다. 메모리가 부족한 경우에만 실행 옵션으로 크기를 줄인다.
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=7)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--train-step", action="store_true")
    args = parser.parse_args()
    if (args.image_height < 32 or args.image_width < 32 or args.chunk_size < 2
            or args.hidden % 8 or args.layers < 1 or args.decoder_layers < 1):
        parser.error("height/width >= 32, chunk-size >= 2, hidden multiple of 8, layers >= 1")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ACTPolicy(chunk=args.chunk_size, hidden=args.hidden,
                      encoder_layers=args.layers, decoder_layers=args.decoder_layers).to(device)
    # 대형 모델의 메모리 사용량을 고려해 배치 크기를 1로 둔다.
    images = torch.randn(1, 4, 3, args.image_height, args.image_width, device=device)
    joints = torch.randn(1, 14, device=device) * 0.1
    gt = joints.unsqueeze(1) + torch.linspace(
        0.0, 0.2, args.chunk_size, device=device)[None, :, None]
    print(f"Device: {device}; 4 synthetic cameras, 14 control axes")
    shape("input RGB images [B,4,3,H,W]", images)
    shape("input joints [B,14]", joints)
    shape("GT action chunk [B,K,14]", gt)
    if args.image_height >= 480 and args.image_width >= 640:
        print("NOTE: Full-size attention (1202 visual/joint/z tokens) is memory-intensive.")
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
