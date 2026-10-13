"""ACT Figure 4를 따라가는 순전파(Forward Only) 실습.

학습(Backward/Optimizer), Pretrained Weight, 로봇 데이터 사용 없음.
논문을 이해하기 위한 *축소 모델*: 2개 가상 카메라/작은 CNN/작은 Transformer.
목표: 행=시간, 열=14개 제어축인 GT [B,K,14]가 어디에 들어가는지 추적.
"""
import argparse
import torch
from torch import nn
import torch.nn.functional as F


class Figure4Forward(nn.Module):
    def __init__(self, dof=14, chunk_size=4, cameras=2, hidden=32, latent=8):
        super().__init__()
        self.dof, self.k, self.cameras = dof, chunk_size, cameras
        self.hidden, self.latent = hidden, latent

        # FIGURE 4 LEFT: CVAE Encoder (학습 시 GT를 보며 style z 추정)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        self.joint_to_token = nn.Linear(dof, hidden)
        self.action_to_token = nn.Linear(dof, hidden)
        self.cvae_position = nn.Embedding(chunk_size + 2, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=64,
            dropout=0.0, batch_first=True,
        )
        self.cvae_transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.to_mu = nn.Linear(hidden, latent)
        self.to_logvar = nn.Linear(hidden, latent)

        # FIGURE 4 MIDDLE: 카메라 Feature + 현재 joints + style z
        self.image_cnn = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.image_to_token = nn.Linear(8, hidden)
        self.policy_joint = nn.Linear(dof, hidden)
        self.policy_style = nn.Linear(latent, hidden)
        self.image_position = nn.Embedding(cameras * 4, hidden)
        policy_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=64,
            dropout=0.0, batch_first=True,
        )
        self.policy_encoder = nn.TransformerEncoder(policy_layer, num_layers=1)

        # FIGURE 4 RIGHT: K개의 Action Query로 K개 14D Action 생성
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=64,
            dropout=0.0, batch_first=True,
        )
        self.policy_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.action_query_position = nn.Embedding(chunk_size, hidden)
        self.action_head = nn.Linear(hidden, dof)

    def encode_gt(self, joints, gt):
        b = joints.size(0)
        cls = self.cls.expand(b, -1, -1)          # [B,1,D]
        joint_tok = self.joint_to_token(joints).unsqueeze(1)
        action_tok = self.action_to_token(gt)     # [B,K,D]
        tokens = torch.cat([cls, joint_tok, action_tok], dim=1)
        pos = self.cvae_position(torch.arange(self.k + 2, device=joints.device))
        tokens = tokens + pos.unsqueeze(0)       # [B,K+2,D]
        features = self.cvae_transformer(tokens)
        cls_feature = features[:, 0]             # [B,D]
        mu, logvar = self.to_mu(cls_feature), self.to_logvar(cls_feature)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps
        return z, mu, logvar, tokens

    def policy(self, images, joints, z):
        b, cameras, rgb, height, width = images.shape
        assert cameras == self.cameras and rgb == 3
        # [B,C,3,H,W] -> [B*C,3,H,W] -> [B,C*4,8]
        patches = self.image_cnn(images.reshape(b * cameras, rgb, height, width))
        patches = patches.flatten(2).transpose(1, 2).reshape(b, cameras * 4, 8)
        image_tokens = self.image_to_token(patches)
        image_tokens = image_tokens + self.image_position(
            torch.arange(cameras * 4, device=images.device)
        ).unsqueeze(0)
        joint_tok = self.policy_joint(joints).unsqueeze(1)
        z_tok = self.policy_style(z).unsqueeze(1)
        condition_tokens = torch.cat([image_tokens, joint_tok, z_tok], dim=1)
        memory = self.policy_encoder(condition_tokens)
        # Learned fixed-position action queries: GT Action 입력하지 않음
        queries = self.action_query_position(
            torch.arange(self.k, device=images.device)
        ).unsqueeze(0).expand(b, -1, -1)
        decoded = self.policy_decoder(tgt=queries, memory=memory)
        predicted = self.action_head(decoded)    # [B,K,14]
        return predicted, condition_tokens, memory, queries

    def forward(self, images, joints, gt=None):
        if gt is None:                           # 추론: 왼쪽 CVAE 생략
            z = torch.zeros(joints.size(0), self.latent, device=joints.device)
            mu = logvar = cvae_tokens = None
        else:
            z, mu, logvar, cvae_tokens = self.encode_gt(joints, gt)
        pred, condition, memory, queries = self.policy(images, joints, z)
        return pred, z, mu, logvar, cvae_tokens, condition, memory, queries


def show(name, tensor):
    print(f"  {name:28s} {str(tuple(tensor.shape)):16s}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=4)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be >= 1")

    torch.manual_seed(12)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, k, dof, cameras = 2, args.chunk_size, 14, 2

    # 가상의 한 시연에서 얻은 '현재 관절'과 '미래 전문가 Action'
    # 각 행은 한 timestep, 각 열은 한 관절/그리퍼 제어축.
    base = torch.linspace(-0.6, 0.6, dof)
    times = torch.arange(k, dtype=torch.float32).unsqueeze(1)
    gt = base[None, None, :] + 0.03 * times.unsqueeze(0)
    gt = gt.expand(b, -1, -1).clone()           # [B,K,14]
    gt[1] = gt[1] + 0.2                         # 다른 시연 샘플
    joints = gt[:, 0].clone()                   # [B,14]
    images = torch.rand(b, cameras, 3, 32, 32)  # 가상 RGB, 학습된 이미지 정보 없음

    images, joints, gt = images.to(device), joints.to(device), gt.to(device)
    model = Figure4Forward(chunk_size=k, cameras=cameras).to(device).eval()

    print("\n=== 입력 데이터: GT 행렬의 행/열 ===")
    show("camera images [B,C,RGB,H,W]", images)
    show("current joints [B,14]", joints)
    show("GT actions [B,K,14]", gt)
    print("  GT[0]  (행=timestep / 열=제어축 0..13):")
    print(gt[0].cpu().numpy().round(2))
    print("  GT[0, :, 0] = 첫 제어축의 시간별 값:",
          gt[0, :, 0].cpu().tolist())

    with torch.inference_mode():
        pred, z, mu, logvar, cvae_tokens, condition, memory, queries = (
            model(images, joints, gt)
        )
        inference, inference_z, *_ = model(images, joints)

    print("\n=== TRAINING 모드의 Forward (GT 사용; 학습 업데이트는 없음) ===")
    show("CLS + joints + GT tokens", cvae_tokens)
    show("mu", mu)
    show("logvar", logvar)
    show("sampled z", z)
    show("image + joints + z tokens", condition)
    show("policy encoder memory", memory)
    show("action query tokens", queries)
    show("predicted actions [B,K,14]", pred)
    print("  예상: CVAE 입력 Token 수 = 1(CLS) + 1(joints) + K(actions)")
    recon = F.l1_loss(pred, gt)
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1).mean()
    print(f"  Example L1 reconstruction={recon.item():.5f}; "
          f"KL={kl.item():.5f} (random weights; not trained)")

    print("\n=== INFERENCE 모드의 Forward (GT 없음) ===")
    show("z = 0 (CVAE skipped)", inference_z)
    show("predicted actions [B,K,14]", inference)
    print("  z = 0:", bool(torch.all(inference_z == 0)))
    print("\n주의: 무작위 초기화한 축소 모형이므로 Action 값은 의미가 없습니다.")
    print("기존 Figure 4와 데이터 흐름을 대응시키기 위한 순전파 전용 실습입니다.")


if __name__ == "__main__":
    main()
