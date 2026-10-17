# Deep Learning Paper Architectures

> **논문을 읽고, 핵심 Architecture를 직접 구현하며, Tensor와 학습 흐름을 확인하는 PyTorch 학습 프로젝트**

[![Python](https://img.shields.io/badge/Python-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
![Platform](https://img.shields.io/badge/Platform-Ubuntu%20%2F%20Linux-555555)
![Device](https://img.shields.io/badge/Device-CPU%20%7C%20CUDA-337AB7)
![Progress](https://img.shields.io/badge/Studies-7%20implemented-2F855A)

**Deep Learning Paper Architectures**는 Deep Learning, Computer Vision, Transformer 계열 논문의 주요 구성 요소를 간결한 PyTorch 실습으로 재구성하는 저장소입니다.

논문마다 작은 입력을 만들어 **입력 → Forward → Loss → Backward → Weight Update** 흐름을 직접 살펴보거나, 사전학습 모델의 Feature와 Prediction을 시각화합니다.

> [!IMPORTANT]
> 논문의 전체 모델을 재현하거나 공식 구현·벤치마크를 대체하는 프로젝트가 **아닙니다**. 실습마다 단순화 범위가 다르며, 1-step 학습 결과를 실제 모델 성능으로 해석하지 않습니다.

## 실습 목록

| 번호 | 논문 / 모델 | 핵심 학습 내용 | 실습 코드 | 상태 |
|:---:|---|---|---|:---:|
| 01 | ResNet | Residual Connection, Shortcut | [`01_resnet.py`](study/01_resnet.py) | 완료 |
| 02 | U-Net | Encoder–Decoder, Skip Connection | [`02_unet.py`](study/02_unet.py) | 완료 |
| 03 | DeepLabV3+ | Atrous Convolution, ASPP | [`03_deeplabv3plus.py`](study/03_deeplabv3plus.py) | 완료 |
| 04 | Attention Is All You Need | Self-Attention, Encoder–Decoder | [`04_transformer.py`](study/04_transformer.py) | 완료 |
| 05 | Vision Transformer (ViT) | Patch Embedding, CLS Token, MHSA | [`05_vit.py`](study/05_vit.py) | 완료 |
| 06 | DINOv2 | Pretrained Visual Feature, PCA, Cosine Similarity | [`06_dinov2.py`](study/06_dinov2.py) | 완료 |
| 07 | Segment Anything (SAM) | Sparse / Dense Prompt, Mask Refinement | [`07_sam.py`](study/07_sam.py) | 완료 |
| 08 | ACT | Action Chunking, CVAE, Transformer Policy, Temporal Ensembling | [`08_act.py`](study/08_act.py) | 실습 |
| 09 | DDPM | Noise Prediction, Denoising | — | 예정 |
| 10 | Diffusion Policy | Conditional Action Generation | — | 예정 |

'완료'는 **학습용 코드가 저장소에 존재함**을 의미하며, 논문 재현이나 학습 결과의 검증 완료를 뜻하지 않습니다.

## 빠른 시작

### 1. 요구 환경

- Ubuntu / Linux
- Git, Python 3 및 `venv` 모듈
- 패키지 설치를 위한 인터넷 연결
- CUDA 사용 시 호환되는 NVIDIA GPU 및 드라이버

CPU 노트북과 CUDA 지원 연구실 PC를 **동일한 저장소**에서 사용할 수 있습니다.

### 2. 저장소 복제

```bash
git clone https://github.com/nayana224/dl-paper-architectures.git
cd dl-paper-architectures
```

### 3. 개발 환경 설치

```bash
# GPU 자동 감지: CUDA 사용 가능 PC는 CUDA 환경, 그 외는 CPU 환경
bash scripts/setup_env.sh
```

설치 스크립트는 CPU에서는 `.venv-cpu/`, NVIDIA GPU가 감지되면 `.venv-cuda/`를 생성합니다. 각각 독립된 가상환경이며 Git에 포함되지 않습니다.

환경을 직접 선택하려면 다음 명령을 사용합니다.

```bash
bash scripts/setup_env.sh cpu
# 또는
bash scripts/setup_env.sh cuda
```

CUDA 설정은 **PyTorch CUDA 12.8 빌드**를 사용합니다. 호스트 드라이버가 호환되어야 하며, 설치 스크립트에서 `torch.cuda.is_available()`을 확인합니다. 자세한 호환성은 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)를 참고하세요.

### 4. 실습 실행

```bash
# CPU
.venv-cpu/bin/python study/05_vit.py

# NVIDIA GPU
.venv-cuda/bin/python study/05_vit.py
```

다른 실습도 파일명만 바꾸면 됩니다. 예를 들어 `study/01_resnet.py`, `study/04_transformer.py`, `study/07_sam.py`를 실행할 수 있습니다.

## 실습별 학습 포인트

<details>
<summary><strong>01–04 · 핵심 아키텍처와 학습 흐름</strong></summary>

| 실습 | 주요 확인 내용 |
|---|---|
| ResNet | `F(x) + shortcut`, Identity / Projection Shortcut, Gradient |
| U-Net | Downsampling / Upsampling, Skip Concat, Pixel-level Loss |
| DeepLabV3+ | Atrous Convolution, ASPP, Low-level Feature 결합 |
| Transformer | Token Embedding, Positional Encoding, Self / Cross-Attention, Causal Mask |

이 구간은 작은 합성 입력과 축소된 모델을 사용해 Forward, Loss, Backward, Optimizer Step을 확인합니다.

원문과의 차이: ResNet은 소형 BasicBlock, U-Net은 Same Padding과 간소화된 Upsampling, DeepLabV3+는 축소된 Backbone / Decoder, Transformer는 1-layer·`d_model=32`의 학습용 Encoder–Decoder입니다.

**원문:** [ResNet](https://arxiv.org/abs/1512.03385) · [U-Net](https://arxiv.org/abs/1505.04597) · [DeepLabV3+](https://arxiv.org/abs/1802.02611) · [Transformer](https://arxiv.org/abs/1706.03762)

</details>

<details>
<summary><strong>05 · Vision Transformer (ViT)</strong></summary>

224×224 이미지 한 장과 16×16 Patch를 사용해 **Patch Embedding → CLS Token → Position Embedding → Pre-LN Encoder → Classification** 과정을 확인합니다. Self-Attention의 Q/K/V, Score, Softmax와 Value 가중합을 코드에서 직접 구현합니다.

```text
Image             [1, 3, 224, 224]
  ↓ Patchify
Patch Vectors     [1, 196, 768]
  ↓ Linear Projection
Patch Tokens      [1, 196, 128]
  ↓ CLS + Position Embedding
Token Sequence    [1, 197, 128]
  ↓ Transformer Encoder (1 block / 4 heads)
Attention Weights [1, 4, 197, 197]
  ↓ Final CLS + Classification Head
Logits            [1, 3]
  ↓ CrossEntropyLoss / Backward / Optimizer Step
```

출력: `outputs/01_patch_grid.png`, `outputs/01_attention_heatmap.png`

> 합성 Ball 이미지 한 장을 사용한 **무작위 초기화 모델의 1-step 학습 예시**입니다. Attention Heatmap은 학습된 모델의 의미 있는 주목 영역을 나타내지 않습니다.

**자료:** [ViT 논문](https://arxiv.org/abs/2010.11929) · [공식 구현](https://github.com/google-research/vision_transformer)

</details>

<details>
<summary><strong>06 · DINOv2: Pretrained Visual Feature</strong></summary>

사전학습된 `facebook/dinov2-small` 모델에서 CLS / Patch Feature를 추출한 뒤, **PCA-RGB 시각화**와 **중앙 Patch 기준 Cosine Similarity**를 확인합니다. Teacher–Student 학습을 직접 재현하지 않습니다.

```bash
.venv-cpu/bin/python study/06_dinov2.py
.venv-cpu/bin/python study/06_dinov2.py --image /path/to/photo.jpg
```

출력: `outputs/06_dinov2_features.png`

기본 샘플 사진과 Pretrained Weight는 첫 실행 시 다운로드될 수 있습니다. PCA 색상은 클래스 라벨이 아니며 Cosine Similarity는 Segmentation Mask가 아닙니다.

**자료:** [DINOv2 논문](https://arxiv.org/abs/2304.07193) · [공식 구현](https://github.com/facebookresearch/dinov2)

</details>

<details>
<summary><strong>07 · Segment Anything (SAM): Promptable Segmentation</strong></summary>

사전학습된 `facebook/sam-vit-base`를 사용해 **Point**, **Box**, **Point + Box**, **Point + Box + 이전 Mask Logits**의 결과를 비교합니다. Point / Box는 Sparse Prompt, 이전 Mask Logits는 Dense Prompt입니다.

```bash
.venv-cuda/bin/python study/07_sam.py

# 다른 이미지와 Prompt 좌표 지정
.venv-cuda/bin/python study/07_sam.py \
  --image /path/to/photo.jpg \
  --point 320 240 \
  --box 150 100 490 390
```

CPU에서는 `.venv-cuda/bin/python`을 `.venv-cpu/bin/python`으로 변경하세요.

출력: `outputs/07_sam_prompts.png`

기본 예시 사진 및 Model Weight는 첫 실행 시 다운로드될 수 있습니다. Predicted IoU는 실제 GT와 계산한 IoU가 아니고, Mask Refinement가 반드시 품질 향상을 보장하지 않습니다.

**자료:** [SAM 논문](https://arxiv.org/abs/2304.02643) · [공식 구현](https://github.com/facebookresearch/segment-anything)

</details>

<details>
<summary><strong>08 · ACT 배경지식: AE → VAE</strong></summary>

ACT의 CVAE를 이해하기 위한 **독립적인 교육용 예제**입니다. ACT 원본 Policy 구현은 아직 포함하지 않습니다. 두 예제는 같은 가상 14차원 관절 궤적([512,10,14])을 사용하고, 첫 샘플의 세 관절을 시각화합니다.

```bash
git pull origin main
.venv-cuda/bin/python study/act/02_autoencoder.py
.venv-cuda/bin/python study/act/03_vae.py

# CPU 환경: .venv-cpu/bin/python 사용
# 학습 횟수/잠재공간 KL 제약 변경
.venv-cuda/bin/python study/act/03_vae.py --epochs 500 --beta 0.1
```

- `02_autoencoder.py`: Input/GT `[B,10,14]` → Flatten → Encoder → `z[B,8]` → Decoder → Reconstruction `[B,10,14]`; MSE만 사용.
- `03_vae.py`: Encoder → `mu[B,8]`, `logvar[B,8]` → reparameterization → `z[B,8]`; **MSE + beta × KL**로 학습. 시각화 시에는 무작위 샘플 대신 `z=mu`를 사용해 복원값을 일정하게 표시합니다.
- 결과: `outputs/08_act_ae.png`, `outputs/08_act_vae.png` (원본/복원 궤적 및 학습 Loss).

### ACT Figure 4 — Forward Only (Tensor 흐름)

```bash
.venv-cuda/bin/python study/act/04_act_forward.py
# 100개 미래 Action으로 비교
.venv-cuda/bin/python study/act/04_act_forward.py --chunk-size 100
```

- **Input:** 현재 joints `[B,14]`, 가상 카메라 `[B,2,3,32,32]`, GT Action Chunk `[B,K,14]`. GT 행은 미래 시점, 열은 14개 관절·그리퍼 제어축입니다.
- **CVAE Encoder (학습 모드만):** `CLS[1] + joints[1] + GT actions[K]` → `[B,K+2,32]` → `mu,logvar,z[B,8]`.
- **ACT Policy (학습/추론 모두):** 카메라 CNN Feature Token + joints Token + z Token → Policy Transformer Encoder → Position-based Action Queries K개를 사용하는 Transformer Decoder → `[B,K,14]`.
- **추론:** GT 없이 `z=0`으로 왼쪽 CVAE Encoder를 생략합니다.
- **주의:** 논문 흐름을 교육용으로 축소한 **무작위 초기화 모델의 순전파 1회**로, 실측 이미지/실제 ACT 체크포인트/학습/유효한 로봇 동작 예측은 포함되지 않습니다. 표시하는 L1 및 KL 값도 학습된 성능 지표가 아닙니다.

**해석:** 이 값들은 실측 로봇 데이터가 아닌 합성 관절 궤적입니다. VAE의 KL은 표준정규 사전분포 쪽으로 잠재분포를 제약하지만 유효한 동작 생성을 보장하지 않습니다. 두 코드의 출력 품질은 `beta`, 학습 횟수에 따라 달라질 수 있고, CVAE의 **조건부** 동작 생성은 다음 단계에서 다룹니다.

</details>

## 출력 및 파일 구조

```text
dl-paper-architectures/
├── study/
│   ├── 01_resnet.py
│   ├── 02_unet.py
│   ├── 03_deeplabv3plus.py
│   ├── 04_transformer.py
│   ├── 05_vit.py
│   ├── 06_dinov2.py
│   ├── 07_sam.py
│   └── act/
│       ├── 02_autoencoder.py
│       ├── 03_vae.py
│       └── 04_act_forward.py
├── scripts/
│   └── setup_env.sh
├── assets/                   # 예제 입력 이미지 등 (필요시 생성)
├── outputs/                  # 실행 시 생성, Git 추적 제외
├── requirements.txt
└── README.md
```

`outputs/`, 가상환경, 체크포인트 및 다운로드된 예제 이미지는 로컬에서 관리합니다. 대규모 데이터셋이나 학습 Weight를 Git 저장소에 추가하지 않습니다.

## 학습 및 구현 원칙

1. 논문의 **Problem → Core Idea → Architecture → Tensor Shape** 순서로 이해합니다.
2. 핵심 동작을 작은 입력과 간단한 PyTorch 코드로 확인합니다.
3. 가능한 경우 Forward / Loss / Backward / Weight Update의 관계를 관찰합니다.
4. Pretrained Inference와 축소 모델의 1-step 학습을 구분합니다.
5. 공식 구현과 다른 단순화 사항 및 결과 해석의 한계를 명시합니다.

> 본 저장소는 **학습과 연구 이해를 위한 실습 아카이브**입니다. 공식 논문 및 모델 구현의 라이선스와 인용 조건은 각 프로젝트를 확인해 주세요.


### ACT 한 파일로 아키텍처 전체 확인

기존 `study/act/` 코드는 이전 단계별 학습 자료로 유지합니다. **새로 사용하는 통합 ACT 코드는 `study/08_act.py` 한 파일**입니다.

```bash
.venv-cpu/bin/python study/08_act.py
.venv-cpu/bin/python study/08_act.py --train-step
# 연구실 GPU에서 논문 Figure 4에 가까운 설정
.venv-cuda/bin/python study/08_act.py --hidden 512 --layers 4 --decoder-layers 7
```

4개 가상 카메라 + 14차원 관절 + GT action chunk → CVAE posterior → ResNet18 feature → Transformer encoder/decoder → k×14 예측을 한 스크립트에서 출력합니다. 학습 때는 GT를 이용하여 z를 추정하고, 추론 때는 z=0으로 동작합니다. Synthetic data / random weights이므로 로봇 제어에는 사용할 수 없습니다.
