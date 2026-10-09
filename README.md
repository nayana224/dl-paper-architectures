# Deep Learning Paper Architectures

딥러닝 논문의 핵심 **Architecture와 학습 데이터 흐름**에 익숙해지기 위한 개인 PyTorch 실습 모음입니다.

- 논문당 **핵심 학습 코드 1개**를 `study/`에 추가합니다.
- 실제 이미지 1장, 짧은 문장, 작은 Action Sequence 등 **최소 샘플 입력**을 사용합니다.
- 가능한 경우 `Input → Forward → GT/Loss → Backward → Weight Update`와 주요 Tensor Shape를 확인합니다.
- Feature / Attention / Prediction을 관찰할 수 있으면 `outputs/`에 시각화합니다.
- 논문의 **전체 재현이나 공식 구현 대체가 아닙니다.** 공식 모델 실행, 대규모 학습, 벤치마크 재현은 논문별 공식 저장소 또는 별도 프로젝트에서 수행합니다.
- 아직 학습하지 않은 논문은 파일을 미리 만들지 않습니다.

## 개발 환경 자동 설정 (Ubuntu / Linux)

컴퓨터마다 `.venv`는 로컬에 생성되어 GitHub에 올라가지 않습니다. **가상환경을 PC별로 공유할 필요는 없습니다.** CPU와 NVIDIA CUDA를 구분할 수 있도록 아래 스크립트가 `.venv-cpu` 또는 `.venv-cuda`를 자동 생성합니다.

```bash
git pull
bash scripts/setup_env.sh
```

NVIDIA GPU가 `nvidia-smi`로 감지되면 CUDA 12.8 PyTorch 빌드를, 아니면 CPU 빌드를 설치하며 CUDA 사용 여부를 검증합니다. Ubuntu의 NVIDIA 드라이버 버전과 호환성이 필요합니다. 선택을 강제하려면 `bash scripts/setup_env.sh cpu` 또는 `bash scripts/setup_env.sh cuda`를 사용하세요. **기존 `.venv`는 건드리지 않습니다.**

설치 후 실행:

```bash
# CPU 노트북
.venv-cpu/bin/python study/05_vit.py

# NVIDIA GPU 연구실 PC
.venv-cuda/bin/python study/05_vit.py
```

가상환경 이름은 실제 GPU 성능을 결정하지 않습니다. 각 스크립트는 `torch.cuda.is_available()`로 GPU를 자동 선택합니다. 설치에 실패할 경우 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에서 현재 드라이버와 Python 버전에 맞는 명령어를 확인하세요.

## 실습 목록

| 논문 | 핵심 주제 | 코드 | 상태 |
| --- | --- | --- | --- |
| ResNet | Residual Connection | [`study/01_resnet.py`](study/01_resnet.py) | 구현 |
| U-Net | Skip Connection / Dense Prediction | [`study/02_unet.py`](study/02_unet.py) | 구현 |
| DeepLabV3+ | Atrous Convolution / Segmentation | [`study/03_deeplabv3plus.py`](study/03_deeplabv3plus.py) | 구현 |
| Attention Is All You Need | Self-Attention / Encoder–Decoder | [`study/04_transformer.py`](study/04_transformer.py) | 구현 |
| Vision Transformer (ViT) | Patch, CLS, MHSA, Classification | [`study/05_vit.py`](study/05_vit.py) | 구현 |
| DINOv2 | Self-supervised Visual Feature | [`study/06_dinov2.py`](study/06_dinov2.py) | 구현 |
| Segment Anything (SAM) | Promptable Segmentation | `study/07_sam.py` | 선택 |
| ACT | Action Chunking / CVAE | `study/08_act.py` | 예정 |
| DDPM | Noise Prediction / Denoising | `study/09_ddpm.py` | 예정 |
| Diffusion Policy | Conditional Action Generation | `study/10_diffusion_policy.py` | 예정 |

**예정/선택 표시는 구현 완료를 뜻하지 않습니다.** 모든 논문에 같은 분량의 코드를 강제하지 않으며 DINOv2·SAM처럼 필요하면 공식 Demo만 검토하고 넘어갈 수 있습니다.

## 01~04. ViT 이전 핵심 아키텍처

각 파일은 이미지를 직접 그리거나 짧은 token ID를 사용하며, **하나의 Forward / Loss / Backward / Optimizer Step**을 보여주는 교육용 축소 구현입니다.

| 논문 | 입력 및 GT | 코드에서 확인할 핵심 | 논문 원문 / 공식 코드 |
| --- | --- | --- | --- |
| 01 ResNet | 파란 사각형 RGB 이미지, 가상 class ID | F(x)+shortcut, Identity/Projection, Gradients | [Paper](https://arxiv.org/abs/1512.03385) / [Official](https://github.com/KaimingHe/deep-residual-networks) |
| 02 U-Net | 원형 이미지, 픽셀 마스크 | Encoder–Decoder, Skip Concat, Pixel Cross-Entropy | [Paper](https://arxiv.org/abs/1505.04597) / [Official](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) |
| 03 DeepLabV3+ | 사각형 이미지, 픽셀 마스크 | Dilated Convolution, ASPP, Low-level Decoder | [Paper](https://arxiv.org/abs/1802.02611) / [Official](https://github.com/tensorflow/models/tree/master/research/deeplab) |
| 04 Transformer | 가상 Source/Target token ID | Sinusoidal Position, Encoder Self-Attention, Decoder Causal/Cross-Attention, Teacher Forcing | [Paper](https://arxiv.org/abs/1706.03762) / [Annotated Implementation](https://nlp.seas.harvard.edu/annotated-transformer/) |

**원 논문과의 주요 차이:** U-Net은 valid convolution + crop을 same padding과 bilinear upsampling으로 단순화하고, DeepLabV3+는 Backbone과 Output Stride, Atrous Separable Convolution을 단순화했습니다. Transformer는 Layer 1개, D=32의 Post-LN Encoder–Decoder입니다. ResNet은 소형 BasicBlock 예시이며 논문 전체 ResNet-34/50을 구현하지 않았습니다.

실행:

```bash
python study/01_resnet.py
python study/02_unet.py
python study/03_deeplabv3plus.py
python study/04_transformer.py
```

각 스크립트는 임의 초기화 상태에서 **1 step만 학습**하므로 Accuracy/Segmentation 품질/번역 성능을 나타내지 않습니다.

## 05. Vision Transformer

논문: [An Image Is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)  
공식 구현: [google-research/vision_transformer](https://github.com/google-research/vision_transformer)

학습용 소형 ViT를 무작위 초기화하여, Ball 이미지 1장(메모리에서 직접 그림)을 가지고 Forward부터 SGD Weight Update까지 **1 step**을 실행합니다.

```text
Image [1,3,224,224]
 → Patchify [1,196,768]
 → Patch Embedding [1,196,128]
 → CLS + Position Embedding [1,197,128]
 → Pre-LN Encoder (1 block)
     Q/K/V [1,4,197,32] 각각
     Attention weights [1,4,197,197]
     Residual + MLP [1,197,128]
 → Final CLS [1,128]
 → Classification Head [1,3]
 → CrossEntropyLoss(logits, GT)
 → backward() → optimizer.step()
```

설정: 16×16 Patch, D=128, Head=4, 3개 가상 클래스(Bird/Ball/Car), GT=Ball. Self-Attention의 Q/K/V, Attention Score, Softmax, Value 집계를 직접 구현합니다.

**주의:** 1장의 가상 이미지에서 단 1 step만 학습합니다. Prediction, Softmax Score, Heatmap은 실제 분류 성능이나 주목 영역에 관한 실험 근거가 아닙니다.

### 실행

```bash
bash scripts/setup_env.sh
# 아래 중 해당 환경을 선택해 실행
.venv-cpu/bin/python study/05_vit.py
# 또는 .venv-cuda/bin/python study/05_vit.py
```

환경 설정과 CUDA 검증은 위의 자동 설치 스크립트를 이용합니다.

실행하면 다음 파일을 **로컬** `outputs/`에 생성합니다(버전 관리 제외).

- `01_patch_grid.png`: 원본 이미지에 표시한 16×16 Patch 경계
- `01_attention_heatmap.png`: 첫 Head의 CLS Query → Patch Key Attention Weight

Heatmap에는 CLS 자신에 대한 가중치를 제외한 196개 Patch의 가중치를 보여줍니다. 전체 CLS Attention 행(197개 Key)의 Softmax 합은 1입니다. 학습되지 않은 Attention을 모델의 판단 근거로 해석하지 마세요.

## 06. DINOv2 — Pretrained Feature 살펴보기

논문: [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)  
공식 코드: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)  
모델: [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small)

DINOv2는 **Segmentation Mask를 직접 출력하는 모델이 아니라**, 재사용 가능한 이미지 Feature를 추출하는 Backbone입니다. 연구 목적상 내부 Teacher–Student Loss를 다시 구현하지 않고 Pretrained Inference 결과만 확인합니다.

```text
실제 사진 1장 → Image Processor → DINOv2-S/14
→ CLS [1,384] + Patch Feature [1,256,384] (224x224 입력의 경우)
→ Patch Feature PCA-RGB / 중앙 Patch 기준 Cosine Similarity
```

```bash
# 노트북 또는 연구실 PC에서 처음 한 번만 설치
bash scripts/setup_env.sh
# CPU 예시 (CUDA PC라면 .venv-cuda/bin/python)
.venv-cpu/bin/python study/06_dinov2.py

# 직접 찍은 사진으로 실행
.venv-cpu/bin/python study/06_dinov2.py --image /path/to/my_photo.jpg
```

첫 실행에는 PyTorch Hub의 강아지 실사진 샘플과 Hugging Face Pretrained Model Weight를 다운로드하여 캐시합니다. 다운로드가 제한된 환경에서는 `--image`를 지정하세요(모델 Weight 다운로드는 별도로 필요). 실행 결과는 `outputs/06_dinov2_features.png`에 저장합니다.

**해석 주의:** PCA 채널별 색은 클래스 라벨이 아니고, 중앙 Patch와의 Cosine Similarity는 GT Mask나 SAM 출력이 아닙니다. 원본 이미지는 Processor에서 Resize/Crop될 수 있으므로 결과 Grid와 픽셀 위치가 완벽하게 대응하지 않을 수 있습니다. DINOv2 실습에서는 Feature가 구별되는지를 관찰하는 정도로 마무리합니다.

## 자료 관리 원칙

`assets/`는 앞으로 실제 이미지·문장·작은 로봇 데이터 샘플이 필요한 논문을 위한 폴더입니다. 데이터셋 전체, Checkpoint, 대량 출력은 Git에 올리지 않습니다. 실습 결과를 해석할 때는 **교육용 구현 / 공식 논문 구현 / 실제 성능 재현**을 구분합니다.
