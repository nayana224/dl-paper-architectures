# Deep Learning Paper Architectures

딥러닝 논문의 핵심 **Architecture와 학습 데이터 흐름**에 익숙해지기 위한 개인 PyTorch 실습 모음입니다.

- 논문당 **핵심 학습 코드 1개**를 `study/`에 추가합니다.
- 실제 이미지 1장, 짧은 문장, 작은 Action Sequence 등 **최소 샘플 입력**을 사용합니다.
- 가능한 경우 `Input → Forward → GT/Loss → Backward → Weight Update`와 주요 Tensor Shape를 확인합니다.
- Feature / Attention / Prediction을 관찰할 수 있으면 `outputs/`에 시각화합니다.
- 논문의 **전체 재현이나 공식 구현 대체가 아닙니다.** 공식 모델 실행, 대규모 학습, 벤치마크 재현은 논문별 공식 저장소 또는 별도 프로젝트에서 수행합니다.
- 아직 학습하지 않은 논문은 파일을 미리 만들지 않습니다.

## 실습 목록

| 논문 | 핵심 주제 | 코드 | 상태 |
| --- | --- | --- | --- |
| ResNet | Residual Connection | [`study/01_resnet.py`](study/01_resnet.py) | 구현 |
| U-Net | Skip Connection / Dense Prediction | [`study/02_unet.py`](study/02_unet.py) | 구현 |
| DeepLabV3+ | Atrous Convolution / Segmentation | [`study/03_deeplabv3plus.py`](study/03_deeplabv3plus.py) | 구현 |
| Attention Is All You Need | Self-Attention / Encoder–Decoder | [`study/04_transformer.py`](study/04_transformer.py) | 구현 |
| Vision Transformer (ViT) | Patch, CLS, MHSA, Classification | [`study/05_vit.py`](study/05_vit.py) | 구현 |
| DINOv2 | Self-supervised Visual Feature | `study/06_dinov2.py` | 선택 |
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
python3 -m venv .venv  # 처음 한 번만
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python study/05_vit.py
```

기존 `.venv`를 쓰고 있다면 생성 단계는 생략하세요. CUDA 사용 여부는 설치된 PyTorch 환경에 따라 자동 결정됩니다. GPU 설치가 필요하면 [PyTorch 공식 안내](https://pytorch.org/get-started/locally/)를 참고하세요.

실행하면 다음 파일을 **로컬** `outputs/`에 생성합니다(버전 관리 제외).

- `01_patch_grid.png`: 원본 이미지에 표시한 16×16 Patch 경계
- `01_attention_heatmap.png`: 첫 Head의 CLS Query → Patch Key Attention Weight

Heatmap에는 CLS 자신에 대한 가중치를 제외한 196개 Patch의 가중치를 보여줍니다. 전체 CLS Attention 행(197개 Key)의 Softmax 합은 1입니다. 학습되지 않은 Attention을 모델의 판단 근거로 해석하지 마세요.

## 자료 관리 원칙

`assets/`는 앞으로 실제 이미지·문장·작은 로봇 데이터 샘플이 필요한 논문을 위한 폴더입니다. 데이터셋 전체, Checkpoint, 대량 출력은 Git에 올리지 않습니다. 실습 결과를 해석할 때는 **교육용 구현 / 공식 논문 구현 / 실제 성능 재현**을 구분합니다.
