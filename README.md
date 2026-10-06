# ViT PyTorch Study

논문 **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**를 이해하기 위한 개인 구현·실습 저장소입니다. Google의 JAX 구현을 참고 자료로 삼되, 실험하기 쉬운 PyTorch를 사용합니다. Pretrained 실험은 torchvision과 Hugging Face Transformers를 사용합니다. 학습 프레임워크나 논문 결과 재현 프로젝트는 아닙니다.

## 공부 체크리스트

1. **Problem**: 어떤 문제를 해결하는가?
2. **Core Idea**: 이미지를 패치 토큰으로 바꾸는 이유는 무엇인가?
3. **Method**: Patch Embedding부터 Classification Head까지 설명할 수 있는가?
4. **Input / GT / Output / Loss**: 각 텐서와 정답, 학습 목적은 무엇인가?
5. **Evidence**: 논문의 실험 근거와 내 실습 관찰을 구분했는가?
6. **My Observation**: 직접 확인한 현상과 남은 질문은 무엇인가?

## 실습 흐름

| 파일 | 배우는 내용 |
| --- | --- |
| `study/00_prepare_dataset.py` | Oxford-IIIT Pet trainval을 다운로드하고 고정된 6개 품종의 첫 이미지를 선택합니다. 이미지와 클래스·인덱스 정보를 담은 CSV manifest를 assets에 저장합니다. |
| `study/01_patchify.py` | `unfold`로 224×224 이미지를 16×16 패치로 나누고 shape 변화를 확인합니다. 14×14 패치 그리드를 저장합니다. |
| `study/02_patch_embedding.py` | 픽셀 벡터 768차원을 `nn.Linear`로 교육용 D=128에 투영합니다. raw patch dimension과 hidden size의 차이를 확인합니다. |
| `study/03_cls_position.py` | 학습 가능한 CLS token과 Position Embedding을 추가합니다. 196개 패치 토큰이 197개로 바뀝니다. |
| `study/04_encoder_forward.py` | D=128, head=4인 단일 Pre-LN Encoder를 통과합니다. residual 연결과 head별 attention shape를 확인합니다. |
| `study/05_pretrained_inference.py` | torchvision ViT-B/16으로 모든 표본의 ImageNet Top-5 예측을 출력합니다. Pet 정답과 ImageNet 예측은 서로 다른 label space입니다. |
| `study/06_attention_visualization.py` | torchvision의 head 평균 attention에 residual을 더하고 행 정규화한 뒤 layer 순서대로 누적합니다. 실제 입력 crop, CLS rollout 그리드, 확대 맵, overlay를 저장합니다. |
| `study/07_huggingface_vit_gpu.py` | `google/vit-base-patch16-224`를 CUDA 또는 CPU에서 실행합니다. 설정·예측·모든 Hidden State와 Attention shape를 확인합니다. |
| `study/08_attention_inspection.py` | 하나의 layer/head에서 CLS query의 attention을 수치로 확인합니다. CLS→CLS, CLS→patch, Top-10 patch, 행의 합, heatmap과 overlay를 확인합니다. |

00~08은 각각 실행 가능한 스크립트이며, 00에서 만든 assets 외에는 앞 단계의 실행 결과에 의존하지 않습니다. 01~04의 파라미터는 무작위 초기화 상태로 학습하지 않습니다.

## 핵심 Tensor 흐름

```text
Image [B,3,224,224]
→ 16×16 patches
→ 196 patch tokens [B,196,768] (raw pixels: 16²×3)
→ Patch Embedding D=768 [B,196,768]
→ CLS token 추가
→ 197 tokens [B,197,768]
→ Position Embedding
→ Transformer Encoder ×12
→ CLS representation [B,768]
→ Classification Head
→ logits [B,num_classes]
```

ViT-B/16 설정: Patch size **16×16**, Hidden size D **768**, Layers **12**, Heads **12**, MLP dim **3072**. 패치 픽셀 차원 `P²C=768`과 hidden size `D=768`은 수치가 같아도 서로 다른 개념입니다.

Hugging Face 예제의 입력은 `[1,3,224,224]`, Hidden State는 `[1,197,768]`, Attention은 `[1,12,197,197]`입니다. Hidden State는 embedding 출력과 12개 layer 출력을 합쳐 13개, Attention은 12개입니다. Attention의 마지막 두 축은 query와 key이며 각 query 행은 softmax로 정규화됩니다.

## 실행 방법

Ubuntu 22.04와 NVIDIA GPU를 목표로 하며 GPU가 없으면 CPU로 실행합니다. 환경에 맞는 Python 가상환경에서 실행하세요.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python study/00_prepare_dataset.py
python study/01_patchify.py
python study/02_patch_embedding.py
python study/03_cls_position.py
python study/04_encoder_forward.py
python study/05_pretrained_inference.py
python study/06_attention_visualization.py
python study/07_huggingface_vit_gpu.py
python study/08_attention_inspection.py
# layer와 head는 0-based; 기본값은 마지막 layer, head 0
python study/08_attention_inspection.py --layer 11 --head 3
```

Ubuntu 22.04에서 새 가상환경의 pip가 22.0.2라면 의존성 설치 중 `AssertionError`가 발생할 수 있으므로 위의 pip 업데이트를 먼저 실행하세요. 설치가 실패했다면 실습 실행을 멈추고 pip 업데이트 후 `python -m pip install -r requirements.txt`를 다시 실행하세요. `ModuleNotFoundError: torch/torchvision`는 설치 실패의 후속 오류일 수 있습니다.

00은 전체 Oxford-IIIT Pet 데이터셋을 내려받으므로 네트워크와 저장 공간이 필요합니다. 05~08의 첫 실행은 pretrained 가중치를 다운로드합니다. Hugging Face와 torchvision 가중치는 각각 캐시되며 서로 다른 pretrained 모델이므로 예측·attention이 같다고 가정하지 않습니다.

`study/assets/`의 선택된 이미지와 manifest는 버전 관리가 가능합니다. 전체 데이터셋 `study/data/`와 생성 이미지 `study/outputs/`는 git에서 제외됩니다. 스크립트의 경로는 파일 위치 기준이므로 실행 위치에 의존하지 않습니다.

GPU 확인:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

GPU가 감지되지 않으면 설치된 NVIDIA 드라이버와 CUDA 환경에 맞는 PyTorch를 [공식 설치 안내](https://pytorch.org/get-started/locally/)에 따라 설치하세요. 특정 CUDA wheel 버전은 여기서 지정하지 않습니다. `torch`와 `torchvision`은 호환되는 조합으로 설치해야 합니다.

## Attention 해석

**Single-head attention**은 “이 layer의 이 head에서 CLS query가 어떤 key token에 주목하는가?”를 보여줍니다. 08은 CLS→CLS를 포함한 전체 행의 합이 약 1인지 확인합니다. CLS→patch만 떼어내면 그 합은 1보다 작을 수 있으며, 그림에도 재정규화하지 않은 raw attention weight를 사용합니다.

**Attention Rollout**은 여러 head/layer에 걸친 attention 흐름을 누적한 값입니다. 06의 `Relative rollout score`는 클래스별 기여도나 인과적 설명을 의미하지 않습니다. 08도 인과적 설명이 아닙니다. 두 실습은 모델에 입력된 tensor를 역정규화한 이미지 위에 맵을 표시해 전처리 좌표를 맞춥니다.

Hugging Face에서는 attention 행렬 반환을 위해 `attn_implementation="eager"`를 사용합니다. 빠른 attention backend에서는 attention이 반환되지 않을 수 있습니다. torchvision 실습은 forward hook으로 head별 weight 출력을 요청합니다.

## 참고 자료

- [ViT 논문](https://arxiv.org/abs/2010.11929)
- [Google Vision Transformer JAX 구현](https://github.com/google-research/vision_transformer)
- [torchvision ViT-B/16 및 전처리](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html)
- [Hugging Face ViT 모델](https://huggingface.co/google/vit-base-patch16-224)
- [Hugging Face Attention backend](https://huggingface.co/docs/transformers/attention_interface)
- [공부 기록 템플릿](notes/vit_summary.md)
