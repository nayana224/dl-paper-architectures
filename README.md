# ViT PyTorch Study

논문 **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**를 이해하기 위한 개인 구현·실습 저장소입니다. Google의 JAX 구현을 참고 자료로 삼되, 실험하기 쉬운 PyTorch를 사용합니다. Pretrained 실험은 torchvision과 Hugging Face Transformers를 사용합니다. 학습 프레임워크나 논문 전체 결과 재현 프로젝트는 아닙니다.

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
| `study/05_pretrained_inference.py` | torchvision ViT-B/16으로 Pet 이미지를 **ImageNet label space**에서 Top-5 추론합니다. 아직 Pet fine-tuning은 하지 않습니다. |
| `study/06_attention_visualization.py` | torchvision의 head 평균 attention에 residual을 더하고 layer별로 누적한 Attention Rollout을 시각화합니다. |
| `study/07_huggingface_vit_gpu.py` | `google/vit-base-patch16-224`를 GPU에서 실행하고 ViT-B/16의 설정, Hidden State, Attention shape를 확인합니다. |
| `study/08_attention_inspection.py` | 하나의 layer/head에서 CLS query의 raw attention을 직접 수치와 heatmap으로 확인합니다. |
| `study/09_finetuned_pet_inference.py` | 이미 Oxford-IIIT Pet으로 fine-tuning된 공개 ViT checkpoint를 불러와 **37개 Pet class**에서 추론합니다. |
| `study/10_finetune_oxford_pet.py` | ImageNet-21k pretrained ViT-B/16을 불러와 Oxford-IIIT Pet으로 직접 full fine-tuning하고 checkpoint를 저장합니다. |
| `study/11_compare_pretrained_finetuned.py` | 같은 이미지를 ImageNet pretrained 모델과 Pet fine-tuned 모델에 넣어 label space와 prediction의 차이를 비교합니다. |

01~04는 구조를 이해하기 위한 무작위 초기화 실습입니다. 05~09는 pretrained/fine-tuned 모델의 추론 및 내부 관찰이고, 10에서 처음으로 실제 downstream fine-tuning을 수행합니다.

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

Hugging Face 예제의 입력은 `[1,3,224,224]`, Hidden State는 `[1,197,768]`, Attention은 `[1,12,197,197]`입니다. Hidden State는 embedding 출력과 12개 layer 출력을 합쳐 13개, Attention은 12개입니다.

## Pre-training → Fine-tuning 흐름

ViT 논문의 중요한 실험 전략을 이 저장소에서는 다음처럼 단순화해 체험합니다.

```text
[논문의 아이디어]

대규모 데이터셋
(ImageNet-21k / JFT-300M)
        ↓
ViT pre-training
        ↓
downstream task용 classification head
        ↓
Oxford-IIIT Pet 등에서 fine-tuning


[이 저장소의 10번 실습]

google/vit-base-patch16-224-in21k
(ImageNet-21k pretrained ViT-B/16)
        ↓
37-class Oxford-IIIT Pet head
        ↓
Oxford-IIIT Pet train/validation
        ↓
CrossEntropyLoss + AdamW
        ↓
전체 ViT fine-tuning
        ↓
Oxford-IIIT Pet test
        ↓
checkpoints/vit-base-oxford-pet
```

### 05와 09의 차이

```text
05:
Pet 이미지
→ ImageNet 분류용 pretrained ViT
→ 1000개 ImageNet label 중 하나 예측

09:
Pet 이미지
→ Oxford-IIIT Pet으로 이미 fine-tuning된 ViT
→ 37개 Pet 품종 중 하나 예측
```

09의 `schlenat/vit-base-oxford-iiit-pets`는 ViT 논문 저자들이 배포한 공식 Oxford-Pet checkpoint가 아니라, fine-tuned 모델의 결과를 먼저 체감하기 위한 공개 Hugging Face checkpoint입니다.

10은 논문 전체 실험을 그대로 재현하는 것이 아니라, **대규모 pre-training → 작은 downstream dataset fine-tuning**이라는 핵심 transfer learning 흐름을 개인 GPU에서 직접 체험하는 실습입니다.

## 10번 Fine-tuning의 Input / GT / Output / Loss

```text
Input
[B,3,224,224]

GT
[B]
각 값은 0~36의 Oxford-IIIT Pet class index

ViT
↓
CLS representation [B,768]
↓
37-class classification head

Output
[B,37] logits

Loss
CrossEntropyLoss(logits, GT)

Backpropagation
↓
pretrained ViT 전체 parameter + 새 head 업데이트
```

## 실행 방법

Ubuntu 22.04와 NVIDIA GPU를 목표로 하며 GPU가 없으면 CPU로도 일부 실습을 실행할 수 있습니다.

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

# 이미 fine-tuning된 Pet 모델의 결과를 먼저 확인
python study/09_finetuned_pet_inference.py

# 먼저 작은 sample로 fine-tuning pipeline이 정상 동작하는지 확인
python study/10_finetune_oxford_pet.py \
    --epochs 1 \
    --batch-size 8 \
    --max-train-samples 128 \
    --max-test-samples 128

# 문제가 없으면 전체 dataset으로 fine-tuning
python study/10_finetune_oxford_pet.py \
    --epochs 3 \
    --batch-size 8

# 직접 학습한 checkpoint가 있으면 자동으로 사용하고,
# 없으면 공개 fine-tuned checkpoint를 사용
python study/11_compare_pretrained_finetuned.py
```

GPU 메모리가 부족하면 `--batch-size 4` 또는 `--batch-size 2`로 낮추세요. 10은 GPU에서 mixed precision을 사용합니다.

Ubuntu 22.04에서 새 가상환경의 pip가 오래된 버전이라면 의존성 설치 오류가 발생할 수 있으므로 먼저 `python -m pip install --upgrade pip`를 실행하세요.

00은 Oxford-IIIT Pet 데이터셋을 내려받으므로 네트워크와 저장 공간이 필요합니다. pretrained/fine-tuned 모델의 첫 실행에서는 Hugging Face 또는 torchvision model weight를 다운로드합니다.

`study/assets/`의 선택된 이미지와 manifest는 버전 관리할 수 있습니다. 전체 데이터셋 `study/data/`, 생성 이미지 `study/outputs/`, fine-tuned weight `checkpoints/`는 git에서 제외됩니다.

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

GPU가 감지되지 않으면 설치된 NVIDIA 드라이버와 CUDA 환경에 맞는 PyTorch를 공식 설치 안내에 따라 설치하세요. 특정 CUDA wheel 버전은 저장소에서 고정하지 않습니다.

## Attention 해석

**Single-head attention**은 “이 layer의 이 head에서 CLS query가 어떤 key token에 주목하는가?”를 보여줍니다. 08은 CLS→CLS를 포함한 전체 행의 합이 약 1인지 확인합니다.

**Attention Rollout**은 여러 head/layer에 걸친 attention 흐름을 누적한 값입니다. 06의 `Relative rollout score`는 클래스별 기여도나 인과적 설명을 의미하지 않습니다. 08도 인과적 설명이 아닙니다.

Hugging Face에서는 attention 행렬 반환을 위해 `attn_implementation="eager"`를 사용합니다. torchvision 실습은 forward hook으로 head별 weight 출력을 요청합니다.

## 참고 자료

- [ViT 논문](https://arxiv.org/abs/2010.11929)
- [Google Vision Transformer JAX 구현](https://github.com/google-research/vision_transformer)
- [Google ImageNet-21k pretrained ViT-B/16](https://huggingface.co/google/vit-base-patch16-224-in21k)
- [Google ImageNet-1k classification ViT-B/16](https://huggingface.co/google/vit-base-patch16-224)
- [공개 Oxford-IIIT Pet fine-tuned ViT](https://huggingface.co/schlenat/vit-base-oxford-iiit-pets)
- [torchvision ViT-B/16](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html)
- [공부 기록 템플릿](notes/vit_summary.md)
