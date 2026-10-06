# ViT PyTorch Study

ViT 논문을 이해하기 위해 세 가지 흐름만 실습합니다. 논문 전체 재현이 아니라 개인 GPU에서 핵심 학습·추론 흐름을 이해하는 저장소입니다.

## 세 가지 실습

**01_forward_backward.py — “ViT는 어떻게 학습되는가?”**

```text
Image
↓
ViT
↓
Logits
↓
Loss
↓
Backward
↓
Weight Update
```

Pretrained 모델 없이 작은 ViT(D=128, heads=4, Encoder 1개, class 3개)를 직접 구현합니다. Ball 이미지를 메모리에서 그리고 가상 class `0=Bird, 1=Ball, 2=Car` 중 GT=1을 지정해 정확히 1 step 학습합니다. 모든 중간 shape, logits, softmax, loss, gradient와 weight 변경을 출력합니다. 분류 성능을 평가하는 실험은 아닙니다.

**02_pretrained_inference.py — “Pretrained ViT란 무엇인가?”**

```text
Large Dataset
↓
Pre-training
↓
Pretrained ViT
↓
Inference
```

`google/vit-base-patch16-224`는 **ImageNet-21k pre-training → ImageNet-1k fine-tuning**을 거친 checkpoint입니다. 위 그림은 개념 흐름이며, 이 파일은 이미 두 학습 단계를 마친 weight로 첫 Pet 표본의 ImageNet Top-5를 출력할 뿐입니다. 모델 설정은 image=224, patch=16, D=768, layers=12, heads=12입니다. [Google 모델 카드](https://huggingface.co/google/vit-base-patch16-224)

**03_finetuned_inference.py — “Fine-tuning하면 무엇이 달라지는가?”**

```text
Pretrained ViT
↓
Downstream Dataset
↓
Fine-tuning
↓
Task-specific prediction
```

같은 Pet 이미지 6장을 A: Google ImageNet 1,000-class 모델, B: Oxford-IIIT Pet 37-class 모델에 넣어 GT·Top-1을 비교합니다. 공개 `schlenat/vit-base-oxford-iiit-pets`는 A를 Pet 데이터셋으로 추가 fine-tuning한 **사용자 공개 모델이며 ViT 논문 저자의 공식 Oxford-Pet checkpoint가 아닙니다**. 이 파일도 추가 학습 없이 추론만 수행합니다. 각 모델의 processor를 사용합니다. [Pet 모델 카드](https://huggingface.co/schlenat/vit-base-oxford-iiit-pets)

두 모델의 label space가 다르므로 점수를 그대로 비교하거나 ImageNet 예측을 Pet 품종 정답과 일대일 비교하지 않습니다. 표본은 Oxford-IIIT Pet trainval에서 골랐으며 공개 모델 학습 이미지와 겹칠 수 있습니다. 이 비교는 test accuracy 측정이 아닙니다.

## 논문과 연결

논문의 pre-training dataset 예는 ImageNet, ImageNet-21k, JFT-300M입니다. Downstream dataset 예는 ImageNet, CIFAR-10, CIFAR-100, Oxford-IIIT Pets, Oxford Flowers-102, VTAB입니다. 같은 dataset도 실험 조건에 따라 pre-training 또는 downstream 역할을 할 수 있습니다. 여기서는 대규모 pre-training이나 전체 Pet fine-tuning을 직접 실행하지 않습니다. [ViT 논문](https://arxiv.org/html/2010.11929v2)

## 핵심 용어

| 용어 | 의미 |
| --- | --- |
| Pre-training | 큰 dataset에서 먼저 학습 |
| Pretrained model | Pre-training이 끝난 weight를 가진 모델 |
| Inference | Parameter update 없이 prediction만 수행 |
| Fine-tuning | Pretrained weight를 downstream task에 맞게 추가 학습 |
| Logits | Classification Head의 raw class score. 확률이 아님 |
| Softmax | Logits를 합이 1인 class별 상대 score로 변환. 보정된 정답 확률은 보장하지 않음 |
| Loss | 예측과 GT의 차이를 scalar로 계산. 여기서는 CrossEntropyLoss에 logits와 정수 GT를 입력 |
| Backward | Parameter별 gradient 계산 |
| Optimizer step | Gradient를 이용해 weight update |

## 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python study/01_forward_backward.py
python study/02_pretrained_inference.py
python study/03_finetuned_inference.py
```

Ubuntu 22.04에서는 오래된 pip의 설치 오류를 피하도록 pip부터 업데이트합니다. CUDA가 감지되면 GPU를, 아니면 CPU를 자동 사용합니다. GPU가 감지되지 않으면 드라이버 환경에 맞는 PyTorch·torchvision을 [공식 설치 안내](https://pytorch.org/get-started/locally/)에 따라 설치하세요.

`assets/`의 기존 Pet 이미지 6장과 manifest를 재사용하므로 전체 dataset을 다운로드하지 않습니다. 02·03은 첫 실행에 필요한 모델 weight를 Hugging Face 캐시에 다운로드하며 이후 재사용합니다. 03은 GPU 메모리 사용을 줄이도록 모델을 하나씩 실행합니다.

기존 00~11 스크립트는 위 세 파일로 대체했습니다. 기존 로컬 데이터·출력 그림·checkpoint는 보존하되 새 실습에서는 사용하지 않습니다. `notes/vit_summary.md`는 이전 구조에서의 실습 기록으로 보존합니다.
