# [Vision Transformer]

## 1. 이 논문이 해결하려는 문제

TODO: 이미지 인식에서 Transformer를 적용하려는 문제를 내 말로 정리한다.

## 2. 기존 방법의 한계

TODO: 논문이 논의하는 기존 접근의 한계와 근거를 기록한다.

## 3. 핵심 아이디어

이미지를 고정 크기 패치로 나누고 각 패치를 토큰으로 임베딩하여 Transformer Encoder에 입력한다.

## 4. 모델 구조

Patch Embedding → CLS + Position Embedding → Pre-LN Encoder → CLS representation → Classification Head.

실습에서 확인한 ViT-B/16 흐름: `[1,3,224,224]` → 패치 196개 → CLS 포함 197개 토큰 → Hidden State `[1,197,768]` → logits `[1,1000]`.

TODO: 이 흐름을 논문의 그림·수식과 연결한다.

## 5. Input / GT / Output / Loss

- Input: 이미지 tensor `[B,3,224,224]` (본 실습의 pretrained 모델 기준).
- GT: 분류 클래스 인덱스. Oxford-IIIT Pet 품종과 ImageNet 클래스는 다른 label space이다.
- Output: logits `[B,num_classes]`.
- Loss: 지도 분류 학습의 cross-entropy. 이 저장소는 추론·구조 관찰만 하며 학습하지 않는다.

## 6. 핵심 실험 결과

TODO: 논문에서 직접 확인한 dataset, 비교 조건, metric, 수치, 표·그림 번호를 함께 기록한다.

## 7. 내가 직접 한 실습

### 2026-10-06 실행 기록

사용자 실행 로그 기준으로 00~08을 완료했다. 아래 값은 논문 실험 결과가 아니라 이번 표본에 대한 관찰이다.

- Python 3.10 가상환경, PyTorch `2.14.1+cu130`.
- CUDA available: `True`, GPU: NVIDIA GeForce RTX 5070 Laptop GPU.
- 설치 중 pip 22.0.2의 `AssertionError`가 발생했다. pip 26.2.1로 업데이트 후 의존성 설치를 완료했다.
- 00: Oxford-IIIT Pet trainval에서 6개 표본과 `study/assets/manifest.csv` 생성.
- 01: `[3,224,224]` → `[3,14,14,16,16]` → `[14,14,3,16,16]` → `[196,3,16,16]` → `[196,768]`.
- 02: 무작위 Linear로 `[196,768]` → `[196,128]` 투영.
- 03: CLS 추가 후 `[197,128]`; CLS와 Position 모두 `requires_grad=True`.
- 04: 교육용 Encoder 출력 `[1,197,128]`, Attention `[1,4,197,197]`, CLS `[1,128]`. 학습은 수행하지 않았다.
- 06: torchvision에서 12개 layer의 Attention `[1,12,197,197]` 추출.
- 07: Hugging Face 설정 D=768, layer=12, head=12, patch=16, image=224 확인.
- 07: Hidden State 13개(embedding 출력 + 12개 layer 출력), 각각 `[1,197,768]`; Attention 12개, 각각 `[1,12,197,197]`.

### 05: torchvision ImageNet Top-1

| 이미지 | Pet 품종 (manifest) | ImageNet Top-1 | Softmax 점수 |
| --- | --- | --- | --- |
| pet_00.jpg | Abyssinian | Egyptian cat | 0.1785 |
| pet_05.jpg | Bengal | tabby | 0.4070 |
| pet_10.jpg | Chihuahua | Chihuahua | 0.8900 |
| pet_15.jpg | Great Pyrenees | Great Pyrenees | 0.8723 |
| pet_20.jpg | Maine Coon | tabby | 0.6131 |
| pet_25.jpg | Pug | pug | 0.8619 |

07의 Hugging Face 모델은 `pet_00.jpg`를 Egyptian cat, softmax 점수 `0.4496440887`로 예측했다. torchvision과 가중치·전처리가 다르므로 점수가 일치할 필요는 없다. Softmax 점수는 보정된 정답 확률을 보장하지 않는다.

## 8. Feature / Prediction Visualization

### 08: 마지막 layer의 head 0

Layer index 11, head index 0 (모두 0-based)의 CLS query 행을 확인했다.

- Attention tensor: `[1,12,197,197]`.
- CLS → CLS: `0.0004795285`.
- 전체 CLS attention 행의 합: `0.9999998808` (부동소수점 오차 범위에서 1).
- CLS → 패치 부분의 합: 약 `0.9995204`; CLS → CLS를 제외하므로 1보다 작다.

| 순위 | 패치 행 | 패치 열 | Raw attention weight |
| --- | --- | --- | --- |
| 1 | 11 | 12 | 0.04773250 |
| 2 | 12 | 9 | 0.04678351 |
| 3 | 3 | 13 | 0.04351560 |
| 4 | 7 | 4 | 0.04242747 |
| 5 | 7 | 3 | 0.04158621 |
| 6 | 2 | 13 | 0.04077305 |
| 7 | 12 | 13 | 0.03297426 |
| 8 | 11 | 11 | 0.03144066 |
| 9 | 1 | 7 | 0.02760664 |
| 10 | 10 | 11 | 0.02340090 |

이 가중치는 해당 head가 CLS 업데이트를 위해 각 토큰의 Value를 섞는 비율이다. 클래스 예측의 기여율로 해석하지 않는다. 마지막 layer의 패치 표현에는 앞선 layer에서 다른 위치의 정보도 이미 섞여 있다.

### 생성된 그림과 해석할 때의 기준

로컬 `study/outputs/`에 다음 파일이 생성됐으며 git에는 포함하지 않는다.

- `06_attention_rollout.png`: head 평균 + residual + 행 정규화를 적용한 attention을 12개 layer에 걸쳐 누적한 결과.
- `08_layer11_head0_heatmap.png`: 특정 layer/head의 CLS → 패치 raw attention.
- `08_layer11_head0_overlay.png`: 같은 raw attention을 실제 모델 입력 이미지에 겹친 결과.

Single-head attention과 Rollout은 서로 다른 관찰 방법이며 둘 다 인과적 설명은 아니다. 06과 08은 pretrained 모델도 다르므로 그림 차이를 누적 방법만의 영향으로 단정할 수 없다.

TODO: 그림을 직접 보고 밝은 패치가 동물의 어느 부위 또는 배경과 겹치는지 기록한다. 아직 위치에 대한 시각적 판단은 기록하지 않았다.

## 9. 틀린 사례와 원인 추정

관찰: `pet_00.jpg`의 Pet 정답은 Abyssinian이고 두 pretrained 모델의 ImageNet Top-1은 Egyptian cat이었다. Label space가 다르므로 이 차이만으로 Pet 품종 분류의 오답이라고 판정하지 않는다.

TODO: 이미지와 전체 예측 목록을 확인하고, 가설을 기록한다. Attention이 배경에 집중하더라도 그것만으로 예측 실패의 원인으로 단정하지 않는다.

## 10. 이 논문에서 내가 가져갈 한 문장

TODO: 공부 후 내 문장으로 정리한다.
