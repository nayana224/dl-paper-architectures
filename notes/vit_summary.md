# [Vision Transformer]

## 1. 이 논문이 해결하려는 문제

TODO: 이미지 인식에서 Transformer를 적용하려는 문제를 내 말로 정리한다.

## 2. 기존 방법의 한계

TODO: 논문이 논의하는 기존 접근의 한계와 근거를 기록한다.

## 3. 핵심 아이디어

이미지를 고정 크기 패치로 나누고 각 패치를 토큰으로 임베딩하여 Transformer Encoder에 입력한다.

## 4. 모델 구조

Patch Embedding → CLS + Position Embedding → Pre-LN Encoder → CLS representation → Classification Head.

TODO: ViT-B/16 shape를 직접 써 보고 논문의 그림·수식과 연결한다.

## 5. Input / GT / Output / Loss

- Input: 이미지 tensor `[B,3,224,224]` (본 실습의 pretrained 모델 기준).
- GT: 분류 클래스 인덱스. Oxford-IIIT Pet 품종과 ImageNet 클래스는 다른 label space이다.
- Output: logits `[B,num_classes]`.
- Loss: 지도 분류 학습의 cross-entropy. 이 저장소는 추론·구조 관찰만 하며 학습하지 않는다.

## 6. 핵심 실험 결과

TODO: 논문에서 직접 확인한 dataset, 비교 조건, metric, 수치, 표·그림 번호를 함께 기록한다.

## 7. 내가 직접 한 실습

TODO: 실행한 00~08 스크립트, 환경, tensor shape와 관찰을 기록한다.

## 8. Feature / Prediction Visualization

TODO: Top-5 예측과 attention 그림을 기록한다. Single-head attention과 Attention Rollout을 구분하고 인과적 설명으로 단정하지 않는다.

## 9. 틀린 사례와 원인 추정

TODO: 실제 관찰과 가설을 구분한다. Label space 차이를 오분류로 혼동하지 않는다.

## 10. 이 논문에서 내가 가져갈 한 문장

TODO: 공부 후 내 문장으로 정리한다.
