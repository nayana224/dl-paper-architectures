# ACT Figure 4 — 코드로 아키텍처 읽기

이 문서는 Zhao et al. (2023), **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware**, Section IV 및 Figure 4를 공부하기 위한 자료입니다. 기존 `04_act_forward.py`는 소형 입문 예제로 유지하고, `05_act_architecture.py`에서 보다 상세한 구조를 확인합니다.

## 실행

저장소 루트에서:

```bash
python study/act/05_act_architecture.py
python study/act/05_act_architecture.py --train-step
# 논문 그림에 근접한 hidden/layer 설정 (상당히 느리고 메모리 사용량 큼)
python study/act/05_act_architecture.py --hidden 512 --layers 4 --decoder-layers 7 --image-size 64
```

필요 패키지: `torch`, `torchvision`. CPU에서도 실행할 수 있지만 ResNet18 4개 카메라와 Transformer의 역전파는 느릴 수 있습니다.

## 모듈과 논문 Figure 4의 대응

| 그림 | 구현 | 의미 | Shape (B=1, K=8, D=128 기본 예제) |
|---|---|---|---|
| Step 1: 4 RGB Images | `images` | top / front / left wrist / right wrist에 해당하는 4-view 배열(여기서는 랜덤 데이터) | [1,4,3,64,64] |
| Step 1: joints | `joints` | 양팔 관절+그리퍼의 현재 상태 | [1,14] |
| Step 1: action sequence | `gt` | 현재 시점부터 K개의 목표 관절 행동(GT) | [1,8,14] |
| Step 2: CLS | `ActionLatentEncoder.cls` | 잠재 스타일 인코딩용 집계 토큰 | [1,1,128] |
| Step 2: joint linear | `ActionLatentEncoder.joints` | 관절 상태를 임베딩 | [1,1,128] |
| Step 2: action linear | `ActionLatentEncoder.actions` | GT actions 임베딩 | [1,8,128] |
| Step 2: Transformer encoder | `ActionLatentEncoder.encoder` | [CLS, joint, actions...] 문맥화 | [1,10,128] |
| Step 2: posterior | `mu / logvar` | 32차원 정규분포 매개변수 | 각각 [1,32] |
| Step 2: sampled z | `z` | reparameterization | [1,32] |
| Step 3: ResNet18 | `MultiViewResNet.features` | 카메라별 동일 가중치의 convolutional feature extractor | [4,512,h,w] |
| Step 3: projected visual tokens | `MultiViewResNet.project` | CNN 공간 특징을 D차원으로 변환 + 2D 위치 임베딩 | [1,4*h*w,128] |
| Step 3: joints & latent tokens | `joint_token / latent_token` | 현재 관절과 z를 조건에 추가 | [1,1,128] 각각 |
| Step 3: policy Transformer encoder | `policy_encoder` | 이미지/관절/z Token 통합 | [1,4*h*w+2,128] |
| Step 3: action query | `query_positions` | 미래 K개 시점에 대응하는 Query | [1,8,128] |
| Step 3: policy Transformer decoder | `policy_decoder` | Query가 Encoder memory에 cross-attention | [1,8,128] |
| Step 3: action head | `action_head` | K개 타임스텝의 14차원 행동 예측 | [1,8,14] |

**카메라 순서:** 이 예제에서 camera axis의 index 0~3은 각각 top, front, left wrist, right wrist로 해석합니다. 실제 모델은 카메라별 이미지를 정해진 순서로 제공해야 합니다. 이미지 순서 자체가 학습되는 것은 아닙니다.

**14차원:** ALOHA 양팔 각각 6개 arm joints + 1개 gripper 값, 총 14개입니다. Action은 각 시점의 목표 관절 상태이며, velocity 명령이나 언어 토큰이 아닙니다.

## 먼저 봐야 할 코드 순서

1. `main()`: `images, joints, gt`를 어떻게 생성하는지 확인
2. `ActionLatentEncoder.forward()`: GT를 포함해 `mu, logvar, z`를 계산하는 과정 확인
3. `MultiViewResNet.forward()`: `[B,C,3,H,W] → [B*C,3,H,W] → [B,C*h*w,D]` 추적
4. `ACTPolicy.forward()`: 학습일 때와 추론일 때 다른 분기 확인
5. `act_loss()`: reconstruction(L1) + beta × KL 확인
6. `temporal_ensemble()`: 겹치는 Chunk가 같은 시각에 대해 낸 출력을 조합

## 핵심 질문

- 왜 GT action은 **CVAE encoder**에만 들어가고 Policy decoder에 직접 들어가지 않을까?
- 왜 camera feature는 classifier logits가 아니라 **spatial tokens**여야 할까?
- Transformer decoder의 K개 action query는 **GT의 임베딩**일까, **정해진 위치 쿼리**일까?
- 추론할 때 GT도 posterior encoder도 없다면 z는 무엇으로 설정할까?
- 동일한 실행 시각에 여러 과거 Chunk가 대응한다면 어떤 행동을 사용할까?

## 교육용 모델과 논문의 차이

- 코드의 기본값은 실행 부담을 낮춘 `D=128`, Encoder 2층, Decoder 2층입니다. 명령어로 논문 Figure 4의 512 / 4층 / 7층 구조에 접근할 수 있습니다.
- 표준 torchvision ResNet18은 마지막 convolution feature가 512채널입니다. 원문 도식의 구체적인 CNN output 차원과 동일하다고 가정하지 않습니다. `1x1 Conv` projection과 2D sinusoidal position을 사용합니다.
- 원 논문의 데이터 증강, 실제 ALOHA 데이터 전처리, optimizer/scheduler 구성, pretrained CNN 활용 여부, 정밀한 Transformer 세부 구현까지 복제한 공식 구현은 아닙니다.
- 학습 샘플은 완전히 합성된 Tensor입니다. Loss나 output이 작다고 실제 로봇의 행동이 올바르다는 의미가 아닙니다.
- `--train-step`은 역전파 연결 확인용 **한 번의 업데이트**입니다. 장시간 학습, 체크포인트 저장, 로봇 실행은 제공하지 않습니다.
- `temporal_ensemble`은 논문의 최신/과거 예측 가중 결합 아이디어를 이해하는 최소 예제입니다. 예측 이력 버퍼의 실시간 관리, 제어 주파수, 안전 제약은 구현하지 않았습니다.

## 최소 실습 과제

1. `--chunk-size 4`와 `--chunk-size 10`에서 CVAE Token 수와 decoder Query 수가 어떻게 달라지는지 출력 비교.
2. `--train-step` 전후로 `GT`의 Shape, Prediction의 Shape, Gradient가 있는 모듈을 확인.
3. `temporal_ensemble`의 `decay`를 0과 1로 바꾸고 최신 예측의 가중치가 왜 달라지는지 설명.
4. 카메라 네 장의 입력 중 하나를 바꾸고 동일한 `z=0` 조건에서 출력이 달라지는지 확인(랜덤 모델의 값은 의미가 없으며 단지 연결 검증).
