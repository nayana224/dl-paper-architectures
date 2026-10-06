"""이미 Oxford-IIIT Pet으로 fine-tuning된 ViT를 불러와 추론한다.

05에서는 ImageNet 분류용 ViT에 Pet 이미지를 넣었다.
09에서는 Oxford-IIIT Pet 37개 품종에 맞게 fine-tuning된 ViT를 사용한다.

주의:
- 아래 Hugging Face checkpoint는 ViT 논문 저자들이 배포한 공식 Oxford-Pet checkpoint가 아니다.
- fine-tuning 결과를 먼저 체감하기 위한 공개 학습 checkpoint이다.
"""

import csv
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


ROOT = Path(__file__).resolve().parent

# 공개된 Oxford-IIIT Pet fine-tuned ViT checkpoint.
# base model은 google/vit-base-patch16-224이다.
MODEL_ID = "schlenat/vit-base-oxford-iiit-pets"


def load_ground_truth():
    """00에서 만든 manifest를 읽어 파일명 -> Pet 품종 이름으로 변환한다."""
    manifest = ROOT / "assets" / "manifest.csv"
    if not manifest.exists():
        return {}

    with manifest.open(encoding="utf-8") as file:
        return {
            row["filename"]: row["breed"]
            for row in csv.DictReader(file)
        }


def main():
    image_paths = sorted((ROOT / "assets").glob("pet_*.jpg"))
    if not image_paths:
        raise SystemExit("먼저 python study/00_prepare_dataset.py를 실행하세요.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Fine-tuned model:", MODEL_ID)

    # Processor는 resize / normalize 등 모델이 기대하는 입력 전처리를 수행한다.
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)

    # 이 모델의 classification head는 Oxford-IIIT Pet의 class에 맞게 이미 학습되어 있다.
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model = model.to(device).eval()

    ground_truth = load_ground_truth()

    print("num_labels:", model.config.num_labels)
    print()

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")

        # PIL 이미지를 [1, 3, 224, 224] pixel_values tensor로 변환한다.
        inputs = processor(images=image, return_tensors="pt").to(device)

        # inference이므로 gradient를 계산하지 않는다.
        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = logits.softmax(dim=-1)[0]

        values, indices = probabilities.topk(min(5, model.config.num_labels))

        print(f"{image_path.name}")
        if image_path.name in ground_truth:
            print("GT (Oxford-IIIT Pet):", ground_truth[image_path.name])

        for rank, (value, index) in enumerate(zip(values, indices), 1):
            label = model.config.id2label[index.item()]
            print(f"{rank}: {label} {value.item():.4f}")
        print()

    print("05와의 차이:")
    print("- 05: ImageNet label space에서 추론")
    print("- 09: Oxford-IIIT Pet용으로 fine-tuning된 label space에서 추론")


if __name__ == "__main__":
    main()
