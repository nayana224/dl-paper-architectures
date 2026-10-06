"""같은 Pet 표본에서 ImageNet 모델과 Pet fine-tuned 모델의 예측을 비교한다.
Fine-tuning: pretrained representation을 downstream task에 맞게 추가 학습한다.
이 파일은 이미 fine-tuning된 weight를 불러오며 추가 학습하지 않는다.
공개 Pet checkpoint는 ViT 논문 저자의 공식 Oxford-Pet checkpoint가 아니다.
모델 카드: https://huggingface.co/schlenat/vit-base-oxford-iiit-pets
"""
import csv
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

IMAGENET_MODEL_ID = "google/vit-base-patch16-224"
PET_MODEL_ID = "schlenat/vit-base-oxford-iiit-pets"


def main():
    assets = Path(__file__).resolve().parents[1] / "assets"
    with (assets / "manifest.csv").open(encoding="utf-8", newline="") as file:
        samples = list(csv.DictReader(file))[:6]
    if len(samples) < 3:
        raise SystemExit("assets/manifest.csv에 최소 3장의 filename, breed 정보를 준비하세요.")
    # 다운로드 전에 표본과 GT가 준비됐는지 확인한다.
    images = [(row, Image.open(assets / row["filename"]).convert("RGB")) for row in samples]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    predictions = {}
    # 두 모델을 순서대로 실행해 GPU에 두 모델의 weight를 동시에 올리지 않는다.
    for title, model_id in [("Pretrained model", IMAGENET_MODEL_ID), ("Fine-tuned model", PET_MODEL_ID)]:
        print(f"\n[{title}] model id: {model_id}")
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id).to(device).eval()
        print("num_labels:", model.config.num_labels)
        if model_id == PET_MODEL_ID and model.config.num_labels != 37:
            raise RuntimeError("Oxford-IIIT Pet 37-class checkpoint인지 확인하세요.")
        predictions[title] = []
        for row, image in images:
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.inference_mode():
                logits = model(**inputs).logits
                probabilities = logits.softmax(dim=-1)[0]
            index = probabilities.argmax().item()
            predictions[title].append((model.config.id2label[index], probabilities[index].item()))
        del inputs, logits, probabilities, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for index, (row, image) in enumerate(images):
        print(f"\nImage: {row['filename']}\nGT: {row['breed']}")
        for title in ["Pretrained model", "Fine-tuned model"]:
            label, score = predictions[title][index]
            print(f"[{title}]\nPrediction: {label} (softmax={score:.4f})")
    # A는 ImageNet 1000개 class, B는 Oxford-IIIT Pet 37개 품종 label space이다.
    print("\nA: ImageNet label space / B: Oxford-IIIT Pet 37-class label space")
    print("두 모델의 label space가 달라 softmax 점수를 직접 성능 비교로 해석하지 않습니다.")
    print("표본은 trainval에서 선택됐습니다. 공개 모델 학습 표본과 겹칠 수 있어 test accuracy가 아닙니다.")


if __name__ == "__main__":
    main()
