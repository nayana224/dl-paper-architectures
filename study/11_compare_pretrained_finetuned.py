"""같은 Pet 이미지를 ImageNet pretrained 모델과 Pet fine-tuned 모델에 넣어 비교한다.

목적:
- Pretrained 모델: 원래 label space(ImageNet)에서 무엇을 예측하는가?
- Fine-tuned 모델: downstream label space(Oxford-IIIT Pet)에서 무엇을 예측하는가?

10을 실행해서 만든 로컬 checkpoint가 있으면 그것을 우선 사용한다.
아직 10을 실행하지 않았다면 공개 Oxford-IIIT Pet fine-tuned checkpoint를 사용한다.
"""

import csv
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent

IMAGENET_MODEL_ID = "google/vit-base-patch16-224"
LOCAL_FINETUNED = REPO_ROOT / "checkpoints" / "vit-base-oxford-pet"
PUBLIC_FINETUNED = "schlenat/vit-base-oxford-iiit-pets"


def load_manifest():
    manifest_path = ROOT / "assets" / "manifest.csv"
    if not manifest_path.exists():
        return {}

    with manifest_path.open(encoding="utf-8") as file:
        return {
            row["filename"]: row["breed"]
            for row in csv.DictReader(file)
        }


def predict_top1(model, processor, image, device):
    """한 이미지에 대해 Top-1 label과 softmax score를 반환한다."""
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = logits.softmax(dim=-1)[0]

    index = probabilities.argmax().item()
    return model.config.id2label[index], probabilities[index].item()


def main():
    image_paths = sorted((ROOT / "assets").glob("pet_*.jpg"))
    if not image_paths:
        raise SystemExit("먼저 python study/00_prepare_dataset.py를 실행하세요.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # A) ImageNet 분류용 pretrained 모델.
    imagenet_processor = AutoImageProcessor.from_pretrained(IMAGENET_MODEL_ID)
    imagenet_model = AutoModelForImageClassification.from_pretrained(
        IMAGENET_MODEL_ID
    ).to(device).eval()

    # B) Oxford-IIIT Pet으로 fine-tuning된 모델.
    # 직접 10을 실행했다면 로컬 checkpoint를 사용한다.
    # 그렇지 않으면 공개 fine-tuned checkpoint로 비교 실습을 할 수 있다.
    if LOCAL_FINETUNED.exists():
        finetuned_source = str(LOCAL_FINETUNED)
        source_description = "직접 fine-tuning한 local checkpoint"
    else:
        finetuned_source = PUBLIC_FINETUNED
        source_description = "공개 Oxford-IIIT Pet fine-tuned checkpoint"

    pet_processor = AutoImageProcessor.from_pretrained(finetuned_source)
    pet_model = AutoModelForImageClassification.from_pretrained(
        finetuned_source
    ).to(device).eval()

    print("ImageNet model:", IMAGENET_MODEL_ID)
    print("Pet model:", finetuned_source)
    print("Pet model source:", source_description)
    print()

    ground_truth = load_manifest()

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")

        imagenet_label, imagenet_score = predict_top1(
            imagenet_model,
            imagenet_processor,
            image,
            device,
        )
        pet_label, pet_score = predict_top1(
            pet_model,
            pet_processor,
            image,
            device,
        )

        print(image_path.name)
        if image_path.name in ground_truth:
            print("GT                  :", ground_truth[image_path.name])
        print(
            f"ImageNet pretrained : {imagenet_label} "
            f"({imagenet_score:.4f})"
        )
        print(
            f"Pet fine-tuned      : {pet_label} "
            f"({pet_score:.4f})"
        )
        print()

    print("해석:")
    print("- 두 모델은 같은 ViT 계열이지만 classification task와 label space가 다르다.")
    print("- fine-tuning은 pretrained representation을 downstream Pet 분류에 맞게 조정한다.")
    print("- softmax score 자체를 보정된 정답 확률로 해석하지 않는다.")


if __name__ == "__main__":
    main()
