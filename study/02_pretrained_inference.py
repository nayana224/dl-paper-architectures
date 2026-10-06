"""이미 학습된 Google ViT를 불러와 ImageNet Top-5 inference만 수행한다.
모델 카드: https://huggingface.co/google/vit-base-patch16-224
ImageNet-21k pre-training -> ImageNet-1k fine-tuning을 거친 checkpoint이다.
이 코드는 pre-training이나 fine-tuning을 직접 수행하지 않는다.
"""
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "google/vit-base-patch16-224"


def main():
    assets = Path(__file__).resolve().parents[1] / "assets"
    paths = sorted(assets.glob("pet_*.jpg"))
    if not paths:
        raise SystemExit("assets/에 pet_*.jpg 표본 이미지를 준비하세요.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("model id:", MODEL_ID)
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID).to(device).eval()
    for name in ["image_size", "patch_size", "hidden_size", "num_hidden_layers", "num_attention_heads"]:
        print(f"{name}: {getattr(model.config, name)}")
    image = Image.open(paths[0]).convert("RGB")
    print("Image:", paths[0].name, "원본 크기:", image.size)
    inputs = processor(images=image, return_tensors="pt").to(device)
    print("processor 입력 tensor:", list(inputs["pixel_values"].shape))
    # eval은 학습용 동작을 끄고 inference_mode는 gradient 계산을 끈다.
    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = logits.softmax(dim=-1)[0]
    print("logits shape:", list(logits.shape))
    print("Top-5 prediction (ImageNet label space):")
    values, indices = probabilities.topk(5)
    for rank, (value, index) in enumerate(zip(values, indices), 1):
        print(f"{rank}: {model.config.id2label[index.item()]} {value.item():.4f}")
    print("Softmax 점수는 보정된 정답 확률을 보장하지 않습니다.")


if __name__ == "__main__":
    main()
