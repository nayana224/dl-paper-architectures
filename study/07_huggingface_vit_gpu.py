"""Hugging Face ViT의 GPU 추론, Hidden State, Attention shape를 관찰한다."""
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification

MODEL_ID = "google/vit-base-patch16-224"


def main():
    print("torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음 (CPU 사용)")
    assets = Path(__file__).resolve().parent / "assets"
    paths = sorted(assets.glob("pet_*.jpg"))
    if not paths:
        raise SystemExit("먼저 python study/00_prepare_dataset.py를 실행하세요.")
    image = Image.open(paths[0]).convert("RGB")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    # SDPA 대신 eager를 사용해야 attention 행렬을 직접 반환받을 수 있다.
    model = ViTForImageClassification.from_pretrained(
        MODEL_ID, attn_implementation="eager",
    ).to(device).eval()
    inputs = processor(images=image, return_tensors="pt").to(device)
    for name in ["hidden_size", "num_hidden_layers", "num_attention_heads", "patch_size", "image_size"]:
        print(f"{name}: {getattr(model.config, name)}")
    with torch.inference_mode():
        result = model(**inputs, output_hidden_states=True, output_attentions=True, return_dict=True)
    probabilities = result.logits.softmax(dim=-1)
    confidence, prediction = probabilities[0].max(dim=0)
    print("input shape:", list(inputs["pixel_values"].shape))
    print("logits shape:", list(result.logits.shape))
    print("prediction:", model.config.id2label[prediction.item()])
    print("confidence:", confidence.item())
    print("tokens: 197 = CLS 1 + 14×14 patches 196")
    print("hidden states 수 (embedding 출력 + 12개 layer 출력):", len(result.hidden_states))
    for index, hidden in enumerate(result.hidden_states):
        print(f"hidden_states[{index}]: {list(hidden.shape)}")
    if result.attentions is None or any(a is None for a in result.attentions):
        raise RuntimeError("Attention이 반환되지 않았습니다. eager 설정과 Transformers 버전을 확인하세요.")
    print("attention 수:", len(result.attentions))
    for index, attention in enumerate(result.attentions):
        print(f"attentions[{index}]: {list(attention.shape)}")
    print("confidence는 ImageNet softmax 점수이며 보정된 확률을 보장하지 않습니다.")


if __name__ == "__main__":
    main()
