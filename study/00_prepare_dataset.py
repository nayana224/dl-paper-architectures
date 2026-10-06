"""Oxford-IIIT Pet trainval에서 고정된 6개 품종의 첫 이미지를 선택한다."""
import csv
from pathlib import Path
from torchvision.datasets import OxfordIIITPet

ROOT = Path(__file__).resolve().parent


def main():
    dataset = OxfordIIITPet(ROOT / "data", split="trainval", download=True)
    assets = ROOT / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    # 데이터셋 클래스 순서에 따른 고정된 표본; 매번 동일한 인덱스를 고른다.
    selected_classes = {0, 5, 10, 15, 20, 25}
    rows = []
    for index in range(len(dataset)):
        image, label = dataset[index]
        if label not in selected_classes:
            continue
        breed = dataset.classes[label]
        filename = f"pet_{label:02d}.jpg"
        image.convert("RGB").save(assets / filename)
        rows.append({"filename": filename, "dataset_index": index,
                     "pet_label": label, "breed": breed})
        selected_classes.remove(label)
        if not selected_classes:
            break
    with (assets / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["filename", "dataset_index", "pet_label", "breed"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)}개 이미지와 manifest.csv 저장: {assets}")


if __name__ == "__main__":
    main()
