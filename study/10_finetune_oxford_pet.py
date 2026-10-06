"""ImageNet-21k pretrained ViT-B/16을 Oxford-IIIT Pet으로 직접 fine-tuning한다.

논문의 transfer learning 흐름을 작은 규모로 체험하는 실습이다.

ImageNet-21k pretrained ViT
    -> Oxford-IIIT Pet용 37-class classification head
    -> Oxford-IIIT Pet train/validation
    -> Cross Entropy Loss
    -> 전체 ViT parameter fine-tuning
    -> test accuracy
    -> checkpoint 저장

논문 전체 재현이 아니라, "대규모 pre-training -> downstream fine-tuning" 흐름을
PyTorch/Hugging Face에서 직접 확인하는 것이 목적이다.
"""

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import OxfordIIITPet
from transformers import AutoImageProcessor, ViTForImageClassification


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent

# google/vit-base-patch16-224와 달리 이 checkpoint는 ImageNet-21k pre-training용 backbone이다.
# 즉, Oxford-IIIT Pet fine-tuning을 논문의 transfer setting에 더 가깝게 체험할 수 있다.
BASE_MODEL_ID = "google/vit-base-patch16-224-in21k"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "vit-base-oxford-pet"


class OxfordPetForViT(Dataset):
    """torchvision Oxford-IIIT Pet을 Hugging Face ViT 입력 tensor로 바꾼다."""

    def __init__(self, split, processor, download=True):
        self.dataset = OxfordIIITPet(
            ROOT / "data",
            split=split,
            target_types="category",
            download=download,
        )
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        # AutoImageProcessor가 resize / rescale / normalize를 수행한다.
        pixel_values = self.processor(
            images=image,
            return_tensors="pt",
        )["pixel_values"].squeeze(0)

        return pixel_values, label

    @property
    def classes(self):
        return self.dataset.classes


def limit_dataset(dataset, max_samples):
    """빠른 smoke test를 위해 앞쪽 일부 sample만 선택할 수 있게 한다."""
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, range(max_samples))


def evaluate(model, loader, device, loss_fn):
    """validation/test loss와 accuracy를 계산한다."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.inference_mode():
        for pixel_values, labels in loader:
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(pixel_values=pixel_values).logits
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_count += labels.size(0)

    return total_loss / total_count, total_correct / total_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="빠른 실행 확인용. 예: 128",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="빠른 실행 확인용. 예: 128",
    )
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Base pretrained model:", BASE_MODEL_ID)

    # 1) Pretrained ViT가 기대하는 입력 전처리를 불러온다.
    processor = AutoImageProcessor.from_pretrained(BASE_MODEL_ID)

    # 2) Oxford-IIIT Pet dataset을 준비한다.
    # trainval은 다시 train/validation으로 90:10 분할하고,
    # 공식 test split은 마지막 평가에서만 사용한다.
    full_trainval = OxfordPetForViT("trainval", processor)
    test_dataset = OxfordPetForViT("test", processor)

    classes = full_trainval.classes
    num_labels = len(classes)

    train_size = int(len(full_trainval) * 0.9)
    val_size = len(full_trainval) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_trainval,
        [train_size, val_size],
        generator=generator,
    )

    train_dataset = limit_dataset(train_dataset, args.max_train_samples)
    test_dataset = limit_dataset(test_dataset, args.max_test_samples)

    print(f"Oxford-IIIT Pet classes: {num_labels}")
    print(f"train: {len(train_dataset)}")
    print(f"validation: {len(val_dataset)}")
    print(f"test: {len(test_dataset)}")

    id2label = {index: name for index, name in enumerate(classes)}
    label2id = {name: index for index, name in enumerate(classes)}

    # 3) ImageNet-21k pretrained ViT backbone을 불러온다.
    # num_labels=37을 지정하면 Oxford-IIIT Pet용 classification head가 새로 붙는다.
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL_ID,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model = model.to(device)

    print()
    print("Fine-tuning tensor 흐름")
    print("Input  : [B, 3, 224, 224]")
    print("GT     : [B] (0~36 class index)")
    print("Output : [B, 37] logits")
    print("Loss   : CrossEntropyLoss")
    print("학습   : pretrained ViT 전체 + 새 classification head")
    print()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # 4) Fine-tuning용 Loss와 Optimizer.
    # 분류 문제이므로 GT class index와 37-class logits 사이에 Cross Entropy를 사용한다.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    # GPU에서는 mixed precision으로 메모리 사용량과 연산량을 줄인다.
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 5) Fine-tuning loop.
    for epoch in range(1, args.epochs + 1):
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_count = 0

        for step, (pixel_values, labels) in enumerate(train_loader, 1):
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(pixel_values=pixel_values).logits
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            running_correct += (logits.argmax(dim=-1) == labels).sum().item()
            running_count += labels.size(0)

            if step % 50 == 0 or step == len(train_loader):
                print(
                    f"[Epoch {epoch}/{args.epochs}] "
                    f"step {step}/{len(train_loader)} "
                    f"train_loss={running_loss / running_count:.4f} "
                    f"train_acc={running_correct / running_count:.4f}"
                )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            device,
            loss_fn,
        )
        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    # 6) 학습에 사용하지 않은 test split으로 마지막 성능을 확인한다.
    test_loss, test_acc = evaluate(
        model,
        test_loader,
        device,
        loss_fn,
    )

    print()
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_acc:.4f}")

    # 7) 직접 fine-tuning한 모델과 processor를 로컬 checkpoint로 저장한다.
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CHECKPOINT_DIR)
    processor.save_pretrained(CHECKPOINT_DIR)

    print("Saved checkpoint:", CHECKPOINT_DIR)
    print("다음: python study/11_compare_pretrained_finetuned.py")


if __name__ == "__main__":
    main()
