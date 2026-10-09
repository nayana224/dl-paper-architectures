"""SAM: 한 이미지에서 Point / Box / 조합 / Dense Mask refinement 비교.

공식 Pretrained SAM checkpoint 사용; 학습(Weight Update)은 하지 않는다.
Point/Box는 Sparse Prompt, 이전 Mask logits는 Dense Prompt로 재사용한다.
"""
import argparse
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "assets" / "sam_dog.jpg"
SAMPLE_URL = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
OUTPUT = ROOT / "outputs" / "07_sam_prompts.png"
CORRECTED_OUTPUT = ROOT / "outputs" / "07_sam_corrected_box.png"
CHECKPOINT = "facebook/sam-vit-base"


def get_image(path_arg):
    path = Path(path_arg).expanduser() if path_arg else SAMPLE
    if not path.exists():
        if path_arg:
            raise FileNotFoundError(f"이미지 파일이 없습니다: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        print("실제 사진 다운로드:", SAMPLE_URL)
        urlretrieve(SAMPLE_URL, path)
    return Image.open(path).convert("RGB"), path


def draw_overlay(ax, image, mask, point=None, box=None):
    ax.imshow(image)
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[mask, :] = [0.12, 0.70, 0.95, 0.48]
    ax.imshow(rgba)
    if box is not None:
        x0, y0, x1, y1 = box
        ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0,
                                   fill=False, color="lime", linewidth=2))
    if point is not None:
        ax.scatter([point[0]], [point[1]], color="yellow", marker="*", s=145,
                   edgecolor="black", linewidth=0.7)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description="SAM Sparse/Dense prompt inference")
    parser.add_argument("--image", type=str, help="사진 경로(기본: 샘플 강아지 사진)")
    parser.add_argument("--point", type=float, nargs=2, metavar=("X", "Y"),
                        help="Positive point 픽셀 좌표")
    parser.add_argument("--box", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                        help="Bounding box 픽셀 좌표")
    parser.add_argument("--corrected-box", type=float, nargs=4,
                        metavar=("X0", "Y0", "X1", "Y1"),
                        help="비교할 넓은 Box 좌표 (기본: 샘플 강아지 전체를 덮는 비율)")
    args = parser.parse_args()

    image, path = get_image(args.image)
    w, h = image.size
    # 기본 좌표는 샘플 강아지 사진의 대략적인 중앙 객체에 맞춘 비율이다.
    # 다른 사진은 --point / --box로 직접 지정하는 것을 권장한다.
    point = args.point or (0.52*w, 0.52*h)
    box = args.box or (0.25*w, 0.13*h, 0.83*w, 0.94*h)
    px, py = point
    x0, y0, x1, y1 = box
    if not (0 <= px < w and 0 <= py < h and 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError(f"Point/Box는 사진 크기 {w}x{h} 안에 있어야 합니다.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| Image:", path, "| Size:", (w, h))
    print("Point:", tuple(round(x,1) for x in point), "| Box:", tuple(round(x,1) for x in box))

    processor = SamProcessor.from_pretrained(CHECKPOINT)
    model = SamModel.from_pretrained(CHECKPOINT).to(device).eval()

    def run(label, points=None, boxes=None, previous_logits=None, multimask=True):
        kwargs = {"images": image, "return_tensors": "pt"}
        if points is not None:
            kwargs["input_points"] = [[list(points)]]  # [batch, point_batch, num_points, xy]
            kwargs["input_labels"] = [[[1]]]           # Positive point
        if boxes is not None:
            kwargs["input_boxes"] = [[list(boxes)]]    # [batch, point_batch, xyxy]
        inputs = processor(**kwargs).to(device)
        # Dense Prompt: Conv2d는 [B, 1, 256, 256] 형태의 Mask logits을 받는다.
        if previous_logits is not None:
            if previous_logits.ndim != 4 or previous_logits.shape[1] != 1:
                raise ValueError(f"input_masks must be [B,1,H,W], got {tuple(previous_logits.shape)}")
            inputs["input_masks"] = previous_logits

        with torch.inference_mode():
            output = model(**inputs, multimask_output=multimask)

        scores = output.iou_scores[0, 0]
        index = int(scores.argmax().item())
        # processor는 입력 원본 크기에 맞춰 Mask logits을 복원한다.
        resized = processor.image_processor.post_process_masks(
            output.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
            binarize=False,
        )[0]  # [point_batch, mask_candidates, original_height, original_width]
        full_mask = resized[0, index].numpy() > 0
        low_res = output.pred_masks[:, 0, index].unsqueeze(1).detach()  # [B,1,256,256]
        print(f"{label}: low-res {list(output.pred_masks.shape)}, "
              f"predicted IoU {scores.detach().cpu().tolist()}, selected {index}, "
              f"foreground pixels {int(full_mask.sum())}")
        return full_mask, low_res

    point_mask, _ = run("Point", points=point)
    box_mask, _ = run("Box", boxes=box)
    both_mask, both_logits = run("Point + Box", points=point, boxes=box)
    # 동일 Image Embedding을 모델 내부에서 재계산한다. 아래는 Mask logits 재사용을
    # 보여주는 실습이며, 성능 최적화/이미지 캐싱 자체를 시연하는 코드는 아니다.
    refined_mask, _ = run(
        "Point + Box + previous Mask", points=point, boxes=box,
        previous_logits=both_logits, multimask=False,
    )

    original_cases = [
        ("Positive Point", point_mask, point, None),
        ("Bounding Box", box_mask, None, box),
        ("Point + Box", both_mask, point, box),
        ("Point + Box + Dense Mask", refined_mask, point, box),
    ]
    # 이미지 전체의 강아지를 감싸도록 기존 Box보다 넓힌 두 번째 조건.
    # 다른 사진에서 기본값이 적합하지 않으면 --corrected-box를 직접 지정한다.
    corrected_box = args.corrected_box or (0.08*w, 0.02*h, 0.97*w, 0.96*h)
    cx0, cy0, cx1, cy1 = corrected_box
    if not (0 <= cx0 < cx1 <= w and 0 <= cy0 < cy1 <= h):
        raise ValueError(f"Corrected Box는 사진 크기 {w}x{h} 안에 있어야 합니다.")
    print("Corrected Box:", tuple(round(x, 1) for x in corrected_box))
    corrected_box_mask, _ = run("Corrected Box", boxes=corrected_box)
    corrected_both_mask, corrected_logits = run(
        "Point + Corrected Box", points=point, boxes=corrected_box,
    )
    corrected_refined_mask, _ = run(
        "Point + Corrected Box + previous Mask",
        points=point, boxes=corrected_box,
        previous_logits=corrected_logits, multimask=False,
    )
    corrected_cases = [
        ("Positive Point (reference)", point_mask, point, None),
        ("Corrected Bounding Box", corrected_box_mask, None, corrected_box),
        ("Point + Corrected Box", corrected_both_mask, point, corrected_box),
        ("Point + Corrected Box + Dense Mask",
         corrected_refined_mask, point, corrected_box),
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for output_path, cases in (
        (OUTPUT, original_cases),
        (CORRECTED_OUTPUT, corrected_cases),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        for ax, (name, mask, marker, bounds) in zip(axes.flat, cases):
            draw_overlay(ax, image, mask, marker, bounds)
            ax.set_title(name)
        fig.savefig(output_path, dpi=140)
        plt.close(fig)
        print("Saved:", output_path)
    print("Predicted IoU는 GT와 계산한 실제 IoU가 아닌 모델의 Mask 품질 예측입니다.")
    print("Refinement 결과가 반드시 개선되는 것은 아닙니다.")


if __name__ == "__main__":
    main()
