from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from datasets import load_dataset
from PIL import Image


OUTPUT_WIDTH = 425
OUTPUT_HEIGHT = 550
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class PipelineConfig:
    output_width: int = OUTPUT_WIDTH
    output_height: int = OUTPUT_HEIGHT
    gaussian_kernel: int = 5
    canny_low: int = 50
    canny_high: int = 150
    min_contour_area_ratio: float = 0.20
    polygon_epsilon_ratio: float = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatic document rectification pipeline."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Optional local input directory with images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where rectified images are written.",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="ShubUpad/CS-132-Computer-Vision-Midterm",
        help="HuggingFace dataset ID to use when --input-dir is omitted.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to read from HuggingFace.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of images to process.",
    )
    return parser.parse_args()


def iter_local_images(input_dir: Path) -> Iterable[Path]:
    for item in sorted(input_dir.iterdir()):
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            yield item


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def rectify_document(image_bgr: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """
    Placeholder for the full CV pipeline.
    Chunk 2+ will add preprocessing, contour extraction, corner ordering,
    and perspective warp.
    """
    _ = config
    return cv2.resize(image_bgr, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_CUBIC)


def write_output(image_bgr: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), image_bgr)
    if not success:
        raise ValueError(f"Failed to write output image: {output_path}")


def process_local_folder(
    input_dir: Path, output_dir: Path, config: PipelineConfig, limit: int | None = None
) -> int:
    count = 0
    for image_path in iter_local_images(input_dir):
        if limit is not None and count >= limit:
            break
        image = read_image(image_path)
        rectified = rectify_document(image, config)
        output_name = f"output_{image_path.stem}.jpg"
        write_output(rectified, output_dir / output_name)
        count += 1
        print(f"[local] processed: {image_path.name} -> {output_name}")
    return count


def _to_bgr(pil_image: Image.Image) -> np.ndarray:
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def process_hf_dataset(
    dataset_id: str,
    split: str,
    output_dir: Path,
    config: PipelineConfig,
    limit: int | None = None,
) -> int:
    dataset = load_dataset(dataset_id, split=split)
    count = 0
    for idx, sample in enumerate(dataset):
        if limit is not None and count >= limit:
            break
        # We will tighten schema handling in the next chunk.
        raw = sample.get("image")
        if raw is None:
            print(f"[hf] skipped index {idx}: missing 'image' field")
            continue
        if not isinstance(raw, Image.Image):
            raw = Image.fromarray(np.array(raw))
        image = _to_bgr(raw)
        rectified = rectify_document(image, config)
        output_name = f"output_{idx:05d}.jpg"
        write_output(rectified, output_dir / output_name)
        count += 1
        print(f"[hf] processed index {idx} -> {output_name}")
    return count


def main() -> None:
    args = parse_args()
    config = PipelineConfig()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_dir is not None:
        if not args.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
        if not args.input_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {args.input_dir}")
        total = process_local_folder(args.input_dir, args.output_dir, config, args.limit)
    else:
        total = process_hf_dataset(
            dataset_id=args.hf_dataset,
            split=args.split,
            output_dir=args.output_dir,
            config=config,
            limit=args.limit,
        )

    print(f"Done. Wrote {total} rectified images to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
