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
    bilateral_d: int = 7
    bilateral_sigma_color: int = 50
    bilateral_sigma_space: int = 50
    adaptive_block_size: int = 35
    adaptive_c: int = 10
    morph_kernel: int = 5
    morph_iterations: int = 1
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


def _ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def preprocess_and_binarize(
    image_bgr: np.ndarray, config: PipelineConfig
) -> tuple[np.ndarray, np.ndarray, str]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    k = _ensure_odd(max(3, config.gaussian_kernel))
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    smooth = cv2.bilateralFilter(
        blurred,
        d=config.bilateral_d,
        sigmaColor=config.bilateral_sigma_color,
        sigmaSpace=config.bilateral_sigma_space,
    )

    _, otsu_mask = cv2.threshold(
        smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    adaptive_block = _ensure_odd(max(3, config.adaptive_block_size))
    adaptive_mask = cv2.adaptiveThreshold(
        smooth,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        adaptive_block,
        config.adaptive_c,
    )

    # In this dataset the document is brighter than background.
    # Pick the threshold branch that yields a plausible foreground ratio.
    otsu_ratio = float(np.count_nonzero(otsu_mask)) / float(otsu_mask.size)
    method = "otsu"
    binary = otsu_mask
    if otsu_ratio < 0.05 or otsu_ratio > 0.80:
        method = "adaptive"
        binary = adaptive_mask

    kernel_size = max(3, config.morph_kernel)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    cleaned = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=config.morph_iterations,
    )
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        kernel,
        iterations=config.morph_iterations,
    )

    return gray, cleaned, method


def order_corners_clockwise(points: np.ndarray) -> np.ndarray:
    pts = points.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(d)]
    bottom_left = pts[np.argmax(d)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def _is_valid_quad(quad: np.ndarray, image_shape: tuple[int, int, int]) -> bool:
    if quad.shape != (4, 2):
        return False
    h, w = image_shape[:2]
    area = cv2.contourArea(quad.reshape(-1, 1, 2))
    if area < 0.10 * float(h * w):
        return False
    return cv2.isContourConvex(quad.reshape(-1, 1, 2).astype(np.int32))


def detect_document_corners(
    gray: np.ndarray, binary_mask: np.ndarray, config: PipelineConfig
) -> tuple[np.ndarray | None, str]:
    edges = cv2.Canny(binary_mask, config.canny_low, config.canny_high)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, "no-contours"

    min_area = config.min_contour_area_ratio * float(gray.shape[0] * gray.shape[1])
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        peri = cv2.arcLength(contour, True)
        epsilon = config.polygon_epsilon_ratio * peri
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            quad = order_corners_clockwise(quad)
            if _is_valid_quad(quad, (*gray.shape, 1)):
                return quad, "contour-quad"

    # Fallback: use minAreaRect from largest valid-area contour.
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.float32)
        box = order_corners_clockwise(box)
        if _is_valid_quad(box, (*gray.shape, 1)):
            return box, "min-area-rect"

    return None, "no-valid-quad"


def rectify_document(image_bgr: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """
    Placeholder for the full CV pipeline.
    Chunk 2+ will add preprocessing, contour extraction, corner ordering,
    and perspective warp.
    """
    gray, binary_mask, threshold_method = preprocess_and_binarize(image_bgr, config)
    foreground_ratio = float(np.count_nonzero(binary_mask)) / float(binary_mask.size)
    corners, corner_method = detect_document_corners(gray, binary_mask, config)
    status = "ok" if corners is not None else "fallback-no-corners"
    print(
        "[pipeline] preprocessing+corners: "
        f"threshold={threshold_method}, foreground_ratio={foreground_ratio:.3f}, "
        f"corner_method={corner_method}, status={status}"
    )

    # Chunk 3 finds/validates document corners.
    # Chunk 4 will use these corners to compute perspective homography.
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
