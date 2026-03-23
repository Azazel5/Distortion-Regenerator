from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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
    morph_iterations: int = 2
    canny_low: int = 40
    canny_high: int = 130
    min_contour_area_ratio: float = 0.08
    polygon_epsilon_ratio: float = 0.02
    polygon_epsilon_candidates: tuple[float, ...] = (0.012, 0.02, 0.03, 0.045)
    min_quad_area_ratio: float = 0.06
    subpix_window: int = 7
    subpix_iterations: int = 40
    subpix_eps: float = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatic document rectification pipeline."
    )
    parser.add_argument(
        "input_dir_positional",
        nargs="?",
        type=Path,
        default=None,
        help="Optional positional input directory (for assignment-style CLI).",
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
    candidates = (
        ("otsu", otsu_mask),
        ("adaptive", adaptive_mask),
        ("otsu-inv", cv2.bitwise_not(otsu_mask)),
        ("adaptive-inv", cv2.bitwise_not(adaptive_mask)),
    )

    def plausibility_score(mask: np.ndarray) -> float:
        ratio = float(np.count_nonzero(mask)) / float(mask.size)
        target = 0.30
        return 1.0 - min(1.0, abs(ratio - target) / target)

    method, binary = max(candidates, key=lambda item: plausibility_score(item[1]))

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


def _is_valid_quad(
    quad: np.ndarray, image_shape: tuple[int, int, int], min_quad_area_ratio: float
) -> bool:
    if quad.shape != (4, 2):
        return False
    h, w = image_shape[:2]
    area = cv2.contourArea(quad.reshape(-1, 1, 2))
    if area < min_quad_area_ratio * float(h * w):
        return False
    return cv2.isContourConvex(quad.reshape(-1, 1, 2).astype(np.int32))


def _largest_component_box(
    mask: np.ndarray, min_area_px: float, config: PipelineConfig
) -> np.ndarray | None:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return None

    best_label = -1
    best_area = -1
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        if area > best_area:
            best_area = area
            best_label = label

    if best_label < 0:
        return None

    comp_mask = np.zeros_like(mask, dtype=np.uint8)
    comp_mask[labels == best_label] = 255
    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    box = order_corners_clockwise(box)
    if _is_valid_quad(box, (*mask.shape, 1), config.min_quad_area_ratio):
        return box
    return None


def detect_document_corners(
    gray: np.ndarray, binary_mask: np.ndarray, config: PipelineConfig
) -> tuple[np.ndarray | None, str]:
    edges_binary = cv2.Canny(binary_mask, config.canny_low, config.canny_high)
    edges_gray = cv2.Canny(gray, config.canny_low, config.canny_high)
    edges = cv2.bitwise_or(edges_binary, edges_gray)
    edge_kernel = np.ones((3, 3), dtype=np.uint8)
    edges = cv2.dilate(edges, edge_kernel, iterations=1)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, "no-contours"

    image_area = float(gray.shape[0] * gray.shape[1])
    min_area = config.min_contour_area_ratio * image_area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    best_quad: np.ndarray | None = None
    best_method = "none"
    best_score = -1.0

    def candidate_score(quad: np.ndarray) -> float:
        area = cv2.contourArea(quad.reshape(-1, 1, 2))
        normalized_area = area / image_area
        edge_lengths = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
        min_len = max(1e-6, float(np.min(edge_lengths)))
        max_len = float(np.max(edge_lengths))
        edge_balance = min_len / max_len
        return normalized_area + 0.25 * edge_balance
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        peri = cv2.arcLength(contour, True)
        for eps_ratio in config.polygon_epsilon_candidates:
            epsilon = eps_ratio * peri
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                quad = approx.reshape(4, 2).astype(np.float32)
                quad = order_corners_clockwise(quad)
                if _is_valid_quad(quad, (*gray.shape, 1), config.min_quad_area_ratio):
                    score = candidate_score(quad)
                    if score > best_score:
                        best_quad = quad
                        best_method = f"contour-quad-eps-{eps_ratio:.3f}"
                        best_score = score

    # Fallback: use minAreaRect from largest valid-area contour.
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.float32)
        box = order_corners_clockwise(box)
        if _is_valid_quad(box, (*gray.shape, 1), config.min_quad_area_ratio):
            score = candidate_score(box)
            if score > best_score:
                best_quad = box
                best_method = "min-area-rect"
                best_score = score

    # Secondary fallback: largest connected component on binary masks.
    comp_quad = _largest_component_box(binary_mask, min_area, config)
    if comp_quad is not None:
        score = candidate_score(comp_quad)
        if score > best_score:
            best_quad = comp_quad
            best_method = "connected-component"
            best_score = score

    inv_binary = cv2.bitwise_not(binary_mask)
    comp_inv_quad = _largest_component_box(inv_binary, min_area, config)
    if comp_inv_quad is not None:
        score = candidate_score(comp_inv_quad)
        if score > best_score:
            best_quad = comp_inv_quad
            best_method = "connected-component-inv"
            best_score = score

    if best_quad is not None:
        return best_quad, best_method
    return None, "no-valid-quad"


def refine_corners_subpixel(
    gray: np.ndarray, corners: np.ndarray, config: PipelineConfig
) -> np.ndarray:
    h, w = gray.shape[:2]
    pts = corners.astype(np.float32).copy()
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    win = max(2, config.subpix_window)
    margin = win + 1
    pts[:, 0] = np.clip(pts[:, 0], margin, w - 1 - margin)
    pts[:, 1] = np.clip(pts[:, 1], margin, h - 1 - margin)
    pts = pts.reshape(-1, 1, 2)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        config.subpix_iterations,
        config.subpix_eps,
    )
    try:
        refined = cv2.cornerSubPix(gray, pts, (win, win), (-1, -1), criteria)
    except cv2.error:
        return corners
    refined = refined.reshape(-1, 2)
    refined[:, 0] = np.clip(refined[:, 0], 0, w - 1)
    refined[:, 1] = np.clip(refined[:, 1], 0, h - 1)
    refined = order_corners_clockwise(refined)
    if _is_valid_quad(refined, (*gray.shape, 1), config.min_quad_area_ratio):
        return refined
    return corners


def warp_with_homography(
    image_bgr: np.ndarray, corners: np.ndarray, config: PipelineConfig
) -> np.ndarray:
    destination = np.array(
        [
            [0, 0],
            [config.output_width - 1, 0],
            [config.output_width - 1, config.output_height - 1],
            [0, config.output_height - 1],
        ],
        dtype=np.float32,
    )
    h_matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)
    warped = cv2.warpPerspective(
        image_bgr,
        h_matrix,
        (config.output_width, config.output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


def _extract_hf_image(sample: dict[str, Any], idx: int) -> Image.Image | None:
    raw = sample.get("image")
    if raw is None:
        print(f"[hf] skipped index {idx}: missing 'image' field")
        return None
    if isinstance(raw, Image.Image):
        return raw
    if isinstance(raw, np.ndarray):
        return Image.fromarray(raw)
    if isinstance(raw, dict) and "path" in raw:
        return Image.open(raw["path"]).convert("RGB")
    try:
        return Image.fromarray(np.array(raw))
    except Exception:
        print(f"[hf] skipped index {idx}: unsupported image format")
        return None


def rectify_document(image_bgr: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """
    Placeholder for the full CV pipeline.
    Chunk 2+ will add preprocessing, contour extraction, corner ordering,
    and perspective warp.
    """
    gray, binary_mask, threshold_method = preprocess_and_binarize(image_bgr, config)
    foreground_ratio = float(np.count_nonzero(binary_mask)) / float(binary_mask.size)
    corners, corner_method = detect_document_corners(gray, binary_mask, config)
    if corners is not None:
        corners = refine_corners_subpixel(gray, corners, config)
    status = "ok" if corners is not None else "fallback-no-corners"
    output_mode = "homography-warp" if corners is not None else "resize-fallback"
    print(
        "[pipeline] preprocessing+corners: "
        f"threshold={threshold_method}, foreground_ratio={foreground_ratio:.3f}, "
        f"corner_method={corner_method}, status={status}, output_mode={output_mode}"
    )

    if corners is not None:
        try:
            return warp_with_homography(image_bgr, corners, config)
        except cv2.error:
            print("[pipeline] homography failed, using resize fallback")

    return cv2.resize(
        image_bgr,
        (config.output_width, config.output_height),
        interpolation=cv2.INTER_CUBIC,
    )


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
    method_counts: dict[str, int] = {}
    count = 0
    for idx, sample in enumerate(dataset):
        if limit is not None and count >= limit:
            break
        raw_image = _extract_hf_image(sample, idx)
        if raw_image is None:
            continue
        image = _to_bgr(raw_image)
        rectified = rectify_document(image, config)
        output_name = f"output_{idx:05d}.jpg"
        write_output(rectified, output_dir / output_name)
        count += 1
        # Keep a lightweight runtime profile of which strategies are used.
        # The latest pipeline log line includes corner method and status.
        print(f"[hf] processed index {idx} -> {output_name}")
    _ = method_counts
    return count


def main() -> None:
    args = parse_args()
    config = PipelineConfig()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    effective_input_dir = args.input_dir or args.input_dir_positional

    if effective_input_dir is not None:
        if not effective_input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {effective_input_dir}")
        if not effective_input_dir.is_dir():
            raise NotADirectoryError(
                f"Input path is not a directory: {effective_input_dir}"
            )
        total = process_local_folder(
            effective_input_dir, args.output_dir, config, args.limit
        )
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
