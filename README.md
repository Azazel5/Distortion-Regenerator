# Automated Document Aligner (Midterm Project)

This project builds an automatic computer vision pipeline that takes distorted photos of a document and outputs rectified, frontal views.

The final deliverable is a single script, `vision.py`, that processes all images in an input folder and writes rectified outputs to a new output folder.

## Objective

- Detect the page boundary from each input image.
- Localize the four document corners in a consistent order.
- Estimate a homography from detected corners to a canonical page.
- Warp the original image to produce a clean frontal document view.

## Dataset

- Synthetic dataset (provided by course staff):  
  [Google Drive Link](https://drive.google.com/file/d/1pAuTSutQdl25-Ifzs5Zw5Q-MCvcPGTGu/view?usp=sharing)

  This dataset has been uploaded to HuggingFace for easy inference during training
  [HuggingFace](https://huggingface.co/datasets/ShubUpad/CS-132-Computer-Vision-Midterm/tree/main)

- Expected inputs are `.jpg` / `.png` images containing one letter-sized page (`11" x 8.5"`) under perspective distortion.

## Output Requirements

- Script must run from command line:
  - `python3 vision.py`
- For each input image, generate one output image in a new output folder.
- Final output resolution must be exactly `550 x 425` pixels (50 DPI for `11" x 8.5"`).
- Use one fixed parameter set for all images (fully automatic, no per-image manual tuning).

## Planned Pipeline

### 1) Preprocessing and Binarization

- Convert input image to grayscale.
- Apply edge-preserving denoising (Gaussian blur first; bilateral as fallback if needed).
- Segment document from background using thresholding:
  - Start with Otsu threshold.
  - Fall back to adaptive threshold when global thresholding fails.
- Apply morphology (close/open) to strengthen page region and remove small artifacts.

### 2) Edge and Contour Extraction

- Run Canny edge detection on the cleaned binary/grayscale image.
- Find contours and rank candidates by area and polygon quality.
- Prefer the largest plausible quadrilateral contour as the document boundary.
- Fallback path (if contour approach fails): detect boundary lines with Hough and intersect lines.

### 3) Corner Localization and Ordering

- Approximate contour with `cv2.approxPolyDP` to obtain four vertices.
- Refine corners (if necessary) using sub-pixel corner refinement (`cv2.cornerSubPix`) on grayscale data.
- Enforce consistent order: `[top-left, top-right, bottom-right, bottom-left]`.
- Validate geometry (convexity, minimum area, reasonable edge lengths) before rectification.

### 4) Geometric Rectification

- Use detected corners as source points.
- Use canonical page corners for destination points:
  - `[(0,0), (424,0), (424,549), (0,549)]` for width `425`, height `550`.
- Compute homography with OpenCV (`cv2.getPerspectiveTransform` or robust alternative).
- Warp with `cv2.warpPerspective` to output a color rectified image of shape `550x425`.

## Initial Parameter Plan

These are starting values that will be tuned once we run full-batch tests:

- Gaussian blur kernel: `5x5`
- Canny thresholds: `50, 150`
- Contour area threshold: at least `20%` of image area
- Polygon epsilon for approximation: `1.5% - 3.0%` of contour perimeter
- Morphology kernel: `3x3` to `5x5`, 1-2 iterations

Final chosen values and rationale will be updated here after evaluation on representative samples.

## Repository Structure (Target)

- `vision.py` - main CLI pipeline script
- `output_samples/` - representative rectified outputs (5-10 examples)
- `README.md` - method, parameters, and usage notes
- `requirements.txt` - Python dependencies

## Setup

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Minimum required libraries:

- `opencv-python`
- `numpy`
- `pillow`

## Run

```bash
python3 vision.py
```

The script will create an output directory in the current working directory and save one rectified file per input image.

## Evaluation Plan

- Run on a representative split of easy, medium, and hard distortions.
- Visually inspect:
  - page straightness,
  - corner alignment,
  - text readability near boundaries,
  - failure modes (partial page, heavy shadows, extreme perspective).
- Track which fallback branch was used for each image to debug robustness.

## Notes

- This is an individual assignment.
- AI tools can support implementation/debugging/documentation, but all submitted work must be understood and validated by the student.
