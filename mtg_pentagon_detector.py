from __future__ import annotations

import argparse
import itertools
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]
DetectionResult = Optional[Dict[str, Any]]


def contour_centers(gray: np.ndarray) -> Tuple[List[Point], np.ndarray]:
    """Return contour centroids from a thresholded grayscale image."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    centers: List[Point] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 10:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        centers.append((cx, cy))

    deduped = sorted(set(centers))
    return deduped, mask


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _order_vertices(vertices: Sequence[Point]) -> List[Point]:
    cx = sum(p[0] for p in vertices) / len(vertices)
    cy = sum(p[1] for p in vertices) / len(vertices)
    return sorted(vertices, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def _normalize_camera_frame(frame: np.ndarray) -> np.ndarray:
    """Convert common Pi camera frame layouts into a BGR image for OpenCV."""
    if not isinstance(frame, np.ndarray):
        raise TypeError("Expected a NumPy array from the Pi camera")

    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if frame.ndim != 3:
        raise ValueError("Expected a 2D grayscale or 3D color frame")

    channels = frame.shape[2]
    if channels == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if channels == 4:
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    raise ValueError(f"Unsupported channel count: {channels}")


def pentagon_score(vertices: Sequence[Point]) -> Tuple[float, Point, List[Point]]:
    """Score how close five points are to a regular pentagon.

    Lower scores are better.
    """
    if len(vertices) != 5:
        raise ValueError("Exactly five vertices are required")

    ordered = _order_vertices(vertices)
    cx = int(round(sum(p[0] for p in ordered) / 5))
    cy = int(round(sum(p[1] for p in ordered) / 5))
    center = (cx, cy)

    side_lengths = [_dist(ordered[i], ordered[(i + 1) % 5]) for i in range(5)]
    radii = [_dist(p, center) for p in ordered]
    angles = []
    for i in range(5):
        p1 = ordered[i]
        p2 = ordered[(i + 1) % 5]
        angle = math.atan2(p2[1] - cy, p2[0] - cx) - math.atan2(
            p1[1] - cy, p1[0] - cx
        )
        angles.append(angle)

    angles = [((angle + math.pi * 3) % (math.pi * 2)) - math.pi for angle in angles]
    expected_angle = (2 * math.pi) / 5
    angle_error = sum(abs(abs(a) - expected_angle) for a in angles) / 5

    mean_side = sum(side_lengths) / 5
    mean_radius = sum(radii) / 5
    side_error = (
        0.0
        if mean_side == 0
        else sum(abs(s - mean_side) for s in side_lengths) / (5 * mean_side)
    )
    radius_error = (
        0.0
        if mean_radius == 0
        else sum(abs(r - mean_radius) for r in radii) / (5 * mean_radius)
    )

    score = side_error + radius_error + (angle_error / expected_angle)
    return score, center, ordered


def find_best_pentagon(
    candidates: Sequence[Point], max_score: float = 0.25
) -> DetectionResult:
    """Find the best pentagon candidate from a set of points."""
    if len(candidates) < 5:
        return None

    best: DetectionResult = None
    best_score = float("inf")

    for combo in itertools.combinations(candidates, 5):
        score, center, ordered = pentagon_score(combo)
        if score < best_score:
            best_score = score
            best = {
                "score": score,
                "center": center,
                "vertices": ordered,
            }

    if best is None or best_score > max_score:
        return None

    return best


def annotate_image(
    image: np.ndarray,
    result: DetectionResult,
    candidates: Sequence[Point],
) -> np.ndarray:
    """Draw candidates and detected pentagon on a copy of the image."""
    out = image.copy()

    for p in candidates:
        cv2.circle(out, (int(p[0]), int(p[1])), 3, (0, 255, 255), -1)

    if result is None:
        return out

    center = result["center"]
    vertices = result["vertices"]

    for p in vertices:
        cv2.circle(out, p, 6, (0, 255, 0), -1)

    for i in range(5):
        cv2.line(out, vertices[i], vertices[(i + 1) % 5], (255, 0, 0), 2)

    cv2.circle(out, center, 8, (0, 0, 255), -1)
    return out


def detect_mtg_back_center(
    frame: np.ndarray, max_score: float = 0.25
) -> Dict[str, Any]:
    """Detect a pentagon from a Pi camera frame such as Picamera2.capture_array()."""
    image = _normalize_camera_frame(frame)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    candidates, mask = contour_centers(gray)
    result = find_best_pentagon(candidates, max_score=max_score)
    annotated = annotate_image(image, result, candidates)

    return {
        "result": result,
        "candidates": candidates,
        "mask": mask,
        "annotated": annotated,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the detector against an image file as a stand-in for a Pi camera frame."
    )
    parser.add_argument(
        "image_path",
        help="Path to an image file to load into a NumPy array before detection.",
    )
    parser.add_argument(
        "--max-score",
        type=float,
        default=0.25,
        help="Maximum acceptable pentagon score.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    frame = cv2.imread(args.image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {args.image_path}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    data = detect_mtg_back_center(frame, max_score=args.max_score)

    if data["result"] is None:
        print("No good pentagon found")
    else:
        print("Pentagon center:", data["result"]["center"])
        print("Score:", data["result"]["score"])
        print("Vertices:", data["result"]["vertices"])

    cv2.imwrite("mask.jpg", data["mask"])
    cv2.imwrite("pentagon_result.jpg", data["annotated"])


if __name__ == "__main__":
    main()
