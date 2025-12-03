"""Video processing and visualization utilities for gaze estimation."""

import cv2
import numpy as np
from typing import Tuple, Dict, Any


def compute_roi_bounds(
    width: int,
    height: int,
    horizontal_margin: float = 0.1,
    top_margin: float = 0.35,
    footer_margin: float = 0.05,
) -> Tuple[int, int, int, int]:
    """Compute the ROI (Region of Interest) bounds for screen detection.

    Approximates the monitor area located just below a webcam.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        horizontal_margin: Fraction of width to leave on each side.
        top_margin: Fraction of height for top margin.
        footer_margin: Fraction of height for bottom margin.

    Returns:
        Tuple of (left, top, right, bottom) pixel coordinates.
    """
    left = int(width * horizontal_margin)
    right = width - left
    top = int(height * top_margin)
    bottom = min(height - int(height * footer_margin), height)
    return left, top, right, bottom


def is_point_in_roi(
    point: Tuple[int, int],
    roi_bounds: Tuple[int, int, int, int],
) -> bool:
    """Check if a point lies within the ROI bounds.

    Args:
        point: (x, y) coordinates to check.
        roi_bounds: (left, top, right, bottom) ROI boundaries.

    Returns:
        True if point is inside ROI, False otherwise.
    """
    x, y = point
    left, top, right, bottom = roi_bounds
    return left <= x <= right and top <= y <= bottom


def draw_roi(
    frame: np.ndarray,
    roi_bounds: Tuple[int, int, int, int],
    is_active: bool,
    config: Dict[str, Any] | None = None,
) -> None:
    """Draw ROI rectangle and status on frame.

    Args:
        frame: Image frame to draw on (modified in place).
        roi_bounds: (left, top, right, bottom) ROI boundaries.
        is_active: Whether someone is currently looking at ROI.
        config: Optional visualization config section.
    """
    # Default colors (BGR)
    if config:
        active_color = tuple(config.get("roi_active_color", [0, 200, 0]))
        inactive_color = tuple(config.get("roi_inactive_color", [0, 0, 255]))
        thickness = config.get("roi_thickness", 3)
        font_scale = config.get("font_scale_roi", 0.75)
        font_thickness = config.get("font_thickness", 2)
    else:
        active_color = (0, 200, 0)
        inactive_color = (0, 0, 255)
        thickness = 3
        font_scale = 0.75
        font_thickness = 2

    left, top, right, bottom = roi_bounds
    color = active_color if is_active else inactive_color
    label = "Screen ROI"
    status = "LOOKING" if is_active else "NOT LOOKING"

    cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
    text_y = max(top - 15, 30)
    cv2.putText(
        frame,
        f"{label}: {status}",
        (left, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        font_thickness,
        cv2.LINE_AA,
    )


def draw_gazepoint(
    frame: np.ndarray,
    gazepoint: Tuple[int, int],
    in_roi: bool,
    config: Dict[str, Any] | None = None,
) -> None:
    """Draw gazepoint indicator on frame.

    Args:
        frame: Image frame to draw on (modified in place).
        gazepoint: (x, y) coordinates of the gazepoint.
        in_roi: Whether gazepoint is inside ROI.
        config: Optional visualization config section.
    """
    if config:
        in_color = tuple(config.get("gazepoint_in_roi", [0, 220, 0]))
        out_color = tuple(config.get("gazepoint_out_roi", [0, 0, 255]))
        outline_color = tuple(config.get("gazepoint_outline", [0, 0, 0]))
        radius = config.get("gazepoint_radius", 10)
        outline_radius = config.get("gazepoint_outline_radius", 12)
    else:
        in_color = (0, 220, 0)
        out_color = (0, 0, 255)
        outline_color = (0, 0, 0)
        radius = 10
        outline_radius = 12

    x, y = gazepoint
    dot_color = in_color if in_roi else out_color
    cv2.circle(frame, (x, y), radius, dot_color, -1)
    cv2.circle(frame, (x, y), outline_radius, outline_color, 2)


def draw_face_status(
    frame: np.ndarray,
    bbox: np.ndarray,
    track_id: int | None,
    in_roi: bool,
    config: Dict[str, Any] | None = None,
) -> None:
    """Draw face bounding box status text.

    Args:
        frame: Image frame to draw on (modified in place).
        bbox: Bounding box as [x1, y1, x2, y2].
        track_id: Optional track ID for the face.
        in_roi: Whether the face's gaze is in ROI.
        config: Optional visualization config section.
    """
    if config:
        in_color = tuple(config.get("gazepoint_in_roi", [0, 220, 0]))
        out_color = tuple(config.get("gazepoint_out_roi", [0, 0, 255]))
        font_scale = config.get("font_scale", 0.6)
        font_thickness = config.get("font_thickness", 2)
        text_offset = config.get("text_offset_y", 10)
    else:
        in_color = (0, 220, 0)
        out_color = (0, 0, 255)
        font_scale = 0.6
        font_thickness = 2
        text_offset = 10

    dot_color = in_color if in_roi else out_color

    status_segments = []
    if track_id is not None:
        status_segments.append(f"ID {track_id}")
    status_segments.append("LOOKING AT ROI" if in_roi else "LOOKING AWAY")
    status_text = " - ".join(status_segments)

    text_origin = (int(bbox[0]), max(int(bbox[1]) - text_offset, 25))
    cv2.putText(
        frame,
        status_text,
        text_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        dot_color,
        font_thickness,
        cv2.LINE_AA,
    )


def crop_face(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> np.ndarray | None:
    """Safely crop a face region from frame.

    Args:
        frame: Source image frame.
        bbox: Bounding box as (x1, y1, x2, y2).

    Returns:
        Cropped face image, or None if invalid.
    """
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    h, w = frame.shape[:2]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]

