import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Callable, Optional

def select_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def crop_face(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray | None:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    h, w = img.shape[:2]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def score_color(score: float) -> Tuple[int, int, int]:
    r = int(255 * (1 - score))
    g = int(255 * score)
    return (0, g, r)

def draw_box(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    score: float,
    conf: float,
):
    x1, y1, x2, y2 = bbox
    color = score_color(score)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text_ec = f"EC:{score:.2f}"
    text_conf = f"{conf:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text_conf, (x1, y1 - 6), font, 0.5, color, 1)
    cv2.putText(frame, text_ec, (x1, y2 + 15), font, 0.5, (255, 255, 255), 1)

def process_video(
    src: str | Path,
    fn: Callable[[np.ndarray], np.ndarray],
    dst: str | Path | None = None,
    display: bool = False,
    skip: int = 0,
):
    p = Path(src)
    if not p.exists():
        raise FileNotFoundError(p)
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise IOError(p)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = None
    if dst:
        d = Path(dst)
        d.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(d), fourcc, fps, (w, h))
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if skip and idx % (skip + 1) != 0:
                idx += 1
                continue
            out = fn(frame)
            if writer:
                writer.write(out)
            if display:
                cv2.imshow("proc", out)
                if cv2.waitKey(1) == ord("q"):
                    break
            idx += 1
    finally:
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

def process_image(
    src: str | Path,
    fn: Callable[[np.ndarray], np.ndarray],
    dst: str | Path | None = None,
):
    p = Path(src)
    if not p.exists():
        raise FileNotFoundError(p)
    img = cv2.imread(str(p))
    if img is None:
        raise ValueError(p)
    out = fn(img)
    if dst:
        d = Path(dst)
        d.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d), out)
    return out
