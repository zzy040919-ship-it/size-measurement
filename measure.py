"""OpenCV 工件尺寸测量。"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    box: np.ndarray
    width_px: float
    height_px: float
    area_px: float
    center: tuple[int, int]


def _contours(image: np.ndarray, min_area: float) -> Iterable[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return (c for c in contours if cv2.contourArea(c) >= min_area)


def detect(image: np.ndarray, min_area: float = 500.0) -> list[Detection]:
    """检测前景轮廓并返回按面积降序排列的旋转矩形。"""
    detections: list[Detection] = []
    for contour in _contours(image, min_area):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.int32)
        sides = np.linalg.norm(box - np.roll(box, -1, axis=0), axis=1)
        width_px, height_px = sorted((float(sides[0]), float(sides[1])), reverse=True)
        if cv2.contourArea(contour) / max(width_px * height_px, 1.0) < 0.35:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        detections.append(Detection(box, width_px, height_px, width_px * height_px, (x + w // 2, y + h // 2)))
    return sorted(detections, key=lambda item: item.area_px, reverse=True)


def pixels_per_mm(reference: Detection, reference_length_mm: float) -> float:
    if reference_length_mm <= 0:
        raise ValueError("reference_length_mm 必须大于 0")
    return reference.width_px / reference_length_mm


def annotate(image: np.ndarray, detections: list[Detection], scale: float) -> np.ndarray:
    output = image.copy()
    for index, item in enumerate(detections, 1):
        cv2.drawContours(output, [item.box], -1, (0, 0, 255), 2)
        width_mm = item.width_px / scale
        height_mm = item.height_px / scale
        area_mm2 = width_mm * height_mm
        x, y = item.center
        label = f"#{index} {width_mm:.1f} x {height_mm:.1f} mm | {area_mm2:.1f} mm2"
        cv2.putText(output, label, (max(8, x - 220), max(28, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 2, cv2.LINE_AA)
    return output


def _read(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return image


def measure_image(input_path: str | Path, output_path: str | Path, scale: float, min_area: float) -> list[Detection]:
    image = _read(input_path)
    detections = detect(image, min_area)
    if not detections:
        raise RuntimeError("没有检测到工件，请调整光照、背景或 --min-area")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotate(image, detections, scale)):
        raise OSError(f"无法写入结果: {output_path}")
    return detections


def run_camera(camera_id: int, scale: float, min_area: float) -> None:
    camera = cv2.VideoCapture(camera_id)
    if not camera.isOpened():
        raise RuntimeError(f"无法打开摄像头 {camera_id}")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            cv2.imshow("OpenCV Size Measurement | q: quit", annotate(frame, detect(frame, min_area), scale))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于参考物标定的 OpenCV 工件尺寸测量")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="待测图片")
    source.add_argument("--camera", type=int, help="摄像头编号，例如 0")
    parser.add_argument("--reference-image", help="参考物图片；取面积最大的轮廓完成标定")
    parser.add_argument("--reference-mm", type=float, help="参考物实际长度（mm）")
    parser.add_argument("--pixel-per-mm", type=float, help="已知标定结果，跳过参考图")
    parser.add_argument("--output", default="outputs/result.jpg", help="结果图路径")
    parser.add_argument("--min-area", type=float, default=500.0, help="最小轮廓面积（像素）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scale = args.pixel_per_mm
    if scale is None:
        if not args.reference_image or args.reference_mm is None:
            raise SystemExit("请提供 --pixel-per-mm，或同时提供 --reference-image 和 --reference-mm")
        references = detect(_read(args.reference_image), args.min_area)
        if not references:
            raise SystemExit("参考图中没有检测到参考物")
        scale = pixels_per_mm(references[0], args.reference_mm)
        print(f"calibration: {scale:.4f} px/mm")
    if scale <= 0:
        raise SystemExit("--pixel-per-mm 必须大于 0")
    if args.camera is not None:
        run_camera(args.camera, scale, args.min_area)
        return
    detections = measure_image(args.image, args.output, scale, args.min_area)
    print(f"detections: {len(detections)}")
    for index, item in enumerate(detections, 1):
        print(f"#{index}: {item.width_px / scale:.2f} x {item.height_px / scale:.2f} mm")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
