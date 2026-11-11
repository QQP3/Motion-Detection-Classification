
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class MotionRegion:
    x: int
    y: int
    width: int
    height: int
    area: float

    @property
    def center(self) -> Tuple[int, int]:
        return (
            self.x + self.width // 2,
            self.y + self.height // 2,
        )

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


class MotionDetector:
    def __init__(
        self,
        min_area: float = 500.0,
        max_area: Optional[float] = None,
    ):
        if min_area < 0:
            raise ValueError("min_area cannot be negative")

        if max_area is not None and max_area < min_area:
            raise ValueError("max_area must be >= min_area")

        self.min_area = min_area
        self.max_area = max_area

    def detect(self, foreground_mask: np.ndarray) -> List[MotionRegion]:
        if not isinstance(foreground_mask, np.ndarray):
            raise TypeError("foreground_mask must be a NumPy array")

        if foreground_mask.size == 0:
            raise ValueError("foreground_mask cannot be empty")

        if foreground_mask.dtype != np.uint8:
            raise TypeError("foreground_mask must have uint8 dtype")

        if foreground_mask.ndim != 2:
            raise ValueError("foreground_mask must be a single-channel image")

        # findContours may modify its input, so use a copy.
        contour_input = foreground_mask.copy()

        contours, _ = cv2.findContours(
            contour_input,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        regions = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.min_area:
                continue

            if self.max_area is not None and area > self.max_area:
                continue

            x, y, width, height = cv2.boundingRect(contour)

            regions.append(
                MotionRegion(
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    area=area,
                )
            )

        return regions

    def draw(
        self,
        frame: np.ndarray,
        regions: List[MotionRegion],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        show_area: bool = True,
    ) -> np.ndarray:
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty NumPy array")

        if frame.dtype != np.uint8:
            raise TypeError("frame must have uint8 dtype")

        if frame.ndim not in (2, 3):
            raise ValueError("frame must be grayscale or color")

        if frame.ndim == 3 and frame.shape[2] != 3:
            raise ValueError("color frame must have 3 channels")

        output = frame.copy()

        for region in regions:
            x, y, width, height = region.bbox

            cv2.rectangle(
                output,
                (x, y),
                (x + width, y + height),
                color,
                thickness,
            )

            if show_area:
                label = f"Area: {region.area:.0f}"

                cv2.putText(
                    output,
                    label,
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        return output
