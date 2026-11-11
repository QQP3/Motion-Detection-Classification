
from typing import Optional, Tuple

import cv2
import numpy as np


class InputNormalizer:
    """Normalize video frames before processing."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        grayscale: bool = False,
    ):
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive.")

        self.width = width
        self.height = height
        self.grayscale = grayscale

    def normalize(
        self,
        frame: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Resize and normalize a video frame."""

        if frame is None or not isinstance(frame, np.ndarray):
            return None

        if frame.size == 0:
            return None

        if frame.ndim == 2:
            frame = cv2.cvtColor(
                frame, cv2.COLOR_GRAY2BGR
            )

        elif frame.ndim == 3:
            channels = frame.shape[2]

            if channels == 1:
                frame = cv2.cvtColor(
                    frame, cv2.COLOR_GRAY2BGR
                )

            elif channels == 4:
                frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGRA2BGR
                )

            elif channels != 3:
                return None

        else:
            return None

        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating):
                if not np.all(np.isfinite(frame)):
                    return None

                if frame.min() >= 0 and frame.max() <= 1:
                    frame = frame * 255.0

            frame = np.clip(frame, 0, 255).astype(np.uint8)

        frame = cv2.resize(
            frame,
            (self.width, self.height),
            interpolation=cv2.INTER_LINEAR,
        )

        if self.grayscale:
            frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2GRAY
            )

        return frame

    def get_output_shape(self) -> Tuple[int, ...]:
        """Return expected normalized frame shape."""

        if self.grayscale:
            return self.height, self.width

        return self.height, self.width, 3
