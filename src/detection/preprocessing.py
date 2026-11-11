
from typing import Optional, Tuple

import cv2
import numpy as np


class FramePreprocessor:
    """Prepare video frames for motion detection."""

    def __init__(
        self,
        blur_kernel: Tuple[int, int] = (5, 5),
        grayscale: bool = True,
    ):
        if (
            len(blur_kernel) != 2
            or any(k <= 0 or k % 2 == 0 for k in blur_kernel)
        ):
            raise ValueError(
                "Blur kernel dimensions must be positive odd numbers."
            )

        self.blur_kernel = blur_kernel
        self.grayscale = grayscale

    def process(
        self,
        frame: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Convert, validate, and blur a video frame."""

        if (
            frame is None
            or not isinstance(frame, np.ndarray)
            or frame.size == 0
        ):
            return None

        if frame.dtype != np.uint8:
            return None

        if frame.ndim == 2:
            processed = frame

        elif frame.ndim == 3:
            channels = frame.shape[2]

            if channels == 1:
                processed = frame[:, :, 0]

            elif channels == 3:
                if self.grayscale:
                    processed = cv2.cvtColor(
                        frame, cv2.COLOR_BGR2GRAY
                    )
                else:
                    processed = frame

            elif channels == 4:
                if self.grayscale:
                    processed = cv2.cvtColor(
                        frame, cv2.COLOR_BGRA2GRAY
                    )
                else:
                    processed = cv2.cvtColor(
                        frame, cv2.COLOR_BGRA2BGR
                    )

            else:
                return None

        else:
            return None

        processed = cv2.GaussianBlur(
            processed,
            self.blur_kernel,
            0,
        )

        return processed
