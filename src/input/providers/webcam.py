
from typing import Optional, Tuple

import cv2
import numpy as np

from .base_provider import BaseProvider


class WebcamProvider(BaseProvider):
    """Cross-platform webcam input provider."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        name: Optional[str] = None,
    ):
        super().__init__(
            name or f"Webcam-{camera_index}"
        )

        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None

    def open(self) -> bool:
        if self.is_open:
            return True

        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            self.release()
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.is_open = True
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_open or self.cap is None:
            return False, None

        ret, frame = self.cap.read()

        if not ret or frame is None:
            return False, None

        return True, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.is_open = False
