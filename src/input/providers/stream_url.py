
from typing import Optional, Tuple

import cv2
import numpy as np

from .base_provider import BaseProvider


class StreamURLProvider(BaseProvider):
    """Network video stream input provider."""

    def __init__(
        self,
        url: str,
        name: Optional[str] = None,
    ):
        super().__init__(name or "NetworkStream")

        self.url = url
        self.cap = None

    def open(self) -> bool:
        if self.is_open:
            return True

        self.cap = cv2.VideoCapture(self.url)

        if not self.cap.isOpened():
            self.release()
            return False

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

    def get_fps(self) -> float:
        if self.cap is None:
            return 0.0

        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_resolution(self) -> Tuple[int, int]:
        if self.cap is None:
            return 0, 0

        width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        return width, height
