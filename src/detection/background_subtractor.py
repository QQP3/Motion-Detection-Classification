
from typing import Optional

import cv2
import numpy as np


class BackgroundSubtractor:
    """Background subtraction using OpenCV MOG2."""

    def __init__(
        self,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = False,
    ):
        if history <= 0:
            raise ValueError("History must be positive.")

        if var_threshold <= 0:
            raise ValueError("Variance threshold must be positive.")

        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows

        self.subtractor = self._create_subtractor()

    def _create_subtractor(self):
        """Create a fresh MOG2 background subtractor."""

        return cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows,
        )

    def apply(
        self,
        frame: Optional[np.ndarray],
        learning_rate: float = -1,
    ) -> Optional[np.ndarray]:
        """Generate a foreground mask from an input frame."""

        if (
            frame is None
            or not isinstance(frame, np.ndarray)
            or frame.size == 0
        ):
            return None

        if frame.dtype != np.uint8:
            return None

        if frame.ndim == 2:
            pass

        elif frame.ndim == 3:
            if frame.shape[2] != 3:
                return None

        else:
            return None

        if not -1 <= learning_rate <= 1:
            raise ValueError(
                "Learning rate must be between -1 and 1."
            )

        mask = self.subtractor.apply(
            frame,
            learningRate=learning_rate,
        )

        return mask

    def reset(self) -> None:
        """Reset the background model."""

        self.subtractor = self._create_subtractor()

    def get_background(self) -> Optional[np.ndarray]:
        """Retrieve the current estimated background."""

        return self.subtractor.getBackgroundImage()
