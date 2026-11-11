
import cv2
import numpy as np


class MaskCleaner:
    def __init__(
        self,
        kernel_size=(3, 3),
        opening_iterations=1,
        closing_iterations=2,
    ):
        if len(kernel_size) != 2:
            raise ValueError("kernel_size must contain two dimensions")

        if any(
            not isinstance(value, int) or value <= 0
            for value in kernel_size
        ):
            raise ValueError("kernel dimensions must be positive integers")

        if opening_iterations < 0 or closing_iterations < 0:
            raise ValueError("iterations cannot be negative")

        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            kernel_size,
        )

        self.opening_iterations = opening_iterations
        self.closing_iterations = closing_iterations

    def clean(self, mask: np.ndarray) -> np.ndarray:
        if not isinstance(mask, np.ndarray):
            raise TypeError("mask must be a NumPy array")

        if mask.size == 0:
            raise ValueError("mask cannot be empty")

        if mask.dtype != np.uint8:
            raise TypeError("mask must have uint8 dtype")

        if mask.ndim != 2:
            raise ValueError("mask must be single-channel")

        cleaned = mask.copy()

        # Remove small foreground noise.
        if self.opening_iterations > 0:
            cleaned = cv2.morphologyEx(
                cleaned,
                cv2.MORPH_OPEN,
                self.kernel,
                iterations=self.opening_iterations,
            )

        # Fill small gaps and connect nearby foreground regions.
        if self.closing_iterations > 0:
            cleaned = cv2.morphologyEx(
                cleaned,
                cv2.MORPH_CLOSE,
                self.kernel,
                iterations=self.closing_iterations,
            )

        return cleaned
