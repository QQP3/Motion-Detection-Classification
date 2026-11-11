
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.input.input_normalizer import InputNormalizer


def main():
    normalizer = InputNormalizer(
        width=640,
        height=480,
        grayscale=False,
    )

    test_frames = {
        "BGR": np.zeros((720, 1280, 3), dtype=np.uint8),
        "Grayscale": np.zeros((480, 640), dtype=np.uint8),
        "BGRA": np.zeros((480, 640, 4), dtype=np.uint8),
    }

    for name, frame in test_frames.items():
        normalized = normalizer.normalize(frame)

        assert normalized is not None
        assert normalized.shape == (480, 640, 3)
        assert normalized.dtype == np.uint8

        print(
            f"{name}: SUCCESS | "
            f"Shape: {normalized.shape} | "
            f"Type: {normalized.dtype}"
        )
        plt.imshow(normalized)
        plt.show()
    # Invalid input test
    assert normalizer.normalize(None) is None

    print("All normalization tests passed!")


if __name__ == "__main__":
    main()
