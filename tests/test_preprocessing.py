
import cv2
import numpy as np

from src.detection.preprocessing import FramePreprocessor


def main():
    preprocessor = FramePreprocessor(
        blur_kernel=(5, 5),
        grayscale=True,
    )

    # Test 1: BGR frame
    bgr_frame = np.zeros(
        (480, 640, 3), dtype=np.uint8
    )

    result = preprocessor.process(bgr_frame)

    assert result is not None
    assert result.shape == (480, 640)
    assert result.dtype == np.uint8

    print("BGR to grayscale: SUCCESS")

    # Test 2: Grayscale frame
    gray_frame = np.zeros(
        (480, 640), dtype=np.uint8
    )

    result = preprocessor.process(gray_frame)

    assert result is not None
    assert result.shape == (480, 640)

    print("Grayscale processing: SUCCESS")

    # Test 3: BGRA frame
    bgra_frame = np.zeros(
        (480, 640, 4), dtype=np.uint8
    )

    result = preprocessor.process(bgra_frame)

    assert result is not None
    assert result.shape == (480, 640)

    print("BGRA processing: SUCCESS")

    # Test 4: Invalid input
    assert preprocessor.process(None) is None

    print("Invalid input handling: SUCCESS")

    # Test 5: Invalid blur kernel
    try:
        FramePreprocessor(blur_kernel=(4, 4))
        raise AssertionError("Even kernel should be rejected.")
    except ValueError:
        print("Blur kernel validation: SUCCESS")

    print("\nAll preprocessing tests passed!")


if __name__ == "__main__":
    main()
