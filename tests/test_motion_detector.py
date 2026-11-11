
import cv2
import time
import sys

from src.input.providers.webcam import WebcamProvider
from src.detection.preprocessing import FramePreprocessor
from src.detection.background_subtractor import BackgroundSubtractor
from src.detection.motion_detector import MotionDetector
from src.input.providers.video_file import VideoFileProvider
from src.input.providers.stream_url import StreamURLProvider
from src.detection.mask_cleaner import MaskCleaner
from src.detection.object_tracker import ObjectTracker


def main():
    #camera = WebcamProvider(camera_index=0)
    
    if len(sys.argv) < 2:
        print("Usage: python -m tests.test_video_file_provider <video_path>")
        return

    stream = StreamURLProvider(sys.argv[1])

    preprocessor = FramePreprocessor()
    background_subtractor = BackgroundSubtractor(
        history=500,
        var_threshold=16,
        detect_shadows=False,
    )
    motion_detector = MotionDetector(min_area=500)

    if not stream.open():
        print("Error: Could not open camera.")
        return

    print("Motion detection started.")
    print("Press 'q' to quit.")
    print("Press 'r' to reset the background model.")

    previous_time = time.perf_counter()

    mask_cleaner = MaskCleaner(
        kernel_size=(3, 3),
        opening_iterations=1,
        closing_iterations=2,
    )

    object_tracker = ObjectTracker(
        max_disappeared=10,
        max_distance=50,
    )
    try:
        while True:
            success, frame = stream.read()

            if not success or frame is None:
                print("Warning: Could not read camera frame.")
                break

            processed_frame = preprocessor.process(frame)

            if processed_frame is None:
                continue

            foreground_mask = background_subtractor.apply(
                processed_frame
            )

            cleaned_mask = mask_cleaner.clean(foreground_mask)

            regions = motion_detector.detect(cleaned_mask)

            tracked_objects = object_tracker.update(regions)

            output_frame = motion_detector.draw(
                frame,
                regions,
                show_area=True,
            )

            for object_id, tracked_object in tracked_objects.items():
                x, y, width, height = tracked_object.bbox

                cv2.rectangle(
                    output_frame,
                    (x, y),
                    (x + width, y + height),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    output_frame,
                    f"ID: {object_id}",
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time

            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            cv2.putText(
                output_frame,
                f"Motion regions: {len(regions)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                output_frame,
                f"FPS: {fps:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            cv2.imshow("Live Motion Detection", output_frame)
            cv2.imshow("Original Foreground Mask", foreground_mask)
            cv2.imshow("Cleaned Foreground Mask", cleaned_mask)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("r"):
                background_subtractor.reset()
                object_tracker.reset()
                print("Background model reset.")

    finally:
        stream.release()
        cv2.destroyAllWindows()
        print("Motion detection stopped.")


if __name__ == "__main__":
    main()
