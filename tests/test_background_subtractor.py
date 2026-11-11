
import sys
import time
import cv2
import numpy as np

from src.input.providers.video_file import VideoFileProvider
from src.detection.preprocessing import FramePreprocessor
from src.detection.background_subtractor import BackgroundSubtractor
from src.input.providers.stream_url import StreamURLProvider
from src.input.providers.webcam import WebcamProvider



def main():

    #if len(sys.argv) < 2:
    #    print(
    #        "Usage: python -m tests.test_background_subtractor "
    #        "<video_path>"
    #    )
    #    return
#
    #video_path = sys.argv[1]

    camera = WebcamProvider()

    preprocessor = FramePreprocessor(
        blur_kernel=(5, 5),
        grayscale=True,
    )

    subtractor = BackgroundSubtractor(
        history=500,
        var_threshold=16,
        detect_shadows=False,
    )

    try:
        if not camera.open():
            print("ERROR: Could not open video.")
            return

        #fps = video.get_fps()
#
        #if not np.isfinite(fps) or fps <= 0:
        #    fps = 30.0
#
        #frame_count = video.get_frame_count()

        #frame_period = 1.0 / fps

        print("camera opened successfully.")
        #print(f"FPS: {fps:.2f}")
        #print(f"Total frames: {frame_count}")
        print("Press Q to exit.")
        print("Press SPACE to pause/resume.")
        print("Press R to reset background model.")

        frame_number = 0
        paused = False

        while True:

            if paused:
                key = cv2.waitKey(30) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord(" "):
                    paused = False
                elif key == ord("r"):
                    subtractor.reset()

                continue

            frame_start = time.perf_counter()

            success, frame = camera.read()

            if not success:
                print("\nVideo playback finished.")
                break

            frame_number += 1

            # Step 1: Preprocess frame
            processed = preprocessor.process(frame)

            if processed is None:
                continue

            # Step 2: Background subtraction
            mask = subtractor.apply(processed)

            if mask is None:
                continue

            # Step 3: Extract foreground pixels
            foreground = cv2.bitwise_and(
                frame,
                frame,
                mask=mask
            )

            # Step 4: Create visualization
            display_frame = frame.copy()

            foreground_color = cv2.cvtColor(
                mask,
                cv2.COLOR_GRAY2BGR
            )

            cv2.putText(
                display_frame,
                "Original Video",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                processed,
                "Preprocessed",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                255,
                2
            )

            cv2.putText(
                foreground_color,
                "Foreground Mask",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,),
                2
            )

            cv2.putText(
                foreground,
                "Detected Foreground",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # Display all processing stages
            cv2.imshow("1 - Original", display_frame)
            cv2.imshow("2 - Preprocessed", processed)
            cv2.imshow("3 - Foreground Mask", foreground_color)
            cv2.imshow("4 - Foreground Result", foreground)

            # Progress information
            #if frame_count > 0:
            #    progress = frame_number / frame_count * 100
#
            #    print(
            #        f"\rFrame: {frame_number}/{frame_count} "
            #        f"| Progress: {progress:.1f}%",
            #        end="",
            #        flush=True
            #    )

            # Maintain original video frame rate
            elapsed = time.perf_counter() - frame_start

            #remaining = frame_period - elapsed

            delay = 1#max(1, round(remaining * 1000))

            key = cv2.waitKey(delay) & 0xFF

            if key == ord("q"):
                break

            elif key == ord(" "):
                paused = True

            elif key == ord("r"):
                subtractor.reset()

    finally:
        camera.release()
        cv2.destroyAllWindows()

        print("\nResources released.")


if __name__ == "__main__":
    main()
