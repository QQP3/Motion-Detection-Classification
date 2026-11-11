
import cv2
import time

from src.input.input_manager import InputManager
from src.input.providers.webcam import WebcamProvider
from src.input.providers.stream_url import StreamURLProvider

from src.detection.preprocessing import FramePreprocessor
from src.detection.background_subtractor import BackgroundSubtractor
from src.detection.mask_cleaner import MaskCleaner
from src.detection.motion_detector import MotionDetector
from src.detection.object_tracker import ObjectTracker


STREAM_URL = (
    "https://video.dot.state.mn.us/public/"
    "C661.stream/playlist.m3u8"
)


def create_pipeline():
    """Create independent detection components for one source."""

    return {
        "preprocessor": FramePreprocessor(),
        "background_subtractor": BackgroundSubtractor(
            history=500,
            var_threshold=16,
            detect_shadows=False,
        ),
        "mask_cleaner": MaskCleaner(
            kernel_size=(3, 3),
            opening_iterations=1,
            closing_iterations=2,
        ),
        "motion_detector": MotionDetector(min_area=500),
        "object_tracker": ObjectTracker(
            max_disappeared=10,
            max_distance=50,
        ),
    }



def process_frame(frame, pipeline):
    """Process one frame and measure individual processing stages."""

    timings = {}

    start = time.perf_counter()
    processed_frame = pipeline["preprocessor"].process(frame)
    timings["preprocessing"] = (
        time.perf_counter() - start
    ) * 1000

    if processed_frame is None:
        return None, None, timings

    start = time.perf_counter()
    foreground_mask = pipeline["background_subtractor"].apply(
        processed_frame
    )
    timings["background_subtraction"] = (
        time.perf_counter() - start
    ) * 1000

    start = time.perf_counter()
    cleaned_mask = pipeline["mask_cleaner"].clean(
        foreground_mask
    )
    timings["mask_cleaning"] = (
        time.perf_counter() - start
    ) * 1000

    start = time.perf_counter()
    regions = pipeline["motion_detector"].detect(
        cleaned_mask
    )
    timings["motion_detection"] = (
        time.perf_counter() - start
    ) * 1000

    start = time.perf_counter()
    tracked_objects = pipeline["object_tracker"].update(
        regions
    )
    timings["object_tracking"] = (
        time.perf_counter() - start
    ) * 1000

    start = time.perf_counter()

    output_frame = pipeline["motion_detector"].draw(
        frame,
        regions,
        show_area=False,
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

    timings["visualization"] = (
        time.perf_counter() - start
    ) * 1000

    timings["total_processing"] = sum(timings.values())

    return output_frame, cleaned_mask, timings

def main():

    input_manager = InputManager()

    input_manager.add_provider(
        WebcamProvider(
            camera_index=0,
            name="Camera-0",
        )
    )

    input_manager.add_provider(
        StreamURLProvider(
            url=STREAM_URL,
            name="URL-Stream",
        )
    )

    pipelines = {
        "Camera-0": create_pipeline(),
        "URL-Stream": create_pipeline(),
    }

    previous_times = {
        name: time.perf_counter()
        for name in pipelines
    }

    try:
        print("Opening video sources...")

        open_results = input_manager.open_all()

        for name, success in open_results.items():
            print(
                f"{name}: "
                f"{'Opened successfully' if success else 'Failed to open'}"
            )

        active_sources = {
            name for name, success in open_results.items()
            if success
        }

        if not active_sources:
            print("Error: No video sources could be opened.")
            return

        print("\nMulti-source motion detection started.")
        print("Press 'q' to quit.")
        print("Press 'r' to reset all detection pipelines.")

        while active_sources:

            frames = input_manager.read_all()

            for source_name in list(active_sources):

                success, frame = frames.get(
                    source_name,
                    (False, None),
                )

                if not success or frame is None:
                    print(f"Warning: Failed to read from {source_name}.")
                    continue

                pipeline = pipelines[source_name]

                output_frame, cleaned_mask, timings = process_frame(
                    frame,
                    pipeline,
                )

                if output_frame is None:
                    continue
                
                cv2.putText(
                    output_frame,
                    f"Processing: {timings['total_processing']:.1f} ms",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                
                current_time = time.perf_counter()

                elapsed = current_time - previous_times[source_name]
                previous_times[source_name] = current_time

                fps = 1.0 / elapsed if elapsed > 0 else 0.0

                tracked_objects = pipeline["object_tracker"].objects

                cv2.putText(
                    output_frame,
                    f"Source: {source_name}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

                cv2.putText(
                    output_frame,
                    f"FPS: {fps:.1f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

                cv2.putText(
                    output_frame,
                    f"Tracked objects: {len(tracked_objects)}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

                cv2.imshow(
                    f"Motion Detection - {source_name}",
                    output_frame,
                )

                cv2.imshow(
                    f"Cleaned Mask - {source_name}",
                    cleaned_mask,
                )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("r"):
                for pipeline in pipelines.values():
                    pipeline["background_subtractor"].reset()
                    pipeline["object_tracker"].reset()

                print("All detection pipelines reset.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        input_manager.release_all()
        cv2.destroyAllWindows()

        print("All video sources released.")


if __name__ == "__main__":
    main()
