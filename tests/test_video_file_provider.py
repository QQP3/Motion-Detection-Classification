
import sys
import cv2

from src.input.providers.video_file import VideoFileProvider


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m tests.test_video_file_provider <video_path>")
        return

    video = VideoFileProvider(sys.argv[1])

    try:
        if not video.open():
            print("ERROR: Could not open video.")
            return

        print("Video opened successfully.")
        print("FPS:", video.get_fps())
        print("Frame count:", video.get_frame_count())
        print("Resolution:", video.get_resolution())
        print("Press Q to exit.")

        fps = video.get_fps()

#       If FPS is invalid go with a common FPS 
        if fps <= 0:
            fps = 30.0
        delay = max(1, int(1000 / fps))

        while True:
            success, frame = video.read()

            if not success:
                print("Video playback finished.")
                break

            cv2.imshow("Video File Provider Test", frame)

            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break

    finally:
        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

