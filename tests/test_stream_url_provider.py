
import sys
import cv2

from src.input.providers.stream_url import StreamURLProvider


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python -m tests.test_stream_url_provider <stream_url>"
        )
        return

    stream = StreamURLProvider(sys.argv[1])

    try:
        if not stream.open():
            print("ERROR: Could not open network stream.")
            return

        print("Stream opened successfully.")
        print("FPS:", stream.get_fps())
        print("Resolution:", stream.get_resolution())
        print("Press Q to exit.")

        fps = stream.get_fps()

        if fps <= 0:
            fps = 30.0

        delay = max(1, round(1000 / fps))

        while True:
            success, frame = stream.read()

            if not success:
                print("Stream frame unavailable.")
                break

            cv2.imshow("Network Stream Test", frame)

            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break

    finally:
        stream.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
