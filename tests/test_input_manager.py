
import cv2

from src.input.input_manager import InputManager
from src.input.providers.webcam import WebcamProvider


def main():
    manager = InputManager()

    manager.add_provider(
        WebcamProvider(camera_index=0, name="Camera-0")
    )

    manager.add_provider(
        WebcamProvider(camera_index=1, name="Camera-1")
    )

    try:
        results = manager.open_all()

        for name, success in results.items():
            print(f"{name}: {'OPEN' if success else 'FAILED'}")

        active_cameras = [
            name for name, success in results.items()
            if success
        ]

        if not active_cameras:
            print("No cameras available.")
            return

        print("Press Q to exit.")

        while True:
            frames = manager.read_all()

            for name, (success, frame) in frames.items():
                if success and frame is not None:
                    cv2.imshow(name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        manager.release_all()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

