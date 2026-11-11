
from src.input.providers.webcam import WebcamProvider


def main():
    camera = WebcamProvider(
        camera_index=0,
        width=640,
        height=480,
    )

    try:
        if not camera.open():
            print("ERROR: Could not open webcam.")
            return

        print("Webcam opened successfully.")
        print("Press Q to exit.")

        while True:
            success, frame = camera.read()

            if not success:
                print("Frame capture failed.")
                break

            cv2.imshow("Webcam Provider Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import cv2
    main()
