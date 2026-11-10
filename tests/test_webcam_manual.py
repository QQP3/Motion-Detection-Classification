
import cv2


def test_camera(device_id: int) -> None:
    print(f"\nTesting camera device: {device_id}")

    cap = cv2.VideoCapture(device_id)

    if not cap.isOpened():
        print(f"Could not open camera {device_id}")
        return

    print("Camera opened successfully!")

    # Request a reasonable initial resolution.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ret, frame = cap.read()

    if not ret or frame is None:
        print("Camera opened, but no frame was received.")
        cap.release()
        return

    height, width, channels = frame.shape

    print(f"Frame received: {width}x{height}")
    print(f"Channels: {channels}")

    print("Press Q or ESC to close the camera window.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame.")
            break

        cv2.imshow(f"Camera {device_id}", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    for camera_id in [0, 1]:
        test_camera(camera_id)
