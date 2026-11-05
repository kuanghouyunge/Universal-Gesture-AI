
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"


def main():
    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头，请检查摄像头连接")
        return

    print("Universal Gesture AI started.")
    print("Press Q to exit.")

    with vision.HandLandmarker.create_from_options(options) as detector:
        frame_timestamp_ms = 0

        while True:
            success, frame = cap.read()

            if not success:
                print("无法读取摄像头画面")
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            frame_timestamp_ms += 33

            results = detector.detect_for_video(
                mp_image,
                frame_timestamp_ms
            )

            if results.hand_landmarks:
                for hand_index, landmarks in enumerate(results.hand_landmarks):

                    h, w, _ = frame.shape

                    # Draw 21 hand landmarks
                    points = []

                    for landmark in landmarks:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)

                        points.append((x, y))

                        cv2.circle(
                            frame,
                            (x, y),
                            5,
                            (0, 255, 0),
                            -1
                        )

                    # Draw hand connections
                    for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:
                        start = points[connection.start]
                        end = points[connection.end]

                        cv2.line(
                            frame,
                            start,
                            end,
                            (255, 0, 0),
                            2
                        )

                    # Display handedness
                    handedness = results.handedness[hand_index][0].category_name

                    cv2.putText(
                        frame,
                        handedness,
                        (points[0][0], points[0][1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2
                    )

                    print(
                        f"Hand {hand_index + 1}: {handedness}, "
                        f"Landmarks: {len(landmarks)}",
                        end="\r"
                    )

            cv2.imshow(
                "Universal Gesture AI - Hand Detection",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
