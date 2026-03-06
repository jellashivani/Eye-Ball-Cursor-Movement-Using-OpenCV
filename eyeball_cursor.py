import cv2
import mediapipe as mp
import pyautogui

# Start webcam
cam = cv2.VideoCapture(0)

# Get screen size
screen_w, screen_h = pyautogui.size()

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Flip frame
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process face landmarks
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks

    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # Left eye landmark for cursor control
        eye_landmark = landmarks[474:478]

        for lm in eye_landmark:
            x = int(lm.x * frame_w)
            y = int(lm.y * frame_h)

            # Draw circle on eye
            cv2.circle(frame, (x, y), 3, (0, 255, 0))

            # Convert to screen coordinates
            screen_x = screen_w * lm.x
            screen_y = screen_h * lm.y

            # Move mouse
            pyautogui.moveTo(screen_x, screen_y)

        # Blink detection (for click)
        left_eye = landmarks[145]
        right_eye = landmarks[159]

        if (left_eye.y - right_eye.y) < 0.004:
            pyautogui.click()
            pyautogui.sleep(1)

    # Show window
    cv2.imshow("Eye Controlled Mouse", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()