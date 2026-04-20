import cv2
import dlib
from scipy.spatial import distance

# ---------- FUNCTION ----------
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ---------- LOAD MODELS ----------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# ---------- CAMERA ----------
cap = cv2.VideoCapture("http://192.168.1.69:81/stream")  # replace IP

# ---------- SETTINGS ----------
EYE_THRESH = 0.25
EYE_FRAMES = 20
counter = 0

# ---------- LOOP ----------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    # Resize for speed
    frame = cv2.resize(frame, (320, 240))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)

        # LEFT EYE
        left_eye = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
        # RIGHT EYE
        right_eye = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]

        leftEAR = eye_aspect_ratio(left_eye)
        rightEAR = eye_aspect_ratio(right_eye)

        ear = (leftEAR + rightEAR) / 2.0

        # Draw face box
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # DROWSINESS LOGIC
        if ear < EYE_THRESH:
            counter += 1

            if counter >= EYE_FRAMES:
                cv2.putText(frame, "SLEEPING!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            counter = 0

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) == 27:
        break

# ---------- CLEANUP ----------
cap.release()
cv2.destroyAllWindows()