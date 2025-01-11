import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe modules
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize Haar Cascade for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand detection
    hand_results = hands.process(rgb_frame)
    
    # Process pose detection
    pose_results = pose.process(rgb_frame)
    
    # Convert to grayscale for Haar Cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Face and eye detection
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Hand landmark detection
    if hand_results.multi_hand_landmarks:
        for landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Pose detection and yoga pose recognition
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract landmarks
        landmarks = pose_results.pose_landmarks.landmark

        # Example logic for Warrior I Pose
        front_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        front_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        front_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        back_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        back_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
        back_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

        # Calculate angles
        def calculate_angle(a, b, c):
            a = np.array([a.x, a.y])
            b = np.array([b.x, b.y])
            c = np.array([c.x, c.y])
            ba = a - b
            bc = c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)
            return np.degrees(angle)

        front_leg_angle = calculate_angle(front_hip, front_knee, front_ankle)
        back_leg_angle = calculate_angle(back_hip, back_knee, back_ankle)
        hand_distance = abs(left_wrist.x - right_wrist.x)
        
        if abs(front_leg_angle - 90) < 10 and abs(back_leg_angle - 180) < 10 and 0.1 < hand_distance < 0.3:
            cv2.putText(frame, "Warrior I Pose Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Example logic for Warrior II Pose
        shoulder_distance = abs(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x - 
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x)
        if abs(front_leg_angle - 90) < 10 and abs(back_leg_angle - 180) < 10 and shoulder_distance > 0.5:
            cv2.putText(frame, "Warrior II Pose Detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Example logic for Tree Pose
        if abs(back_leg_angle - 180) < 10 and hand_distance < 0.05:
            cv2.putText(frame, "Tree Pose Detected", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    
    cv2.imshow("Hand, Face, and Yoga Pose Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

