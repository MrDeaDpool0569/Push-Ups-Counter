import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize mediapipe pose class.
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate angle between three points.
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point (vertex)
    c = np.array(c)  # End point

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    # Ensure angle is within the correct range.
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Set up video capture from your webcam.
cap = cv2.VideoCapture(0)

# Initialize variables for push-up counter and stage.
counter = 0
stage = None  # This will hold "up" or "down" status.

# Use MediaPipe Pose.
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Recolor image to RGB for processing.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Process the image and detect the pose.
        results = pose.process(image)
        
        # Recolor back to BGR for rendering.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates for the left shoulder, elbow, and wrist.
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Calculate the angle at the left elbow.
            angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            # Display the calculated angle near the left elbow.
            elbow_coords = tuple(np.multiply(left_elbow, [frame.shape[1], frame.shape[0]]).astype(int))
            cv2.putText(image, str(round(angle, 2)), 
                        elbow_coords, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Push-up counter logic:
            # When your arm is nearly extended, consider it the "up" position.
            if angle > 160:
                stage = "up"
            # When your arm is bent (e.g., less than 90 degrees) and the stage was "up",
            # count one push-up and mark the stage as "down".
            if angle < 90 and stage == "up":
                stage = "down"
                counter += 1
                print("Push-up Count:", counter)
            
        except Exception as e:
            # If landmarks aren't detected, just pass.
            pass

        # Display push-up counter on the image.
        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw the pose annotation on the image.
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        
        # Show the final output.
        cv2.imshow('Push-up Counter', image)
        
        # Break the loop if 'q' is pressed.
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the capture and close windows.
cap.release()
cv2.destroyAllWindows()
