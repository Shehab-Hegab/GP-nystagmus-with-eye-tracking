import cv2
import numpy as np
import datetime
import mediapipe as mp
import pandas as pd
from pathlib import Path
import os
from openpyxl import Workbook, load_workbook
import datetime

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Initialize video capture (camera)
cap = cv2.VideoCapture(0)  # Use 0 for default camera, or adjust index for other cameras
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Function to calculate the Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Initialize variables for velocity, frequency, and amplitude
previous_displacement_left, previous_displacement_right = 0, 0
previous_distance_left_by_mm, previous_distance_right_by_mm = 0, 0
oscillation_count_left, oscillation_count_right = 0, 0
max_distance_left_eye, max_distance_right_eye = 0, 0
min_distance_left_eye, min_distance_right_eye = 100, 100
max_amplitude_left, max_amplitude_right = 0, 0
frame_index = 0
Time = 0
Time_with_face = 0
previous_Time = 0

# Data for export
data = [["Index", "Time (s)", "Time with face", "Left Move (mm)", "Right Move (mm)", "Left Velocity (mm/s)", 
         "Right Velocity (mm/s)", "Left Frequency (Hz)", "Right Frequency (Hz)", "Left Amplitude (mm)", "Right Amplitude (mm)"]]

# Directory to save Excel file
output_dir = r"F:\GP\ML\LiveData"
os.makedirs(output_dir, exist_ok=True)


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from webcam.")
        break
        
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Face Mesh
    results = face_mesh.process(rgb_frame)
    
    # Calculate FPS and frame time
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 FPS if not available
    frame_time = 1 / fps
    
    frame_index += 1
    velocity_left, velocity_right = 0, 0
    displacement_left, displacement_right = 0, 0
    frequency_left, frequency_right = 0, 0
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            height, width, _ = frame.shape
            # --- Top nose ---
            top_nose = (
                int(face_landmarks.landmark[6].x * width),
                int(face_landmarks.landmark[6].y * height)
            )
            cv2.circle(frame, top_nose, 3, (0, 0, 255), -1)
            
            # --- Left Eye ---
            left_eye_center = (
                int(face_landmarks.landmark[468].x * width),
                int(face_landmarks.landmark[468].y * height)
            )
            cv2.circle(frame, left_eye_center, 3, (255, 0, 0), -1)
            
            # --- Right Eye ---
            right_eye_center = (
                int(face_landmarks.landmark[473].x * width),
                int(face_landmarks.landmark[473].y * height)
            )
            cv2.circle(frame, right_eye_center, 3, (255, 0, 0), -1)
            
            
            
            # --- Eye left edge ---
            left_eye_left_edge = (
                int(face_landmarks.landmark[471].x * width),
                int(face_landmarks.landmark[471].y * height)
            )
            cv2.circle(frame, left_eye_left_edge, 3, (0, 255, 255), -1) #circle 
            # --- Eye right edge ---
            left_eye_right_edge = (
                int(face_landmarks.landmark[469].x * width),
                int(face_landmarks.landmark[469].y * height)
            )
            cv2.circle(frame, left_eye_right_edge, 3, (0, 255, 255), -1) #circle 
            radius_eye_left=euclidean_distance(left_eye_left_edge, left_eye_right_edge)
            distance_left_eye_and_nose=euclidean_distance(top_nose, left_eye_center)
            distance_left_by_mm=float(distance_left_eye_and_nose*12/radius_eye_left)
            if previous_distance_left_by_mm != 0:
                displacement_left = float(distance_left_by_mm - previous_distance_left_by_mm)
                velocity_left = abs(displacement_left / (Time-previous_Time))
                if previous_displacement_left != 0 and (displacement_left*previous_displacement_left) < 0:
                    oscillation_count_left += 1
                previous_displacement_left = displacement_left
                max_distance_left_eye = max(max_distance_left_eye,distance_left_by_mm)
                min_distance_left_eye = min(min_distance_left_eye,distance_left_by_mm)
                max_amplitude_left = (max_distance_left_eye-min_distance_left_eye)/2
            else:
                velocity_left = 0
            previous_distance_left_by_mm = distance_left_by_mm

            frequency_left = oscillation_count_left / Time_with_face if Time_with_face > 0 else 0
            
            
          
            # --- Eye right edge ---
            right_eye_left_edge = (
                int(face_landmarks.landmark[476].x * width),
                int(face_landmarks.landmark[476].y * height)
            )
            cv2.circle(frame, right_eye_left_edge, 3, (0, 255, 255), -1) #circle 
            # --- Eye right edge ---
            right_eye_right_edge = (
                int(face_landmarks.landmark[474].x * width),
                int(face_landmarks.landmark[474].y * height)
            )
            cv2.circle(frame, right_eye_right_edge, 3, (0, 255, 255), -1) #circle 
            radius_eye_right=euclidean_distance(right_eye_left_edge, right_eye_right_edge)
            distance_right_eye_and_nose=euclidean_distance(top_nose, right_eye_center)
            distance_right_by_mm=float(distance_right_eye_and_nose*12/radius_eye_right)
            if previous_distance_right_by_mm != 0:
                displacement_right = float(distance_right_by_mm - previous_distance_right_by_mm)
                velocity_right = abs(displacement_right / (Time-previous_Time))
                if previous_displacement_right != 0 and (displacement_right*previous_displacement_right) < 0:
                    oscillation_count_right += 1
                previous_displacement_right = displacement_right
                max_distance_right_eye = max(max_distance_right_eye,distance_right_by_mm)
                min_distance_right_eye = min(min_distance_right_eye,distance_right_by_mm)
                max_amplitude_right = (max_distance_right_eye-min_distance_right_eye)/2
                
            else:
                velocity_right = 0
            previous_distance_right_by_mm = distance_right_by_mm
            frequency_right = oscillation_count_right / Time_with_face if Time_with_face > 0 else 0
            Time_with_face += frame_time
            previous_Time=Time
          
            
            
            # Add your calculations for displacement, velocity, frequency, and amplitude here...
            # Update Time_with_face
            Time_with_face += frame_time
            previous_Time = Time
            
    # Append data for export
    data.append([frame_index, Time, Time_with_face, displacement_left, displacement_right, 
                 velocity_left, velocity_right, frequency_left, frequency_right, max_amplitude_left, max_amplitude_right])
    
    # Update total time
    Time += frame_time
    
    # Display frame
    cv2.imshow("Webcam Feed", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'ESC'
        break

# Save data to Excel file
output_dir = "F:\GP\ML\LiveData"
# Name of the Excel file
output_file = os.path.join(output_dir, "LiveData.xlsx")

# Create the Excel file if it doesn't exist
if not os.path.exists(output_file):
    workbook = Workbook()
    workbook.save(output_file)
  
# Generate a unique sheet name based on the current time
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
sheet_name = f"Patient_{current_time}"

# Load the workbook safely
try:
    workbook = load_workbook(output_file)
    worksheet = workbook.create_sheet(title=sheet_name)

    # Add the patient data to the new sheet
    for row in data:
        worksheet.append(row)

    # Save the workbook safely
    workbook.save(output_file)
    print(f"Metrics saved to sheet {sheet_name} in {output_file}")

except Exception as e:
    print(f"Error: {e}")
    
    
    

cap.release()
cv2.destroyAllWindows()
