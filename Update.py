import cv2
import numpy as np
import datetime
import mediapipe as mp
import pandas as pd
import os
from openpyxl import Workbook, load_workbook
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import collections # For deque to store history for variance, acceleration, plotting

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5, # Increased for robustness
    min_tracking_confidence=0.5, # Increased for robustness
)

# -----------------------------------------------
# OPTION 1: Use webcam
USE_WEBCAM = True
VIDEO_FILE_PATH = "path/to/your/video.mp4" # Only used if USE_WEBCAM is False

if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    print("Using webcam input.")
else:
    if not os.path.exists(VIDEO_FILE_PATH):
        print(f"Error: Video file not found at {VIDEO_FILE_PATH}")
        exit()
    cap = cv2.VideoCapture(VIDEO_FILE_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {VIDEO_FILE_PATH}.")
        exit()
    print(f"Using video file input: {VIDEO_FILE_PATH}")
# -----------------------------------------------

# Function to calculate Euclidean distance between two 3D points
def euclidean_distance_3d(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 +
                  (point1[1] - point2[1])**2 +
                  (point1[2] - point2[2])**2)

# Removed: eye_aspect_ratio_3d function and related EAR landmark indices

# Landmark indices for eyes (MediaPipe Face Mesh)
LEFT_EYE_CENTER_LM = 468 # Left iris center
RIGHT_EYE_CENTER_LM = 473 # Right iris center

LEFT_EYE_WIDTH_LMS = [362, 263] # Left eye: inner corner (362) to outer corner (263)
RIGHT_EYE_WIDTH_LMS = [133, 33]  # Right eye: inner corner (133) to outer corner (33)

# Initialize tracking variables (3D)
prev_x_left, prev_y_left, prev_z_left = 0.0, 0.0, 0.0
prev_x_right, prev_y_right, prev_z_right = 0.0, 0.0, 0.0

# Velocity history for acceleration calculation (normalized units)
prev_vx_left, prev_vy_left, prev_vz_left = 0.0, 0.0, 0.0
prev_vx_right, prev_vy_right, prev_vz_right = 0.0, 0.0, 0.0

# Initialize variables for frequency oscillation tracking (using X-axis velocity for frequency)
prev_vx_for_freq_left = 0.0 # Storing previous velocities for sign change detection
prev_vx_for_freq_right = 0.0
oscillation_count_left_x = 0
oscillation_count_right_x = 0

frame_index = 0
Time = 0.0 # Total time elapsed (including no-face frames)
Time_with_face = 0.0 # Time when a face was detected

# Removed: Blink detection variables (EAR_THRESHOLD, BLINK_CONSEC_FRAMES, is_blinking_left, etc.)

# History for variance calculation (normalized displacement, e.g., last 30 frames)
HISTORY_LEN = 30
dx_history_left = collections.deque(maxlen=HISTORY_LEN)
dy_history_left = collections.deque(maxlen=HISTORY_LEN)
dz_history_left = collections.deque(maxlen=HISTORY_LEN)
dx_history_right = collections.deque(maxlen=HISTORY_LEN)
dy_history_right = collections.deque(maxlen=HISTORY_LEN)
dz_history_right = collections.deque(maxlen=HISTORY_LEN)

# Model and data paths - NOW A LIST TO INCLUDE OLD DATA
# Ensure these paths are correct for your system
DATA_PATHS = [
    r"F:\GP\ML\LiveData\LiveData.xlsx",   # New data will be saved here
    r"F:\GP\ML\LiveData\Live-Old-Data.xlsx" # Your old data with patients labeled '1'
]
MODEL_PATH = 'tremor_model.joblib'
SCALER_PATH = 'tremor_scaler.joblib'

# Define the full list of features for the ML model (now without blink features)
ALL_ML_FEATURES = [
    'Left_Move_Magnitude_mm', 'Right_Move_Magnitude_mm',
    'Left_Vel_Magnitude_mm_s', 'Right_Vel_Magnitude_mm_s',
    'Left_Freq_Hz', 'Right_Freq_Hz',
    'Left_DX_Norm', 'Left_DY_Norm', 'Left_DZ_Norm',
    'Left_VX_Norm', 'Left_VY_Norm', 'Left_VZ_Norm',
    'Right_DX_Norm', 'Right_DY_Norm', 'Right_DZ_Norm',
    'Right_VX_Norm', 'Right_VY_Norm', 'Right_VZ_Norm',
    'Left_DX_Variance', 'Left_DY_Variance', 'Left_DZ_Variance',
    'Right_DX_Variance', 'Right_DY_Variance', 'Right_DZ_Variance',
    'Left_X_Acceleration_Norm', 'Left_Y_Acceleration_Norm', 'Left_Z_Acceleration_Norm',
    'Right_X_Acceleration_Norm', 'Right_Y_Acceleration_Norm', 'Right_Z_Acceleration_Norm'
]

# No features to exclude from ALL_ML_FEATURES anymore, as blink features are already removed
FEATURES_TO_EXCLUDE = [] 

ML_FEATURES_LIST = [f for f in ALL_ML_FEATURES if f not in FEATURES_TO_EXCLUDE]
print(f"ML Model will use {len(ML_FEATURES_LIST)} features: {ML_FEATURES_LIST}")


# Initialize model and scaler
model = None
scaler = None

# Try to load a pre-trained model
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = load(MODEL_PATH)
        scaler = load(SCALER_PATH)
        print("Loaded pre-trained model successfully!")
        # Verify loaded scaler expects correct number of features
        if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != len(ML_FEATURES_LIST):
            print(f"WARNING: Loaded scaler expects {scaler.n_features_in_} features, but {len(ML_FEATURES_LIST)} are being provided. Forcing retraining.")
            raise ValueError("Scaler feature count mismatch")
    else:
        raise FileNotFoundError # Force retraining if files are missing
except Exception as e:
    print(f"No pre-trained model found or mismatch: {e}. Attempting to train a new model...")
    
    combined_df = pd.DataFrame() # Initialize an empty DataFrame to combine all data
    
    for path in DATA_PATHS:
        if os.path.exists(path):
            print(f"Attempting to load data from: {path}")
            try:
                temp_wb = load_workbook(path)
                # Only load sheets that start with "Session_" for training data
                sheet_names_to_read = [s for s in temp_wb.sheetnames if s.startswith("Session_")]
                
                if not sheet_names_to_read:
                    print(f"No 'Session_' sheets found in {path}. Skipping this file for training.")
                else:
                    print(f"Found sheets in {path} starting with 'Session_': {sheet_names_to_read}") # DEBUG PRINT
                    for sheet_name in sheet_names_to_read:
                        sheet_df = pd.read_excel(path, sheet_name=sheet_name)
                        # Check if sheet is empty or only contains header before adding
                        if not sheet_df.empty and len(sheet_df.columns) > 1: # Basic check for meaningful data
                            combined_df = pd.concat([combined_df, sheet_df], ignore_index=True)
                            print(f"Successfully loaded sheet '{sheet_name}' from {path}. Total rows in combined_df: {len(combined_df)}") # DEBUG PRINT
                        else:
                            print(f"Skipping empty or malformed sheet: '{sheet_name}' in {path}.") # DEBUG PRINT
            except Exception as read_error:
                print(f"FATAL ERROR: Could not load data from Excel file: {path} - {read_error}.")
                print(f"Please ENSURE '{path}' is closed, not corrupted, and saved as a proper Excel .xlsx file.")
                # This error is critical for training, so we'll continue trying other files but note it.
                model = None # Ensure model is not trained if a critical file fails
                scaler = None # Ensure scaler is not trained if a critical file fails
        else:
            print(f"Training data file not found: {path}. Skipping this file for training.")

    # Only try to train if no fatal error prevented it and combined_df is not empty
    if model is None and not combined_df.empty: 
        # Ensure all required ML features exist in the combined DataFrame, filling missing with 0.0
        for feature in ML_FEATURES_LIST:
            if feature not in combined_df.columns:
                print(f"WARNING: Feature '{feature}' not found in combined training data. Adding with default value 0.0. "
                        "This will result in poor model performance unless populated with real data.")
                combined_df[feature] = 0.0 
        
        # Add Label column if missing
        if 'Label' not in combined_df.columns:
            print("WARNING: 'Label' column not found in training data. Adding with default value 0. "
                    "Please label your data (0 or 1 for example) for effective training.")
            combined_df['Label'] = 0 
        
        # Drop rows with any missing values for relevant columns
        columns_for_dropna = ML_FEATURES_LIST + ['Label']
        # Filter columns_for_dropna to only include columns that actually exist in combined_df.columns
        existing_cols_for_dropna = [col for col in columns_for_dropna if col in combined_df.columns]
        combined_df = combined_df.dropna(subset=existing_cols_for_dropna)
        
        if len(combined_df) == 0:
            print("No valid data rows after dropping NaNs. Cannot train model.")
            model = None
            scaler = None
        else:
            X = combined_df[ML_FEATURES_LIST]
            y = combined_df['Label']
            
            # --- DEBUGGING STEP: Print unique labels and their counts ---
            print(f"\n--- Training Data Labels Found ---")
            print(f"Total rows for training: {len(y)}")
            print(f"Unique labels and their counts:\n{y.value_counts()}") # Changed to print on new line for clarity
            print(f"---------------------------------\n")

            # Check if there's enough data for splitting and if y has at least two classes
            if len(X) < 2 or len(y.unique()) < 2:
                print(f"Not enough data ({len(X)} rows) or unique classes ({len(y.unique())}) for effective training. "
                        "Need at least 2 rows and 2 classes (e.g., 0 and 1) in the 'Label' column. Model training skipped.")
                model = None
                scaler = None
            else:
                # Split the dataset into training and testing sets, stratifying to maintain class balance
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train the RandomForest model
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42,
                    class_weight='balanced' # Helps with imbalanced classes
                )
                model.fit(X_train_scaled, y_train)
                
                # Save the trained model and scaler
                dump(model, MODEL_PATH)
                dump(scaler, SCALER_PATH)
                print(f"New model trained successfully! Accuracy: {model.score(X_test_scaled, y_test):.2f}")
                
    else: # This branch is taken if combined_df is empty or a FATAL_ERROR occurred
        print("No valid data collected across all specified data paths for training. Real-time predictions are disabled.")
        model = None
        scaler = None

# Define the Excel output header explicitly, including all ML features and 'Label'
# Filter out the excluded features from the displayed columns too
EXCEL_OUTPUT_HEADER_BASE = [
    "Index", "Time (s)", "TimeWithFace",
    "LeftMove(mm)", "RightMove(mm)",
    "LeftVel(mm/s)", "RightVel(mm/s)",
    "LeftFreq(Hz)", "RightFreq(Hz)", # Frequency is now X-vel based
    "LeftAmp(mm)", "RightAmp(mm)"
]

# Construct the final EXCEL_OUTPUT_HEADER based on ML_FEATURES_LIST (which is now ALL_ML_FEATURES without blink features)
EXCEL_OUTPUT_HEADER = EXCEL_OUTPUT_HEADER_BASE + ML_FEATURES_LIST + ["Prediction", "Label"]

# Initialize data list with the combined header
data_for_excel = [EXCEL_OUTPUT_HEADER]

# Real-time graph visualization variables
PLOT_WIDTH, PLOT_HEIGHT = 600, 300
plot_img = np.zeros((PLOT_HEIGHT, PLOT_WIDTH, 3), dtype=np.uint8)
plot_history_len = 200 # Number of frames to show in graph
vx_left_plot_history = collections.deque(maxlen=plot_history_len)
vy_left_plot_history = collections.deque(maxlen=plot_history_len)
vx_right_plot_history = collections.deque(maxlen=plot_history_len)
vy_right_plot_history = collections.deque(maxlen=plot_history_len)

# Min/Max for plotting normalized velocities (adjust as needed based on observed values)
MAX_VELOCITY_PLOT = 0.05
MIN_VELOCITY_PLOT = -0.05

def normalize_for_plot(value, min_val, max_val, plot_height):
    # Scale value to fit within plot_height (0 to plot_height)
    if max_val == min_val: return plot_height // 2 # Avoid division by zero
    normalized_value = (value - min_val) / (max_val - min_val)
    # Invert y-axis for plot: 0 at top, plot_height at bottom
    return int(plot_height - normalized_value * plot_height)

# Main processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break
    
    frame = cv2.flip(frame, 1) # Flip horizontally for webcam
    h, w, _ = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_time = 1 / fps
    
    # Initialize prediction text to "No face detected" by default
    prediction_text = "No face detected"
    
    # Initialize all features to 0.0 or default values for the current frame
    current_left_move_mm = 0.0
    current_right_move_mm = 0.0
    current_left_vel_mm_s = 0.0
    current_right_vel_mm_s = 0.0
    current_left_freq_hz = 0.0 
    current_right_freq_hz = 0.0 
    current_left_amp_mm = 0.0
    current_right_amp_mm = 0.0

    left_dx_norm, left_dy_norm, left_dz_norm = 0.0, 0.0, 0.0
    left_vx_norm, left_vy_norm, left_vz_norm = 0.0, 0.0, 0.0
    right_dx_norm, right_dy_norm, right_dz_norm = 0.0, 0.0, 0.0
    right_vx_norm, right_vy_norm, right_vz_norm = 0.0, 0.0, 0.0

    left_dx_var, left_dy_var, left_dz_var = 0.0, 0.0, 0.0
    right_dx_var, right_dy_var, right_dz_var = 0.0, 0.0, 0.0

    left_ax_norm, left_ay_norm, left_az_norm = 0.0, 0.0, 0.0
    right_ax_norm, right_ay_norm, right_az_norm = 0.0, 0.0, 0.0

    face_detected_this_frame = False
    
    # Initialize raw_features_map with zeros. This will be used for Excel output
    raw_features_map = {feature: 0.0 for feature in ALL_ML_FEATURES}

    if results.multi_face_landmarks:
        face_detected_this_frame = True
        Time_with_face += frame_time
        
        for face_landmarks in results.multi_face_landmarks:
            left_eye_center_3d = (
                face_landmarks.landmark[LEFT_EYE_CENTER_LM].x,
                face_landmarks.landmark[LEFT_EYE_CENTER_LM].y,
                face_landmarks.landmark[LEFT_EYE_CENTER_LM].z
            )
            left_eye_width_point1 = (
                face_landmarks.landmark[LEFT_EYE_WIDTH_LMS[0]].x,
                face_landmarks.landmark[LEFT_EYE_WIDTH_LMS[0]].y,
                face_landmarks.landmark[LEFT_EYE_WIDTH_LMS[0]].z
            )
            left_eye_width_point2 = (
                face_landmarks.landmark[LEFT_EYE_WIDTH_LMS[1]].x,
                face_landmarks.landmark[LEFT_EYE_WIDTH_LMS[1]].y,
                face_landmarks.landmark[LEFT_EYE_WIDTH_LMS[1]].z
            )
            
            right_eye_center_3d = (
                face_landmarks.landmark[RIGHT_EYE_CENTER_LM].x,
                face_landmarks.landmark[RIGHT_EYE_CENTER_LM].y,
                face_landmarks.landmark[RIGHT_EYE_CENTER_LM].z
            )
            right_eye_width_point1 = (
                face_landmarks.landmark[RIGHT_EYE_WIDTH_LMS[0]].x,
                face_landmarks.landmark[RIGHT_EYE_WIDTH_LMS[0]].y,
                face_landmarks.landmark[RIGHT_EYE_WIDTH_LMS[0]].z
            )
            right_eye_width_point2 = (
                face_landmarks.landmark[RIGHT_EYE_WIDTH_LMS[1]].x,
                face_landmarks.landmark[RIGHT_EYE_WIDTH_LMS[1]].y,
                face_landmarks.landmark[RIGHT_EYE_WIDTH_LMS[1]].z
            )
            
            left_eye_width_3d = euclidean_distance_3d(left_eye_width_point1, left_eye_width_point2)
            right_eye_width_3d = euclidean_distance_3d(right_eye_width_point1, right_eye_width_point2)
            
            if left_eye_width_3d < 1e-6: left_eye_width_3d = 1e-6
            if right_eye_width_3d < 1e-6: right_eye_width_3d = 1e-6

            SCALING_FACTOR_MM_PER_NORM_UNIT = 24 / left_eye_width_3d # Use left eye for a common factor
            
            # Normalized displacements (current - previous)
            left_dx_norm = left_eye_center_3d[0] - prev_x_left
            left_dy_norm = left_eye_center_3d[1] - prev_y_left
            left_dz_norm = left_eye_center_3d[2] - prev_z_left
            
            right_dx_norm = right_eye_center_3d[0] - prev_x_right
            right_dy_norm = right_eye_center_3d[1] - prev_y_right
            right_dz_norm = right_eye_center_3d[2] - prev_z_right
            
            # Convert overall displacement magnitude to 'mm' for Excel columns
            current_left_move_mm = euclidean_distance_3d((0,0,0), (left_dx_norm, left_dy_norm, left_dz_norm)) * SCALING_FACTOR_MM_PER_NORM_UNIT
            current_right_move_mm = euclidean_distance_3d((0,0,0), (right_dx_norm, right_dy_norm, right_dz_norm)) * SCALING_FACTOR_MM_PER_NORM_UNIT
            
            # Normalized Velocities and Accelerations
            if frame_time > 0:
                left_vx_norm = left_dx_norm / frame_time
                left_vy_norm = left_dy_norm / frame_time
                left_vz_norm = left_dz_norm / frame_time
                
                right_vx_norm = right_dx_norm / frame_time
                right_vy_norm = right_dy_norm / frame_time
                right_vz_norm = right_dz_norm / frame_time

                # Current velocities in mm/s for Excel output (magnitude)
                current_left_vel_mm_s = euclidean_distance_3d((0,0,0), (left_vx_norm, left_vy_norm, left_vz_norm)) * SCALING_FACTOR_MM_PER_NORM_UNIT
                current_right_vel_mm_s = euclidean_distance_3d((0,0,0), (right_vx_norm, right_vy_norm, right_vz_norm)) * SCALING_FACTOR_MM_PER_NORM_UNIT

                # Accelerations (normalized units)
                left_ax_norm = (left_vx_norm - prev_vx_left) / frame_time
                left_ay_norm = (left_vy_norm - prev_vy_left) / frame_time
                left_az_norm = (left_vz_norm - prev_vz_left) / frame_time

                right_ax_norm = (right_vx_norm - prev_vx_right) / frame_time
                right_ay_norm = (right_vy_norm - prev_vy_right) / frame_time
                right_az_norm = (right_vz_norm - prev_vz_right) / frame_time
            else:
                left_vx_norm, left_vy_norm, left_vz_norm = 0.0, 0.0, 0.0
                right_vx_norm, right_vy_norm, right_vz_norm = 0.0, 0.0, 0.0
                left_ax_norm, left_ay_norm, left_az_norm = 0.0, 0.0, 0.0
                right_ax_norm, right_ay_norm, right_az_norm = 0.0, 0.0, 0.0
            
            # --- Oscillation Count and Frequency (FIXED) ---
            # Count sign changes of X-velocity for frequency estimation
            # A 0.0 threshold is used to filter out noise near zero velocity.
            VELOCITY_ZERO_THRESHOLD = 0.001 # Small threshold to avoid noisy zero-crossings
            
            if (left_vx_norm > VELOCITY_ZERO_THRESHOLD and prev_vx_for_freq_left < -VELOCITY_ZERO_THRESHOLD) or \
               (left_vx_norm < -VELOCITY_ZERO_THRESHOLD and prev_vx_for_freq_left > VELOCITY_ZERO_THRESHOLD):
                oscillation_count_left_x += 1
            
            if (right_vx_norm > VELOCITY_ZERO_THRESHOLD and prev_vx_for_freq_right < -VELOCITY_ZERO_THRESHOLD) or \
               (right_vx_norm < -VELOCITY_ZERO_THRESHOLD and prev_vx_for_freq_right > VELOCITY_ZERO_THRESHOLD):
                oscillation_count_right_x += 1
            
            # Update previous velocities for frequency calculation for next frame
            prev_vx_for_freq_left = left_vx_norm
            prev_vx_for_freq_right = right_vx_norm

            # Frequency for Excel output (using X-axis oscillation)
            # Divide by 2 because each full oscillation cycle has two zero-crossings (e.g., pos to neg, neg to pos)
            current_left_freq_hz = (oscillation_count_left_x / 2) / Time_with_face if Time_with_face > 0 else 0.0
            current_right_freq_hz = (oscillation_count_right_x / 2) / Time_with_face if Time_with_face > 0 else 0.0

            # Amplitude for Excel (using magnitude of overall displacement)
            current_left_amp_mm = abs(current_left_move_mm)
            current_right_amp_mm = abs(current_right_move_mm)

            # Variance Calculation (for ML)
            dx_history_left.append(left_dx_norm)
            dy_history_left.append(left_dy_norm)
            dz_history_left.append(left_dz_norm)
            dx_history_right.append(right_dx_norm)
            dy_history_right.append(right_dy_norm)
            dz_history_right.append(right_dz_norm)

            if len(dx_history_left) > 1:
                left_dx_var = np.var(list(dx_history_left))
                left_dy_var = np.var(list(dy_history_left))
                left_dz_var = np.var(list(dz_history_left))
                right_dx_var = np.var(list(dx_history_right))
                right_dy_var = np.var(list(dy_history_right))
                right_dz_var = np.var(list(dz_history_right))
            else:
                left_dx_var, left_dy_var, left_dz_var = 0.0, 0.0, 0.0
                right_dx_var, right_dy_var, right_dz_var = 0.0, 0.0, 0.0

            # Removed: Blink Detection logic and EAR calculations

            # Populate raw_features_map with current computed values
            raw_features_map = {
                'Left_Move_Magnitude_mm': current_left_move_mm,
                'Right_Move_Magnitude_mm': current_right_move_mm,
                'Left_Vel_Magnitude_mm_s': current_left_vel_mm_s,
                'Right_Vel_Magnitude_mm_s': current_right_vel_mm_s,
                'Left_Freq_Hz': current_left_freq_hz,
                'Right_Freq_Hz': current_right_freq_hz,
                'Left_DX_Norm': left_dx_norm,
                'Left_DY_Norm': left_dy_norm,
                'Left_DZ_Norm': left_dz_norm,
                'Left_VX_Norm': left_vx_norm,
                'Left_VY_Norm': left_vy_norm,
                'Left_VZ_Norm': left_vz_norm,
                'Right_DX_Norm': right_dx_norm,
                'Right_DY_Norm': right_dy_norm,
                'Right_DZ_Norm': right_dz_norm,
                'Right_VX_Norm': right_vx_norm,
                'Right_VY_Norm': right_vy_norm,
                'Right_VZ_Norm': right_vz_norm,
                'Left_DX_Variance': left_dx_var,
                'Left_DY_Variance': left_dy_var,
                'Left_DZ_Variance': left_dz_var,
                'Right_DX_Variance': right_dx_var,
                'Right_DY_Variance': right_dy_var,
                'Right_DZ_Variance': right_dz_var,
                'Left_X_Acceleration_Norm': left_ax_norm,
                'Left_Y_Acceleration_Norm': left_ay_norm,
                'Left_Z_Acceleration_Norm': left_az_norm,
                'Right_X_Acceleration_Norm': right_ax_norm,
                'Right_Y_Acceleration_Norm': right_ay_norm,
                'Right_Z_Acceleration_Norm': right_az_norm
            }

            current_features_for_ml = [raw_features_map[feature_name] for feature_name in ML_FEATURES_LIST]
            
            # Make a prediction if the model and scaler are loaded
            if model and scaler: # Check if model and scaler are loaded first
                if len(current_features_for_ml) == len(ML_FEATURES_LIST): # Then check feature count
                    try:
                        scaled_features = scaler.transform([current_features_for_ml])
                        pred = model.predict(scaled_features)
                        proba = model.predict_proba(scaled_features)[0]
                        prediction_text = f"State: {pred[0]} ({np.max(proba):.2f})"
                    except Exception as e:
                        prediction_text = f"Prediction error: {str(e)}"
                else:
                    prediction_text = "Feature mismatch for prediction" # Should not happen if ML_FEATURES_LIST is consistent
            else:
                prediction_text = "Model not ready for prediction"
            
            # Update previous values for the next frame (ONLY if face was detected)
            prev_x_left, prev_y_left, prev_z_left = left_eye_center_3d
            prev_x_right, prev_y_right, prev_z_right = right_eye_center_3d
            
            prev_vx_left, prev_vy_left, prev_vz_left = left_vx_norm, left_vy_norm, left_vz_norm
            prev_vx_right, prev_vy_right, prev_vz_right = right_vx_norm, right_vy_norm, right_vz_norm
    else: # No face detected
        # If no face is detected, current_features_for_ml would be empty.
        # Ensure it's populated with default zeros for data consistency in Excel output.
        current_features_for_ml = [0.0] * len(ML_FEATURES_LIST)
        # raw_features_map is already initialized to 0.0 at the start of the loop for this scenario


    # Collect data for each frame for Excel
    # Assemble the row based on the EXCEL_OUTPUT_HEADER structure
    row_data_values = []
    temp_dict_for_excel_output = {
        "Index": frame_index,
        "Time (s)": Time,
        "TimeWithFace": Time_with_face,
        "LeftMove(mm)": current_left_move_mm,
        "RightMove(mm)": current_right_move_mm,
        "LeftVel(mm/s)": current_left_vel_mm_s,
        "RightVel(mm/s)": current_right_vel_mm_s,
        "LeftFreq(Hz)": current_left_freq_hz,
        "RightFreq(Hz)": current_right_freq_hz,
        "LeftAmp(mm)": current_left_amp_mm,
        "RightAmp(mm)": current_right_amp_mm,
        "Prediction": prediction_text,
        "Label": 0 # Default label for new data collected
    }
    # Populate temp_dict_for_excel_output with all ML features from raw_features_map (which holds all computed or zeroed values)
    for feature_name in ALL_ML_FEATURES:
        temp_dict_for_excel_output[feature_name] = raw_features_map.get(feature_name, 0.0)


    # Build the row based on EXCEL_OUTPUT_HEADER (which respects FEATURES_TO_EXCLUDE)
    for header_col in EXCEL_OUTPUT_HEADER:
        row_data_values.append(temp_dict_for_excel_output.get(header_col, 0.0)) # Use .get() with default for safety

    data_for_excel.append(row_data_values)
    
    # Real-Time Graph Visualization
    # Update plot history (using normalized velocities for consistency)
    vx_left_plot_history.append(left_vx_norm)
    vy_left_plot_history.append(left_vy_norm) 
    vx_right_plot_history.append(right_vx_norm)
    vy_right_plot_history.append(right_vy_norm)

    # Redraw plot
    plot_img.fill(0) # Clear previous frame

    # Draw grid lines for readability
    cv2.line(plot_img, (0, PLOT_HEIGHT // 2), (PLOT_WIDTH, PLOT_HEIGHT // 2), (50, 50, 50), 1) # X-axis line (zero velocity)
    cv2.line(plot_img, (PLOT_WIDTH // 2, 0), (PLOT_WIDTH // 2, PLOT_HEIGHT), (50, 50, 50), 1) # Y-axis line

    # Draw velocity plots
    plot_img_labels = [] # To manage legend position
    def draw_plot_line(history, color, label):
        if len(history) > 1:
            for i in range(1, len(history)):
                p1_x = int((i - 1) * PLOT_WIDTH / plot_history_len)
                p1_y = normalize_for_plot(history[i-1], MIN_VELOCITY_PLOT, MAX_VELOCITY_PLOT, PLOT_HEIGHT)
                p2_x = int(i * PLOT_WIDTH / plot_history_len)
                p2_y = normalize_for_plot(history[i], MIN_VELOCITY_PLOT, MAX_VELOCITY_PLOT, PLOT_HEIGHT)
                cv2.line(plot_img, (p1_x, p1_y), (p2_x, p2_y), color, 1)
        cv2.putText(plot_img, label, (10, 20 + 20 * len(plot_img_labels)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        plot_img_labels.append(None) # Just to manage vertical spacing for legend

    draw_plot_line(vx_left_plot_history, (0, 255, 0), "L_VX (G)")    # Green for Left VX
    draw_plot_line(vy_left_plot_history, (255, 0, 0), "L_VY (B)")    # Blue for Left VY
    draw_plot_line(vx_right_plot_history, (0, 255, 255), "R_VX (C)") # Cyan for Right VX
    draw_plot_line(vy_right_plot_history, (255, 0, 255), "R_VY (M)") # Magenta for Right VY

    cv2.putText(plot_img, f"Min: {MIN_VELOCITY_PLOT:.2f}", (10, PLOT_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(plot_img, f"Max: {MAX_VELOCITY_PLOT:.2f}", (10, PLOT_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Eye Velocity Graph", plot_img)
    
    # Display the prediction on the main frame
    cv2.putText(frame, prediction_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Tremor Detection", frame)
    
    Time += frame_time
    frame_index += 1
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Save results to Excel
output_dir = r"F:\GP\ML\LiveData" # Ensure this directory exists or is writable
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "LiveData.xlsx") # This is fixed to LiveData.xlsx for saving new data

try:
    # Always load from LiveData.xlsx for saving new sessions
    if not os.path.exists(output_file):
        print(f"Creating new Excel file for saving: {output_file}")
        wb = Workbook()
        ws_init = wb.active
        ws_init.title = "Initial Sheet" # Default sheet name
        wb.save(output_file)
        wb = load_workbook(output_file) # Re-load to ensure it's loaded correctly
        if "Initial Sheet" in wb.sheetnames:
            wb.remove(wb["Initial Sheet"])
    else:
        try:
            wb = load_workbook(output_file)
        except Exception as load_error:
            print(f"FATAL ERROR: Could not load existing Excel file for saving: {output_file} - {load_error}. This might indicate corruption or wrong format.")
            print(f"Please ENSURE '{output_file}' is closed, not corrupted, and saved as a proper Excel .xlsx file.")
            print("To fix, you may need to delete or rename the old 'LiveData.xlsx' and let the script create a new one.")
            raise # Re-raise the exception to stop execution and alert the user


    sheet_name = f"Session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ws = wb.create_sheet(title=sheet_name)
    
    for row in data_for_excel:
        ws.append(row)
        
    wb.save(output_file)
    print(f"Data saved successfully to {output_file} under sheet: {sheet_name}")
    
except Exception as e:
    print(f"Error saving data: {str(e)}")

cap.release()
cv2.destroyAllWindows()
print("Application terminated.")