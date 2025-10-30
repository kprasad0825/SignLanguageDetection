import cv2
import numpy as np
import os
import mediapipe as mp

# Initialize MediaPipe components
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- Configuration ---
# Path for exported data
DATA_PATH = os.path.join('MP_Data') 

# Actions that we want to detect
# ✅ FIXED: Added 'del', 'nothing', 'space' back in
actions = np.array([
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
])

# Number of sequences (videos) for each action
no_sequences = 100

# Each video is 30 frames in length
sequence_length = 30
# ---------------------

# ✅ --- CRITICAL FIX: Extract ONLY Hand Keypoints ---
def extract_keypoints(results):
    """Extracts hand keypoints (126 total) and returns as a flat array."""
    # 63 points for left hand (21 landmarks * 3 coords)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    # 63 points for right hand (21 landmarks * 3 coords)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    # Final array has 126 points
    return np.concatenate([lh, rh])
# ----------------------------------------------------

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    # ✅ --- NEW: Main loop for interactive collection ---
    while cap.isOpened():
        
        # --- This section is the "Live Feed" ---
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1) # Flip for selfie view
        
        # Make detections
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        
        # Draw landmarks (Hands only for a cleaner view)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        # Show instructions on the screen
        cv2.putText(image, "Live Feed. Press a letter (A-Z) to record.", (15, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "Press 'q' to quit.", (15, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('OpenCV Feed', image)
        
        # --- End of Live Feed section ---

        # Wait for a key press
        key = cv2.waitKey(10) & 0xFF

        # 'q' to quit
        if key == ord('q'):
            break
        
        # Check if the key is a letter
        action = ''
        if ord('a') <= key <= ord('z'):
            action = chr(key).upper()
        elif ord('A') <= key <= ord('Z'):
            action = chr(key)
        
        # If a valid action letter was pressed
        if action in actions:
            print(f"\n--- SELECTED ACTION: {action} ---")
            
            # 1. Get sequence number from user
            sequence_input = input(f"Enter sequence number (0-{no_sequences-1}): ")
            
            try:
                sequence = int(sequence_input)
                if not (0 <= sequence < no_sequences):
                    raise ValueError
            except ValueError:
                print(f"❌ Invalid input. Please enter a number between 0 and {no_sequences-1}.")
                continue # Go back to the live feed
            
            print(f"✅ Preparing to record {action} - Sequence {sequence}. Hold your sign...")

            # ✅ --- CRITICAL FIX: Create folders ---
            # Create the folder path (e.g., MP_Data/A/0)
            seq_path = os.path.join(DATA_PATH, action, str(sequence))
            os.makedirs(seq_path, exist_ok=True)
            # ------------------------------------

            # 2. Start collection loop (30 frames)
            for frame_num in range(sequence_length):
                # Read feed
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)

                # Make detections
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                # Draw landmarks
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # --- Collection Logic ---
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000) # Wait for 2 seconds
                else: 
                    cv2.putText(image, f'Collecting... Frame {frame_num} for {action} Video {sequence}', (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(50) # Small delay between frames

                # Export keypoints (using the FIXED function)
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(seq_path, str(frame_num))
                np.save(npy_path, keypoints)

            print(f"✅ Collection complete for {action} - Sequence {sequence}.")

# Clean up
cap.release()
cv2.destroyAllWindows()