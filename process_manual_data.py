import numpy as np
import os
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical

# --- Configuration ---
DATA_PATH = os.path.join('MP_Data_MLP') # ✅ Read from the folder you just made
actions = np.array([
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
])

# How many images you collected for each action
no_sequences = 100

# How many "jittered" copies to make of each image
AUGMENTATIONS = 20
# ---------------------

X_data, y_data = [], []

print("Processing all collected .npy files...")

# Loop over each action (A, B, C...)
for action_index, action in enumerate(actions):
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        print(f"⚠ No data found for {action}. Skipping.")
        continue
    
    print(f"\n▶ Processing {action}...")
    
    # Loop over each image number (0, 1, 2...)
    for sequence in tqdm(range(no_sequences), desc=f'Processing {action}'):
        npy_path = os.path.join(action_path, f"{sequence}.npy")
        
        if not os.path.exists(npy_path):
            # If a file is missing (e.g., you only collected 50), stop for this action
            print(f"  (Stopped at image {sequence}, file not found)")
            break 
        
        # Load the keypoints (your 126-point array)
        keypoints = np.load(npy_path) 
        
        # Add the "clean" original keypoints
        X_data.append(keypoints)
        y_data.append(action_index) # Add the label (e.g., 0 for 'A')
        
        # ✅ --- Add augmented versions ---
        for _ in range(AUGMENTATIONS):
            noise = np.random.normal(0, 0.002, keypoints.shape) # Add jitter
            augmented_keypoints = keypoints + noise
            X_data.append(augmented_keypoints)
            y_data.append(action_index)

# Convert to numpy arrays
X = np.array(X_data, dtype=np.float32)
y = to_categorical(np.array(y_data, dtype=np.int32)) 

print(f"\nData shapes: X={X.shape}, y={y.shape}")

# Save to disk
np.save('X.npy', X)
np.save('y.npy', y)

print("\n✅ All data processed, augmented, and saved to X.npy and y.npy!")