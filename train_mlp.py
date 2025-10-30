import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- Configuration ---
ACTIONS_COUNT = 29  # A-Z, del, nothing, space
INPUT_SHAPE = 126   # 126 hand keypoints
EPOCHS = 200
BATCH_SIZE = 32
# ---------------------

# Load data from disk
print("Loading data...")
X = np.load('X.npy')
y = np.load('y.npy')

print(f"Data loaded: X={X.shape}, y={y.shape}")

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y # stratify ensures even class split
)

# ------------------------------
# ✅ A MUCH SIMPLER MODEL
# ------------------------------
def create_model():
    model = Sequential([
        # Input layer
        Dense(64, activation='relu', input_shape=(INPUT_SHAPE,)),
        Dropout(0.2), # Dropout helps prevent overfitting
        
        # Hidden layers
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        
        # Output layer
        Dense(ACTIONS_COUNT, activation='softmax') # Softmax for multi-class classification
    ])
    
    model.compile(optimizer='Adam', 
                  loss='categorical_crossentropy', 
                  metrics=['categorical_accuracy'])
    return model

# ------------------------------
# CREATE AND TRAIN
# ------------------------------
model = create_model()
model.summary() # Print model architecture

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)

print("🚀 Training started...")

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

model.save('final_action_model.keras')
print("💾 Training complete. Model saved as final_action_model.keras")