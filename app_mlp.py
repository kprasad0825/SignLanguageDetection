from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import tensorflow as tf
from collections import deque, Counter
from autocorrect import Speller
import threading # We need threading for global variables

app = Flask(__name__)

# --- Global Variables ---
# We need to share these variables between threads
model = load_model('final_action_model.keras', compile=False)
spell = Speller(lang='en')
actions = np.array([
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
])
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ✅ --- Global variables to store the data ---
g_sentence = []
g_current_word = []
g_lock = threading.Lock() # A lock to prevent errors
# ---------------------------------------------

# --- Keypoint Extraction (126 points) ---
def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

# --- Main Generator (Video + Logic) ---
def generate_frames():
    # We use 'global' to modify the variables
    global g_sentence, g_current_word, g_lock
    
    sentence = []
    current_word = []
    predictions = deque(maxlen=20)
    threshold = 0.7
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1) 
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # --- Draw ALL points ---
            face_drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(80,110,10))
            mp_drawing.draw_landmarks(image_bgr, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, landmark_drawing_spec=face_drawing_spec)
            mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image_bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image_bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # --- Prediction & Autocorrect Logic ---
            keypoints = extract_keypoints(results)
            res = model.predict(np.expand_dims(keypoints, axis=0), verbose=0)[0]
            action_pred = actions[np.argmax(res)]
            
            if res[np.argmax(res)] > threshold:
                predictions.append(action_pred)
            else:
                predictions.append('nothing')

            if len(predictions) == 20:
                most_common_pred = Counter(predictions).most_common(1)[0][0]
                
                if most_common_pred == 'space':
                    if current_word:
                        word_str = "".join(current_word)
                        corrected_word = spell(word_str)
                        sentence.append(corrected_word.upper())
                        current_word = []
                
                elif most_common_pred == 'del':
                    if current_word:
                        current_word.pop()
                
                elif most_common_pred != 'nothing':
                    if not current_word or most_common_pred != current_word[-1]:
                        current_word.append(most_common_pred)

            if len(sentence) > 5:
                sentence = sentence[-5:]
            
            # ✅ --- Update the global variables safely ---
            with g_lock:
                g_sentence = sentence[:]
                g_current_word = current_word[:]
            # ---------------------------------------------
            
            # --- Draw Text ON the Video Frame (Optional) ---
            # You can comment these out if you only want text in the HTML boxes
            cv2.rectangle(image_bgr, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image_bgr, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(image_bgr, (0, 440), (640, 480), (80, 80, 80), -1)
            cv2.putText(image_bgr, "".join(current_word), (3, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # --- Yield the final frame ---
            flag, encodedImage = cv2.imencode(".jpg", image_bgr)
            if not flag:
                continue
            
            yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n'

    cap.release()

# --- Simplified Routes ---
@app.route('/')
def index():
    # This just loads the HTML page
    return render_template('index.html') 

@app.route('/video_feed')
def video_feed():
    # This single route runs the generator and streams the video
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ✅ --- NEW Route to get the data ---
@app.route('/get_data')
def get_data():
    # This route just sends the current data as JSON
    global g_sentence, g_current_word, g_lock
    with g_lock:
        data = {
            "sentence": g_sentence,
            "current_word": g_current_word
        }
    return jsonify(data)
# ----------------------------------

# --- Start the App ---
if __name__ == '__main__':
    app.run(debug=True, threaded=True)