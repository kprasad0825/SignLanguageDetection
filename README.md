# Sign Language Detection (MLP-based)

This project detects American Sign Language (ASL) letters in real time using a webcam and converts them into text.  
It uses **MediaPipe** for hand landmark detection and a **Multi-Layer Perceptron (MLP)** deep learning model to predict letters.

---

## 🧠 Project Overview

This is an upgraded version of an earlier LSTM-based model.  
The new version uses a simpler and faster **MLP** model trained on single-frame hand landmarks instead of 30-frame sequences.

### Key Improvements:
- Focuses **only on hand landmarks** (126 points).
- Uses **data augmentation** with small random noise for better generalization.
- **Prediction smoothing** via deque to avoid flickering.
- Smart **sentence formation logic** with `del`, `space`, and `autocorrect`.

---

## 🧰 Technologies Used

- Python  
- MediaPipe  
- OpenCV  
- TensorFlow / Keras  
- NumPy  
- Flask  
- Autocorrect  

---

## ⚙️ How to Run

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
