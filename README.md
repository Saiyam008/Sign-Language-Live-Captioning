# 🤟 Sign Language Live Captioning & Audio Synthesis

A real-time assistive computer vision and deep learning system that detects and translates sign language gestures into **live visual captions** and **spoken audio** using **MediaPipe Holistic**, **TensorFlow/Keras sequential neural networks**, **PyTTSX3 text-to-speech**, and a **Flask / Browser Extension** interface.

---

## 🌟 Overview & Architecture

```mermaid
graph LR
    Camera[Webcam / Video Stream] --> MediaPipe[MediaPipe Holistic Keypoint Tracking]
    MediaPipe --> Keypoints[Extracted Pose & Hand Landmarks]
    Keypoints --> Model[TensorFlow/Keras Sequential Model (action.h5)]
    Model --> Prob[Softmax Gesture Classification]
    Prob --> Text[Live Captioning Overlay]
    Prob --> Audio[PyTTSX3 Text-to-Speech Engine]
```

---

## 🚀 Key Features

- **🤲 Multi-Keypoint Extraction**: Real-time extraction of 33 pose landmarks, 21 left-hand landmarks, and 21 right-hand landmarks using Google MediaPipe Holistic.
- **🧠 Sequential Deep Learning**: Deep neural network architecture (`action.h5`) trained on continuous landmark sequences to recognize dynamic multi-sign gestures (up to 20 sign vocabulary).
- **🔊 Real-Time Audio Synthesis**: Integrated with `pyttsx3` text-to-speech engine to speak out translated sentences instantly in background threads.
- **🌐 Dual Deployment (Flask API & Browser Extension)**:
  - Flask backend server exposing `/predict` endpoint for frames.
  - Interactive web UI and Chrome Extension (`manifest.json`, `popup.html`, `popup.js`) for overlaying captions on live video calls or web pages.
- **📓 Prototyping Notebooks**: Includes comprehensive step-by-step training notebooks with data collection, sequence padding, and model evaluation.

---

## 📁 Repository Structure

```
├── app.py                                            # Main Flask server with MediaPipe + TTS pipeline
├── 20 Signs Sign Language Recognition with Audio.ipynb # Extended training notebook (20 signs)
├── Sign Language.ipynb                              # Core gesture dataset creation and model training
├── action.h5                                         # Trained Keras sequential model weights
├── 04.h5                                             # Model checkpoint
├── mymodel.py / my_prediction_module.py             # Inference helper modules
├── index.html                                        # Web dashboard interface
├── manifest.json / popup.html / popup.js             # Browser extension integration
├── Sign Language Presentation.pdf                    # Architecture presentation slides
└── requirements.txt (or dependencies)
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.8 - 3.10
- Webcam / Video Capture device

### 2. Clone the repository
```bash
git clone https://github.com/Saiyam008/Sign-Language-Live-Captioning.git
cd Sign-Language-Live-Captioning
```

### 3. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install tensorflow opencv-python mediapipe pyttsx3 flask pillow numpy
```

---

## 🏃 Running the Application

### 1. Start the Flask Server
```bash
python app.py
```
The server will initialize the MediaPipe pipeline and load `action.h5`.

### 2. Run the Interactive Jupyter Notebook
```bash
jupyter notebook "20 Signs Sign Language Recognition with Audio.ipynb"
```

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
