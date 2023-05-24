import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from scipy import stats
import pyttsx3
import threading
from IPython.display import display, Javascript
from google.colab.output import eval_js

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    )
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    )

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def perform_prediction():
    actions = np.array(['namaste', 'i love you', 'thanks', 'noaction', 'salute', 'clothes', 'all the best', 'headache'])
    model = load_model('action_05/04.h5')

    def start_capture():
        js = Javascript('''
            async function startCapture() {
                const stream = await navigator.mediaDevices.getUserMedia({video: true});
                const videoElement = document.createElement('video');
                videoElement.srcObject = stream;
                videoElement.autoplay = true;
                document.body.appendChild(videoElement);
                return;
            }
            startCapture();
        ''')
        display(js)

    start_capture()

    def capture_frames():
        video = cv2.VideoCapture(0)
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while True:
                try:
                    ret, frame = video.read()
                    if not ret:
                        break

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    keypoints = extract_keypoints(results)

                    sequence.append(keypoints)
                    sequence = sequence[-30:]
                    if len(sequence) == 30:
                        res = model.predict(np.expand_dims(sequence, axis=0))[0]
                        predictions.append(np.argmax(res))
                        if np.unique(predictions[-10:])[0] != 0 and np.mean(predictions[-10:]) == np.unique(predictions[-10:])[0]:
                            cv2.putText(image, actions[int(stats.mode(predictions[-10:])[0])], (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
                            cv2.putText(image, "Confidence: {:.2f}".format(np.max(res)), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            if np.max(res) > threshold:
                                if actions[int(stats.mode(predictions[-10:])[0])] not in sentence:
                                    sentence.append(actions[int(stats.mode(predictions[-10:])[0])])
                                else:
                                    sentence = []
                        if len(sentence) > 2:
                            engine.say(' '.join(sentence))
                            engine.runAndWait()
                            sentence = []

                    cv2.imshow('Action Recognition', image)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                except Exception as e:
                    print(f"Error predicting: {type(e).__name__}: {e}")

        video.release()
        cv2.destroyAllWindows()

    threading.Thread(target=capture_frames).start()

perform_prediction()
