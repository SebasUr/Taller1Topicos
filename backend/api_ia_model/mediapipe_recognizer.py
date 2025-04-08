import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from .sign_recognizer import SignRecognizer
from typing import Tuple, Union

class MediaPipeSignRecognizer(SignRecognizer):
    def __init__(self, actions):
        super().__init__(actions)
        self.model = self._load_model()
        self.sequence = []
        self.threshold = 0.8
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def _load_model(self):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation="relu", input_shape=(30,1662)))
        model.add(LSTM(128, return_sequences=True, activation="relu"))
        model.add(LSTM(64, return_sequences=False, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(len(self.actions), activation="softmax"))
        model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])

        weights_path = os.path.join(os.path.dirname(__file__), 'tryfewer.h5')
        model.load_weights(weights_path)
        return model


    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, Union[int, None]]:
        image, results = self._mediapipe_detection(frame)
        self._draw_landmarks(image, results)

        keypoints = self._extract_keypoints(results)
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-30:]

        prediction = None
        if len(self.sequence) == 30:
            res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
            pred_index = np.argmax(res)
            if res[pred_index] > self.threshold:
                prediction = pred_index
            image = self._prob_viz(res, image)

        return image, prediction

    def _mediapipe_detection(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def _draw_landmarks(self, image, results):
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.face_landmarks,
                self.mp_holistic.FACEMESH_TESSELATION
            )
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS
            )
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS
            )
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS
            )

    def _extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])

    def _prob_viz(self, res, input_frame):
        output_frame = input_frame.copy()
        for idx, prob in enumerate(res):
            cv2.rectangle(output_frame, (0, 60 + idx*40), (int(prob*100), 90 + idx*40), (0,255,0), -1)
            cv2.putText(output_frame, f'{self.actions[idx]}: {prob:.2f}', (5, 85 + idx*40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        return output_frame