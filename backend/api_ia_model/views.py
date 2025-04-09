from django.http import StreamingHttpResponse, JsonResponse
from django.core.cache import cache
import cv2
import numpy as np

from .mediapipe_recognizer import MediaPipeSignRecognizer

# Acciones soportadas por el modelo
actions = np.array(['hola', 'mi_nombre_es', 'como_estas','chao','buenas_noches', 'INSOR', 'por_favor', 'parado'])

# Colores para visualización (usados internamente en el recognizer si se desea)
colors = [
    (245, 117, 16), (117, 245, 16), (16, 117, 245),
    (16, 245, 117), (117, 16, 245), (245, 16, 117),
    (245, 16, 117), (16, 245, 117)
]

# Singleton para el recognizer
class RecognizerSingleton:
    _instance = None

    @staticmethod
    def get_instance():
        if RecognizerSingleton._instance is None:
            RecognizerSingleton._instance = MediaPipeSignRecognizer(actions)
        return RecognizerSingleton._instance

def gen_frames():
    cap = cv2.VideoCapture(0)
    recognizer = RecognizerSingleton.get_instance()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image, prediction = recognizer.detect(frame)
        if prediction is not None:
            cache.set('current_prediction', prediction)

        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def get_prediction(request):
    prediction = cache.get('current_prediction', None)
    if prediction is not None:
        return JsonResponse({'prediction': int(prediction)})
    else:
        return JsonResponse({'prediction': 'No prediction available'})