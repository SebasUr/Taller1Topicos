from abc import ABC, abstractmethod
from typing import Union, Tuple
import numpy as np
import cv2

class SignRecognizer(ABC):
    def __init__(self, actions):
        self.actions = actions

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, Union[int, None]]:
        """Procesa un frame y devuelve una tupla (imagen procesada, predicción)"""
        pass
