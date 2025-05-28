from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional, Any, Dict


class ActivationFunction(ABC):
    """Abstract base class for activation functions"""
    
    @staticmethod
    @abstractmethod
    def forward(x: np.ndarray) -> np.ndarray:
        pass