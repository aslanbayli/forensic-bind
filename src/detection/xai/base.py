from abc import ABC, abstractmethod
import torch
import numpy as np

class BaseExplainer(ABC):
    """
    Abstract base class for all explanation methods.
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    @abstractmethod
    def explain(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Generate an explanation for the given input.
        
        Args:
            input_tensor (torch.Tensor): Preprocessed input image (B, C, H, W).
            target_class (int, optional): The class index to explain. If None, uses the predicted class.
            
        Returns:
            np.ndarray: A heatmap or attribution map of shape (H, W) or (C, H, W).
        """
        pass

