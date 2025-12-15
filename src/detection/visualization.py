import matplotlib.pyplot as plt
import numpy as np
import cv2

def apply_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5, colormap=cv2.COLORMAP_JET) -> np.ndarray:
    """
    Overlay a heatmap onto an image.
    
    Args:
        image (np.ndarray): Original image (H, W, 3) in [0, 255].
        heatmap (np.ndarray): Heatmap (H, W) in [0, 1].
        alpha (float): Opacity of the heatmap.
        colormap: OpenCV colormap.
        
    Returns:
        np.ndarray: The superimposed image.
    """
    # Resize heatmap to match image dimensions if needed
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
    # Convert heatmap to RGB using colormap
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay

def plot_explanation(image, heatmap, title="Explanation"):
    """
    Simple matplotlib plotter.
    """
    overlay = apply_heatmap(image, heatmap)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title("Heatmap")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

