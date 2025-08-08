# backend/style_transfer.py
import io
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub

# Load TF-Hub model once at import time (slow at first load)
print("Loading TF-Hub model (this may take a few seconds)...")
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
print("TF-Hub model loaded.")

def pil_to_tensor(img: Image.Image, max_dim: int = 512):
    """Convert PIL image to a float32 tensor in [0,1] with shape [1, H, W, 3]."""
    img = img.convert('RGB')
    # Resize to keep max dimension reasonably small for speed
    width, height = img.size
    scale = 1.0
    if max(width, height) > max_dim:
        scale = max_dim / float(max(width, height))
        new_w = int(width * scale)
        new_h = int(height * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    return tf.expand_dims(arr, axis=0)  # [1,H,W,3]

def tensor_to_pil(tensor: tf.Tensor):
    """Convert float32 tensor [1,H,W,3] or [H,W,3] in [0,1] -> PIL Image (uint8)."""
    if isinstance(tensor, tf.Variable) or tf.is_tensor(tensor):
        tensor = tensor.numpy()
    if tensor.ndim == 4:
        tensor = tensor[0]
    tensor = np.clip(tensor, 0.0, 1.0)
    arr = (tensor * 255).astype(np.uint8)
    return Image.fromarray(arr)

def stylize_bytes(content_bytes: bytes, style_bytes: bytes, content_max_dim: int = 512, style_max_dim: int = 256):
    """
    Accepts content & style image bytes, returns a PIL Image of the stylized result.
    Uses TF-Hub magenta arbitrary-image-stylization model.
    """
    try:
        content_img = Image.open(io.BytesIO(content_bytes)).convert('RGB')
    except Exception as e:
        raise ValueError(f"Failed to read content image: {e}")
    try:
        style_img = Image.open(io.BytesIO(style_bytes)).convert('RGB')
    except Exception as e:
        raise ValueError(f"Failed to read style image: {e}")

    # Convert to tensors
    content_tensor = pil_to_tensor(content_img, max_dim=content_max_dim)
    style_tensor = pil_to_tensor(style_img, max_dim=style_max_dim)

    # Run the hub model - returns a list/tuple, take first element
    # Model expects float32 [1,H,W,3] in [0,1]
    outputs = hub_model(tf.constant(content_tensor), tf.constant(style_tensor))
    stylized_tensor = outputs[0]

    # Convert to PIL image
    result_img = tensor_to_pil(stylized_tensor)
    return result_img
