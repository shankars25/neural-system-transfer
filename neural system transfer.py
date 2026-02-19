# -*- coding: utf-8 -*-

import io
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import requests


# -----------------------------
# Helper Functions
# -----------------------------
def load_image(path, max_dim=512):
    """Load image from disk"""
    img = Image.open(path).convert('RGB')
    img.thumbnail((max_dim, max_dim))
    img = np.array(img).astype(np.float32) / 255.0
    return img[None, ...]


def load_image_from_url(url, max_dim=512):
    """Load image from a URL"""
    data = requests.get(url).content
    img = Image.open(io.BytesIO(data)).convert('RGB')
    img.thumbnail((max_dim, max_dim))
    img = np.array(img).astype(np.float32) / 255.0
    return img[None, ...]


def save_image(img_tensor, filename):
    """Save stylized output to disk"""
    img = img_tensor[0]
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(filename)
    print(f"Image saved: {filename}")


def show_images(content, style, stylized):
    plt.figure(figsize=(15, 5))

    titles = ["Content Image", "Style Image", "Stylized Result"]
    images = [content[0], style[0], stylized[0]]

    for i, (title, img) in enumerate(zip(titles, images)):
        plt.subplot(1, 3, i + 1)
        plt.imshow(np.clip(img, 0, 1))
        plt.title(title)
        plt.axis("off")

    plt.show()


# -----------------------------
# Load Pretrained Model
# -----------------------------
MODEL_URL = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
stylize_model = hub.load(MODEL_URL)
print("Model loaded successfully.")


# -----------------------------
# Choose Images
# -----------------------------

# OPTION A — Local images
content_path = "content.jpg"   # <---- replace with your image
style_path = "style.jpg"       # <---- replace with your image

content_image = load_image(content_path)
style_image = load_image(style_path)

# OPTION B — URLs (uncomment if needed)
# content_image = load_image_from_url("https://your-content-url.jpg")
# style_image = load_image_from_url("https://your-style-url.jpg")


# -----------------------------
# Perform Style Transfer
# -----------------------------
stylized_image = stylize_model(
    tf.constant(content_image),
    tf.constant(style_image)
)[0]


# -----------------------------
# Save and Show
# -----------------------------
output_path = "stylized_output.jpg"
save_image(stylized_image, output_path)
show_images(content_image, style_image, stylized_image)
