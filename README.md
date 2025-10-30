# Neural Style Transfer using TensorFlow Hub

This project implements **Neural Style Transfer (NST)** using **TensorFlow Hub's pre-trained arbitrary image stylization model**.  
It blends the *content* of one image with the *style* of another, producing stunning artistic results — for example, applying the painting style of *Van Gogh's Starry Night* to a photograph.
author shankar sutar 
---

##  Overview

Neural Style Transfer uses a **deep neural network** to combine two images:
- **Content Image** → structure and layout (e.g., photo)
- **Style Image** → artistic style (e.g., painting)
- **Output** → image that looks like the content image painted in the style of the style image

This notebook leverages TensorFlow Hub’s **`magenta/arbitrary-image-stylization-v1-256/2`** model — a fast, high-quality pre-trained model ready to use on any image pair.

---

##  Features

 Works seamlessly in **Google Colab**  
 Accepts both **local uploads** and **image URLs**  
 Uses **TensorFlow Hub’s** pre-trained model (no training required)  
 Saves and downloads the stylized image automatically  
 Fully commented, modular, and beginner-friendly  

---

## Requirements

Install dependencies before running (already included in Colab):

```bash
pip install tensorflow tensorflow-hub pillow matplotlib
