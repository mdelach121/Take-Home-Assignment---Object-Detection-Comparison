# Take-Home-Assignment---Object-Detection-Comparison
Molly Delach, MSBA 503, December 3, 2025

This project implements multiple deep learning algorithms to analyze a set of images collected online.

Part A (i): Object Detection

I used **two YOLOv8 models** from the Ultralytics library:

- `yolov8n.pt` – a smaller, faster model
- `yolov8s.pt` – a slightly larger, more accurate model

For each image and each model, we record:
- Inference time (seconds)
- Number of objects detected
- Average detection confidence (probability)
- Class counts (e.g., `person:2, car:1`)

The results are saved in:

- `object_detection_comparison.csv`

These results are used to compare the performance of the two models.

Part A (ii): Additional Image Features

I extracted extra information from the images in two ways:

1. **Dominant Color (Non–deep learning)**  
   - I used K-Means clustering (scikit-learn) on the image pixels to compute the dominant RGB color per image.
   - Output is stored in `extra_image_features.csv`.

2. **Image Similarity with CLIP Embeddings (Deep learning)**  
   -I used the `clip-ViT-B-32` model from `sentence-transformers` to create 512-dimensional embeddings for each image.
   - Embeddings are stored in a local ChromaDB collection.
   - For each image, we find the most similar image in the dataset based on cosine distance between embeddings.

## Code Files

- `cv_project.ipynb` or `cv_project.py` – main project code with:
  - YOLOv8 object detection comparison (Part A (i))
  - Dominant color extraction (Part A (ii))
  - CLIP embeddings + ChromaDB similarity search (Part B)
