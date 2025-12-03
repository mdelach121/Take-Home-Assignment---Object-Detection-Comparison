# Take-Home-Assignment---Object-Detection-Comparison
Molly Delach, MSBA 503, December 3, 2025

This project implements Part A of the assignment using a set of photos taken in Seville, including museum displays, a bull, food, architecture, and street scenes. Two deep learning object detection models are compared on the same images, and an additional non–deep learning feature (dominant color) is extracted from each image.

**Images**

The images used in this project are stored locally in a folder on my computer. In the code, this folder is referenced by the `IMAGE_DIR` variable in `object_detection_project.py`.

Examples of image content:
- Bullfighting costume and museum displays  
- A taxidermy bull  
- A painting of a bullfight and a bullfighter statue  
- Food (fish dish), coffee cup, and Seville landmarks (La Giralda, Real Alcázar interior)

**Part A (i): Object Detection**

I used two deep learning object detection models from the `torchvision` library:

1. `fasterrcnn_resnet50_fpn`
2. `retinanet_resnet50_fpn`

Both models are pre-trained on the COCO dataset and are used to:
- Detect objects in each image
- Measure:
  - Inference time per image (`time_sec`)
  - Number of objects detected above a confidence threshold (`num_objects`)
  - Average confidence score of the detections (`avg_confidence`)

**Part A (ii): Color Detection**
For Part A (ii), I extracted the dominant RGB color from each image using K-Means clustering (with `k=3` clusters) on the pixel values.
