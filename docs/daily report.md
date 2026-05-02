# Daily Progress Report: Anti-Phantom Traffic System

**Date:** May 2, 2026  
**Module:** Computer Vision & Object Detection Testbed  
**Prepared By:** Poh Yu Wei  

---

## 1. Executive Summary
Over the past two days, significant progress was made on the visual detection component of the Anti-Phantom Traffic testbed. After initial challenges with foundation model hallucinations and environment constraints, the workflow was successfully pivoted to a custom-trained YOLOv8 Nano model. The model is now successfully deployed locally and accurately tracking physical vehicle blocks in real-time.

## 2. Yesterday's Progress: Environment Setup & Foundation Model Testing
The primary focus was establishing the training pipeline and testing zero-shot auto-labeling methods. 

* **Hardware & Environment Optimization:** Upgraded the Google Colab environment to utilize a T4 GPU, reducing expected training times from several hours to under 30 minutes. 
* **Dependency Debugging:** Engineered patches for several critical library failures:
    * Bypassed Hugging Face `get_head_mask` errors and fixed a hardcoded device type bug within the Grounding DINO source code.
    * Implemented a secure `torch.load()` patch to bypass PyTorch 2.6's strict `weights_only=True` security block, allowing the loading of custom `.pt` weights.
* **Auto-Labeling Evaluation:** Attempted to generate the dataset automatically using Grounding DINO and `supervision`. Data visualization revealed severe misclassifications, including overlapping multi-class boxes and massive bounding boxes capturing the entire black road surface due to prompt confusion. 

## 3. Today's Progress: Dataset Pivot, Training & Local Deployment
Following the data visualization review, the labeling strategy was completely overhauled to ensure high-fidelity inputs for the YOLOv8 model.

* **Dataset Generation & Roboflow Pivot:** Extracted 1 FPS frames from the `.mov` test video using a custom OpenCV script. Transitioned to Roboflow for data annotation, redefining the detection problem to focus on "rectangular blocks" rather than distinct colors, resulting in highly accurate bounding boxes.
* **Pipeline Automation:** Engineered a Python script to dynamically rewrite absolute paths inside the exported `data.yaml` file, effectively resolving YOLOv8 training crashes caused by missing validation folders in small Roboflow exports.
* **Model Training:** Successfully trained a custom, lightweight YOLOv8 Nano (`yolov8n.pt`) model on the verified dataset.
* **Local Edge Deployment:** Downloaded the optimal weights (`best.pt`) for local laptop inference.
* **Inference Optimization:** 
    * Replaced YOLO's default display mechanism with a custom OpenCV (`cv2`) loop to constrain the live inference viewing window to a manageable size, while continuing to save full-resolution annotated video in the background.
    * Removed `agnostic_nms` and utilized default Intersection over Union (IoU) and confidence thresholds, successfully eliminating ghost detections and stabilizing the tracking boxes.

## 4. Next Steps
With the vision module now providing accurate, reliable object detection for the physical testbed, the next phase of development will focus on the core system logic:
* Extract bounding box coordinates from the YOLOv8 detections.
* Implement tracking logic to calculate individual vehicle speeds and relative distances.
* Begin integrating the rapid-deceleration detection logic to trigger the Anti-Phantom Traffic alerts.
