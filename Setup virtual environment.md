# YOLOv8 Real-Time Webcam Inference Setup

This repository contains a lightweight Python script to run real-time object detection using the pre-trained YOLOv8 Nano model via a laptop webcam.

## Prerequisites
* Python 3.8+ installed
* Built-in or external USB webcam

---

## Step 1: Project Setup
Open your terminal and create a dedicated project directory.

```bash
mkdir YOLO-Webcam-Test
cd YOLO-Webcam-Test
```

## Step 2: Virtual Environment
Create and activate a virtual environment to keep dependencies isolated.

**Create the environment:**
```bash
python -m venv venv
```

**Activate the environment:**
* **Windows (Command Prompt):** `venv\Scripts\activate`
* **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
* **macOS / Linux:** `source venv/bin/activate`

## Step 3: Install Dependencies
Install the required YOLO and OpenCV libraries.

```bash
pip install ultralytics opencv-python
```

## Step 4: The Inference Script
Create a file named `test_original.py` in your project folder and add the following code:

```python
import cv2
from ultralytics import YOLO

def main():
    # 1. Load the pre-trained YOLOv8 Nano model
    # The script will automatically download 'yolov8n.pt' from the internet the first time this runs.
    model = YOLO('yolov8n.pt') 

    # 2. Initialize the laptop webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam inference... Press 'q' to quit.")

    # 3. Process the video stream
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # 4. Run inference
        results = model(frame, stream=True)

        # 5. Parse and display the results
        for r in results:
            # Draw the bounding boxes
            annotated_frame = r.plot()
            cv2.imshow('Testing Original YOLOv8', annotated_frame)

        # 6. Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 7. Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

## Step 5: Run the Model
Execute the script from your terminal:

```bash
python test_original.py
```
*(Note: It will automatically download the `yolov8n.pt` weights file on the very first run).*

## Step 6: Shutting Down
* Click on the video window and press **`q`** to close the camera feed.
* To exit the Python virtual environment in your terminal, type:

```bash
deactivate
```
