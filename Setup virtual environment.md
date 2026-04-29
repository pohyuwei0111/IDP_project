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
Create a file named `main.py` in your project folder and add the following code:

```python
import cv2
from ultralytics import YOLO

def main():
    # Load the pre-trained YOLOv8 Nano model
    model = YOLO('yolov8n.pt') 

    # Initialize the webcam (0 is the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam inference... Press 'q' to quit.")

    # Process the video stream frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Run inference on the current frame
        results = model(frame, stream=True)

        # Display the annotated results
        for r in results:
            annotated_frame = r.plot()
            cv2.imshow('YOLOv8 Real-Time Inference', annotated_frame)

        # Exit mechanism
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

## Step 5: Run the Model
Execute the script from your terminal:

```bash
python main.py
```
*(Note: It will automatically download the `yolov8n.pt` weights file on the very first run).*

## Step 6: Shutting Down
* Click on the video window and press **`q`** to close the camera feed.
* To exit the Python virtual environment in your terminal, type:

```bash
deactivate
```
