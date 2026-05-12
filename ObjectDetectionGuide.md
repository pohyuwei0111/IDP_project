# End-to-End Guide: Custom Object Detection with YOLOv8

This guide walks you through the complete pipeline for training a custom YOLOv8 object detection model, from raw video capture to downloading your trained weights.

---

### Step 1: Capture Video Footage

Start by recording a high-quality video of your toy cars.

* **Pro-tip:** Ensure you have good, even lighting and capture the cars from various angles and distances that mimic the environment where your final model will be used.

### **Step 2: Extract Frames using `vid2img.py**`
<img width="591" height="103" alt="image" src="https://github.com/user-attachments/assets/57a4e06b-0826-4ebf-b671-95d65b928d28" />

Object detection models train on still images, not raw video files.

* Run your `vid2img.py` script to parse the video file and clip it into individual image frames.
* <img width="1116" height="108" alt="image" src="https://github.com/user-attachments/assets/bb82aac8-80a0-4902-8102-72b21b65fdda" />

* Save these extracted frames into a dedicated folder(find manual_dataset) on your local machine.

### **Step 3: Upload the Dataset to Roboflow**
<img width="1890" height="337" alt="image" src="https://github.com/user-attachments/assets/f6c2925f-c260-4884-b510-950f11072ebe" />

Create an account or log in to [Roboflow](https://www.google.com/search?q=https://roboflow.com/) and create a new Object Detection project.
<img width="1892" height="877" alt="image" src="https://github.com/user-attachments/assets/c76c70fe-c0a5-46e4-a64f-cdbe8f48f776" />

* Drag and drop the folder(whole folder:manual_dataset, not image by image) containing your extracted images into the Roboflow web interface to upload them.

### **Step 4: Configure Auto-Labeling**
<img width="1909" height="762" alt="image" src="https://github.com/user-attachments/assets/25dbe8bd-0c07-445a-9a70-85d43674f198" />

Instead of drawing bounding boxes manually, use Roboflow's automated tools.

* Define your classes by naming the object you want to detect (e.g., `toy_car`).
* Provide a clear text description of the object. Roboflow will use foundation models to search your images and automatically draw bounding boxes around objects matching your description.

### **Step 5: Quality Assurance (QA) the Annotations**

Auto-labeling is incredibly fast but rarely perfect.

* Manually click through some samples of your annotated images to verify accuracy.
* Adjust any bounding boxes that are too loose, too tight, or misidentifying the wrong objects. Delete boxes drawn on background noise.

### **Step 6: Export the Dataset**

Once your dataset is fully labeled and generated, it is time to export it for training.
<img width="1620" height="465" alt="image" src="https://github.com/user-attachments/assets/42655eb4-0677-48ba-80aa-d0aacd900e20" />

* Click **Export Dataset** in Roboflow.
* Select **YOLOv8** as your export format.
* Choose the **Download zip to computer** option and save the `.zip` file to your local machine.
<img width="941" height="713" alt="image" src="https://github.com/user-attachments/assets/2c715b11-9dd9-4dc5-bb6f-01723239abd1" />

### **Step 7: Prepare Google Colab**
Save this notebook into colab: https://colab.research.google.com/drive/1qeTSTEVrDjSX4C-BzPSfxAcpklhejHeU?usp=sharing

Open your prepared Google Colab notebook (ensure you are connected to a GPU runtime via `Runtime > Change runtime type > T4 GPU`).

* Use the file explorer menu on the left side of Colab to upload your `.zip` file.
* Your notebook should already contain the standard code blocks required to unzip the dataset and initiate the YOLOv8 training sequence.

### **Step 8: Train the Model**

Execute the code blocks sequentially in your Colab notebook.

* Run the cell to unzip the dataset.
* Run the cell to start the YOLOv8 training command.
* *Note: Depending on the size of your dataset and the number of epochs, this process will take some time. Monitor the output logs to watch the model learn.*
<img width="1704" height="596" alt="image" src="https://github.com/user-attachments/assets/1185a40c-643d-4a6a-b171-977967f77745" />

### **Step 9: Download Your Trained Weights**

Once the training completes, YOLOv8 automatically saves your optimized model.

* In the Colab file explorer, navigate to the output directory. *Note: YOLOv8 typically saves this inside a `weights` subfolder.*
* Locate your model at: `runs/detect/train/weights/best.pt`
* Click the three dots next to `best.pt` and download it to your local machine. This file is your fully trained custom model, ready for deployment!
