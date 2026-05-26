from ultralytics import YOLO

# 1. Load your standard PyTorch model
model = YOLO("car_trained.pt")

print("Starting TensorRT conversion... This might take 5-10 minutes.")

# 2. Export to TensorRT engine format with FP16 speed boost
model.export(
    format="engine", 
    half=True,    # Uses FP16 for massive speed on RTX 4050
    imgsz=640,    # Locks in the resolution
    dynamic=False # Static size is slightly faster
)

print("Conversion complete!")
