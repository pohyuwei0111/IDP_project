# google colab code blocks

!pip install -q \
  autodistill-yolov8

import os
HOME = os.getcwd()
print(HOME)

# The -q means "quiet" so it doesn't flood your screen with text
# The -d tells it to extract everything into a folder named "dataset"
!unzip -q /content/roboflow.zip -d /content/dataset

import yaml
import os

yaml_file = '/content/dataset/data.yaml'

# Load the existing data.yaml
with open(yaml_file, 'r') as f:
    data = yaml.safe_load(f)

# Force the training path to be an absolute path
data['train'] = '/content/dataset/train/images'

# Check if a 'valid' folder actually exists
if os.path.exists('/content/dataset/valid/images'):
    data['val'] = '/content/dataset/valid/images'
else:
    print("⚠️ No 'valid' folder found. Telling YOLO to use 'train' data for validation instead.")
    data['val'] = '/content/dataset/train/images'

# Check if a 'test' folder exists
if 'test' in data:
    if os.path.exists('/content/dataset/test/images'):
        data['test'] = '/content/dataset/test/images'
    else:
        del data['test'] # Delete the test requirement so it doesn't crash

# Save the fixed instructions back to the file
with open(yaml_file, 'w') as f:
    yaml.dump(data, f, sort_keys=False)

print("✅ Successfully updated data.yaml paths! You are ready to train.")

import torch
import torch.serialization
from ultralytics import YOLO

# --- 1. Re-apply the PyTorch 2.6 Security Patch ---
torch.load = torch.serialization.load
if not hasattr(torch.load, "_is_patched"):
    _original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)

    patched_torch_load._is_patched = True
    torch.load = patched_torch_load
    print("PyTorch patch applied safely!")
# --------------------------------------------------

# 2. Load a fresh YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# 3. Train it on your new manual dataset!
model.train(data="/content/dataset/data.yaml", epochs=50)
