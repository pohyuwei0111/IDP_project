import cv2
import os

video_path = "IMG_9603.mov" # Your video file
output_folder = "manual_dataset"

os.makedirs(output_folder, exist_ok=True)
vidcap = cv2.VideoCapture(video_path)
fps = round(vidcap.get(cv2.CAP_PROP_FPS))

success, image = vidcap.read()
count = 0
frame_id = 0

print("Extracting frames...")
while success:
    # Save 1 frame per second to avoid drawing boxes on identical images
    if count % fps == 0:
        cv2.imwrite(os.path.join(output_folder, f"frame_{frame_id}.jpg"), image)
        frame_id += 1
    success, image = vidcap.read()
    count += 1

print(f"Done! Extracted {frame_id} images to the '{output_folder}' folder.")
