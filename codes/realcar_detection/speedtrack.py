import os
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# --- CALIBRATION CONSTANTS (Adjust these for your video) ---
# Estimate how many pixels equal 1 meter in your video. 
# (e.g., If a standard lane is 3.5 meters wide and measures 70 pixels wide on screen, PPM = 20)
PIXELS_PER_METER = 15  

# Minimum number of frames a vehicle must be tracked to generate a graph
MIN_TRACK_FRAMES = 30  
# -----------------------------------------------------------

# 1. Initialize folders and model
os.makedirs("velocity_graphs", exist_ok=True)
model = YOLO("yolo26x.pt")

# 2. Open video file
video_path = "test_video2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

# Get video details
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Set display window size to 640x480 as requested
window_name = "Vehicle Speed Tracker"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 720, 640)

# Data structures to keep track of vehicle movements
# history: {track_id: [(timestamp_in_sec, speed_kmh)]}
history = defaultdict(list)
last_positions = {}  # {track_id: (x, y)}
last_speeds = {}     # {track_id: speed_kmh}

frame_count = 0

print("Processing video and tracking vehicles...")
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    current_time_sec = frame_count / fps

    # Run multi-object tracking (persist=True keeps IDs across frames)
    # Using 'device=0' for your RTX 4050 GPU
    results = model.track(
        source=frame, 
        persist=True, 
        device=0, 
        classes=[2, 3, 5, 7], 
        verbose=False
    )

    # Check if any tracks were detected
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x1, y1, x2, y2 = box
            
            # Using bottom-center of the box for ground position tracking
            cx = (x1 + x2) / 2
            cy = y2 

            # Calculate speed if we have a previous position for this ID
            if track_id in last_positions:
                prev_x, prev_y = last_positions[track_id]
                
                # Distance in pixels
                pixel_distance = np.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)
                
                # Convert to real-world meters and calculate speed
                meters = pixel_distance / PIXELS_PER_METER
                speed_mps = meters * fps  # meters per second
                speed_kmh = speed_mps * 3.6  # kilometers per hour

                # Apply simple exponential smoothing to handle bounding box jitter
                prev_speed = last_speeds.get(track_id, speed_kmh)
                smoothed_speed = (0.7 * prev_speed) + (0.3 * speed_kmh)
            else:
                smoothed_speed = 0.0

            # Update tracking states
            last_positions[track_id] = (cx, cy)
            last_speeds[track_id] = smoothed_speed
            history[track_id].append((current_time_sec, smoothed_speed))

            # Visuals: Draw box, track ID, and Speed on the frame
            label_name = model.names[class_id]
            label_text = f"ID {track_id} | {smoothed_speed:.1f} km/h"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                frame, 
                label_text, 
                (int(x1), int(y1) - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )

    # Show live tracking output (limited to 640x480 display window)
    cv2.imshow(window_name, frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Video processing complete.")

# --- STEP 3: EXPORT VELOCITY GRAPHS ---
print("Generating and exporting velocity graphs...")
exported_count = 0

for track_id, data in history.items():
    # Only generate graphs for cars tracked long enough to yield meaningful data
    if len(data) >= MIN_TRACK_FRAMES:
        times, speeds = zip(*data)

        plt.figure(figsize=(8, 4))
        plt.plot(times, speeds, color="blue", linewidth=2, label=f"Car {track_id}")
        
        plt.title(f"Velocity Profile - Vehicle ID {track_id}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Velocity (km/h)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        
        # Save graph to the output directory
        graph_path = os.path.join("velocity_graphs", f"vehicle_{track_id}_speed.png")
        plt.savefig(graph_path, bbox_inches="tight")
        plt.close()  # Free up system memory
        exported_count += 1

print(f"Successfully generated {exported_count} speed graphs in the 'velocity_graphs/' folder.")
