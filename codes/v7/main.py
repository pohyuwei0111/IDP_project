import cv2
import torch
import torch.serialization
from ultralytics import YOLO
import math
import os
import argparse
import numpy as np
from collections import deque
from datetime import datetime
import time  # For accurate real-time delta (dt) math

# --- PyTorch 2.6 Security Patch ---
torch.load = torch.serialization.load
if not hasattr(torch.load, "_is_patched"):
    _original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    
    patched_torch_load._is_patched = True
    torch.load = patched_torch_load
# -----------------------------------

if __name__ == "__main__":
    # --- 1. SET UP ARGUMENT PARSER ---
    parser = argparse.ArgumentParser(description="YOLO Tracking with Graph and Anti-Jitter")
    parser.add_argument('-c', '--camera', type=int, default=0, help="Camera ID to use (default: 0)")
    args = parser.parse_args()

    video_source = args.camera
    print(f"Opening camera ID {video_source} and starting tracking...")
    
    model = YOLO("car_trained.engine", task="detect") 
    # --- 2. SETUP VIDEO & SAVER ---
    # Use cv2.CAP_DSHOW on Windows to get deeper hardware control (remove if on Mac/Linux)
    cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {video_source}. Check your connection/ID.")
        exit()

    # Request high resolution and 60 FPS from the camera driver
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release() 

    if video_fps == 0 or math.isnan(video_fps):
        video_fps = 30.0
        
    print(f"Camera Negotiated Resolution: {video_w}x{video_h} at {video_fps} FPS")

    output_dir = "runs/detect/custom_track"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"webcam_track_{timestamp}.mp4")



    # --- 3. TRACKING SETUP ---
    results = model.track(
        source=video_source, 
        show=False, 
        save=False, 
        stream=True,
        persist=True,
        conf=0.8,
        imgsz=1280
    )

    window_width = 1280  
    window_height = 720 
    
    previous_positions = {}
    previous_velocities = {}
    previous_times = {} # Tracks exact time each car was last seen
    
    # --- ANTI-JITTER SETTINGS ---
    smoothed_velocities = {} 
    STATIONARY_THRESHOLD = 3.5  # Deadzone in pixels
    SMOOTHING_FACTOR = 0.15     # 15% new speed, 85% old speed
    
    # --- GRAPH SETUP ---
    GRAPH_WIDTH = 600
    GRAPH_HEIGHT = 400
    MAX_HISTORY = 180  # ~3 seconds of history assuming ~30 FPS
    speed_history = {} 
    
    DECEL_THRESHOLD = 40.0
    RECOVERY_SPEED = 30.0  
    shockwave_cars = set() 
    
    # --- NEW: Set writer to None initially ---
    video_writer = None
    
    # --- 4. MAIN LOOP ---
    for r in results:
        annotated_frame = r.plot()
        current_frame_safe_speed = float('inf')
        current_time = time.time() 
        
        # --- NEW: Initialize VideoWriter on the VERY FIRST frame ---
        if video_writer is None:
            # Read the exact height and width of the frame YOLO is outputting
            frame_height, frame_width = annotated_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(save_path, fourcc, video_fps, (frame_width, frame_height))
            print(f"🎬 VideoWriter started at exact resolution: {frame_width}x{frame_height}")
        # -----------------------------------------------------------
        
        active_ids = set() # Track who is on screen right now

        if r.boxes.id is not None: 
            boxes = r.boxes.xywh.cpu() 
            track_ids = r.boxes.id.int().cpu().tolist()
            active_ids = set(track_ids)

            for box, track_id in zip(boxes, track_ids):
                x_center, y_center, width, height = box
                cv2.circle(annotated_frame, (int(x_center), int(y_center)), 5, (0, 0, 255), -1)

                cm_per_pixel = 6.0 / width 
                
                if track_id not in speed_history:
                    speed_history[track_id] = deque(maxlen=MAX_HISTORY)
                
                if track_id in previous_positions and track_id in previous_times:
                    prev_x, prev_y = previous_positions[track_id]
                    prev_time = previous_times[track_id]
                    
                    # --- 4a. CALCULATE DELTA TIME (dt) ---
                    dt = current_time - prev_time
                    if dt <= 0.001:  # Prevent division by zero
                        dt = 0.001
                    
                    dx = x_center - prev_x
                    dy = y_center - prev_y
                    pixel_distance = math.sqrt(dx**2 + dy**2)
                    
                    # --- 4b. INCREASED DEADZONE ---
                    if pixel_distance < STATIONARY_THRESHOLD: 
                        raw_vx, raw_vy = 0.0, 0.0
                    else:
                        raw_vx = (dx * cm_per_pixel) / dt
                        raw_vy = (dy * cm_per_pixel) / dt
                        
                    # --- 4c. EMA SMOOTHING FILTER ---
                    if track_id in smoothed_velocities:
                        prev_smooth_vx, prev_smooth_vy = smoothed_velocities[track_id]
                        current_vx = (SMOOTHING_FACTOR * raw_vx) + ((1.0 - SMOOTHING_FACTOR) * prev_smooth_vx)
                        current_vy = (SMOOTHING_FACTOR * raw_vy) + ((1.0 - SMOOTHING_FACTOR) * prev_smooth_vy)
                    else:
                        current_vx, current_vy = raw_vx, raw_vy
                        
                    current_speed = math.sqrt(current_vx**2 + current_vy**2)
                    
                    # --- 4d. SNAP TO ZERO ---
                    if current_speed < 1.0:
                        current_vx, current_vy, current_speed = 0.0, 0.0, 0.0
                        
                    smoothed_velocities[track_id] = (current_vx, current_vy)
                    speed_history[track_id].append(current_speed)
                        
                    text_color = (0, 255, 255)

                    # --- 4e. VECTOR ACCELERATION MATH ---
                    if track_id in previous_velocities:
                        prev_vx, prev_vy = previous_velocities[track_id]
                        prev_speed = math.sqrt(prev_vx**2 + prev_vy**2)
                        
                        ax = (current_vx - prev_vx) / dt
                        ay = (current_vy - prev_vy) / dt
                        
                        if prev_speed > 0.1:
                            dir_x = prev_vx / prev_speed
                            dir_y = prev_vy / prev_speed
                            forward_acceleration = (ax * dir_x) + (ay * dir_y)
                            
                            if forward_acceleration < -DECEL_THRESHOLD:
                                text_color = (0, 0, 255) 
                                cv2.putText(annotated_frame, "RAPID DECEL!", (int(x_center) - 50, int(y_center) - 55), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
                                shockwave_cars.add(track_id)

                    # Recovery check
                    if track_id in shockwave_cars and current_speed > RECOVERY_SPEED:
                        shockwave_cars.remove(track_id)
                            
                    if track_id in shockwave_cars:
                        if current_speed < current_frame_safe_speed:
                            current_frame_safe_speed = current_speed
                            
                    previous_velocities[track_id] = (current_vx, current_vy)
                    
                    # Print speed on screen
                    speed_text = f"{current_speed:.1f} cm/s"
                    cv2.putText(annotated_frame, speed_text, (int(x_center) - 20, int(y_center) - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                    
                previous_positions[track_id] = (x_center, y_center)
                previous_times[track_id] = current_time 
        
        # --- 5. HOUSEKEEPING (CLEANUP OFF-SCREEN CARS) ---
        for tid in list(speed_history.keys()):
            if tid not in active_ids:
                del speed_history[tid]
                if tid in previous_velocities: del previous_velocities[tid]
                if tid in previous_positions: del previous_positions[tid]
                if tid in smoothed_velocities: del smoothed_velocities[tid]
                if tid in previous_times: del previous_times[tid] 
        
        # --- 6. DRAW UI OVERLAY ---
        cv2.rectangle(annotated_frame, (10, 10), (450, 70), (0, 0, 0), -1)
        
        if current_frame_safe_speed != float('inf'):
            pacing_text = f"SUGGESTED SPEED: {current_frame_safe_speed:.1f} cm/s"
            cv2.putText(annotated_frame, pacing_text, (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(annotated_frame, "TRAFFIC NORMAL", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) 
        
        # --- 7. DRAW REAL-TIME VELOCITY GRAPH ---
        graph_img = np.zeros((GRAPH_HEIGHT, GRAPH_WIDTH, 3), dtype=np.uint8)
        
        # Draw background grid
        cv2.line(graph_img, (50, 0), (50, GRAPH_HEIGHT), (255, 255, 255), 2) # Y Axis
        cv2.line(graph_img, (50, GRAPH_HEIGHT-30), (GRAPH_WIDTH, GRAPH_HEIGHT-30), (255, 255, 255), 2) # X Axis
        cv2.putText(graph_img, "Velocity (cm/s)", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(graph_img, "Time ->", (GRAPH_WIDTH - 80, GRAPH_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Dynamic Y-Axis Scaling
        max_speed_y = 100.0  
        for speeds in speed_history.values():
            if len(speeds) > 0 and max(speeds) > max_speed_y:
                max_speed_y = max(speeds) + 20

        # Plot lines
        for tid, speeds in speed_history.items():
            if len(speeds) < 2: 
                continue
            
            # Persistent random color per track_id
            np.random.seed(tid)
            color = tuple(int(x) for x in np.random.randint(50, 255, 3))
            
            pts = []
            for i, spd in enumerate(speeds):
                x = 50 + int((i / MAX_HISTORY) * (GRAPH_WIDTH - 60))
                y = (GRAPH_HEIGHT - 30) - int((spd / max_speed_y) * (GRAPH_HEIGHT - 50))
                pts.append((x, y))
                
            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(graph_img, [pts], False, color, 2)
            
            # Label
            end_x, end_y = pts[-1][0]
            cv2.putText(graph_img, f"ID:{tid}", (end_x + 5, end_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- 8. SAVE & SHOW DISPLAYS ---
        video_writer.write(annotated_frame)

        display_frame = cv2.resize(annotated_frame, (window_width, window_height))
        cv2.imshow("Custom Detection Window", display_frame)
        cv2.imshow("Real-Time Velocity Graph", graph_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Detection stopped early by user.")
            break

    # --- 9. SHUTDOWN ---
    video_writer.release() 
    cv2.destroyAllWindows()
    print("\nProcessing complete!")
    print(f"Saved to: {save_path}")