import cv2
import torch
import torch.serialization
from ultralytics import YOLO
import math
import os
from datetime import datetime

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
    model = YOLO("car_trained.pt")
    
    # --- CHANGED: 0 is the default ID for your laptop's built-in webcam ---
    # (If you plug in an external USB webcam, you might need to change this to 1 or 2)
    video_source = 0 

    print("Opening laptop webcam and starting tracking...")

    # --- 1. SETUP CUSTOM VIDEO SAVER ---
    cap = cv2.VideoCapture(video_source)
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release() 

    # --- NEW: WEBCAM FPS FALLBACK ---
    # Webcams sometimes fail to report their FPS to OpenCV. 
    # If it returns 0, we force it to assume 30 FPS so your speed math doesn't crash.
    if video_fps == 0 or math.isnan(video_fps):
        video_fps = 30.0
        print("⚠️ Warning: Webcam did not report FPS. Defaulting speed calculations to 30 FPS.")

    # Define the YOLO-style output directory
    output_dir = "runs/detect/custom_track"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Changed the filename to reflect it's a webcam recording
    unique_filename = f"webcam_track_{timestamp}.mp4"
    save_path = os.path.join(output_dir, unique_filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(save_path, fourcc, video_fps, (video_w, video_h))
    # -----------------------------------

    # 2. TRACKING (source is now set to 0)
    results = model.track(
        source=video_source, 
        show=False, 
        save=False, 
        stream=True,
        persist=True 
    )

    window_width = 960  
    window_height = 540 
    previous_positions = {}
    previous_speeds = {}
    
    DECEL_THRESHOLD = 50.0
    
    shockwave_cars = set() 
    target_safe_speed = float('inf')
    
    for r in results:
        annotated_frame = r.plot()

        if r.boxes.id is not None: 
            boxes = r.boxes.xywh.cpu() 
            track_ids = r.boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x_center, y_center, width, height = box
                
                cv2.circle(annotated_frame, (int(x_center), int(y_center)), 5, (0, 0, 255), -1)

                # --- SPEED MATH & DECELERATION ---
                cm_per_pixel = 6.0 / width 

                if track_id in previous_positions:
                    prev_x, prev_y = previous_positions[track_id]
                    pixel_distance = math.sqrt((x_center - prev_x)**2 + (y_center - prev_y)**2)
                    real_distance_cm = pixel_distance * cm_per_pixel
                    
                    current_speed = real_distance_cm * video_fps
                    text_color = (0, 255, 255) 

                    if track_id in previous_speeds:
                        prev_speed = previous_speeds[track_id]
                        acceleration = (current_speed - prev_speed) * video_fps
                        
                        if acceleration < -DECEL_THRESHOLD:
                            text_color = (0, 0, 255) 
                            
                            cv2.putText(annotated_frame, "RAPID DECEL!", (int(x_center) - 50, int(y_center) - 55), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
                                        
                            shockwave_cars.add(track_id)
                            print(f"⚠️ ALARM: Car {track_id} is braking hard! ({acceleration:.1f} cm/s^2)")
                            
                    if track_id in shockwave_cars:
                        if current_speed < target_safe_speed:
                            target_safe_speed = current_speed
                            
                    previous_speeds[track_id] = current_speed
                    
                    speed_text = f"{current_speed:.1f} cm/s"
                    cv2.putText(annotated_frame, speed_text, (int(x_center) - 20, int(y_center) - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                    
                previous_positions[track_id] = (x_center, y_center)
        
        # --- DRAW SUGGESTED SPEED ON SCREEN ---
        if target_safe_speed != float('inf'):
            cv2.rectangle(annotated_frame, (10, 10), (450, 70), (0, 0, 0), -1)
            pacing_text = f"SUGGESTED SPEED: {target_safe_speed:.1f} cm/s"
            cv2.putText(annotated_frame, pacing_text, (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # --- 3. SAVE THE CUSTOM FRAME ---
        video_writer.write(annotated_frame)

        # Show the live feed
        display_frame = cv2.resize(annotated_frame, (window_width, window_height))
        cv2.imshow("Custom Detection Window", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Detection stopped early by user.")
            break

    # --- 4. CLEAN UP ---
    video_writer.release() 
    cv2.destroyAllWindows()

    print("\nProcessing complete!")
    print(f"A saved copy of the tracked webcam video is located at: {save_path}")
