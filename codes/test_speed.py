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
    video_path = "steady.mp4" 

    print(f"Loading {video_path} and starting tracking...")

    # --- 1. SETUP CUSTOM VIDEO SAVER ---
    cap = cv2.VideoCapture(video_path)
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release() 

    # Define the YOLO-style output directory
    output_dir = "runs/detect/custom_track"
    
    # Tell Python to create the folder if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)

    # Get current date and time (YearMonthDay_HourMinuteSecond)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Inject the timestamp into the file name
    unique_filename = f"steady_with_speed_{timestamp}.mp4"
    
    # Combine folder and new unique filename
    save_path = os.path.join(output_dir, unique_filename)

    # Create the OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(save_path, fourcc, video_fps, (video_w, video_h))
    # -----------------------------------
    # Create the OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(save_path, fourcc, video_fps, (video_w, video_h))
    # -----------------------------------

    # 2. TRACKING (Note: save=False because we are saving it ourselves now)
    results = model.track(
        source=video_path, 
        show=False, 
        save=False, 
        stream=True,
        persist=True 
    )

    window_width = 960  
    window_height = 540 
    previous_positions = {}

    for r in results:
        annotated_frame = r.plot()

        if r.boxes.id is not None: 
            boxes = r.boxes.xywh.cpu() 
            track_ids = r.boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x_center, y_center, width, height = box
                
                cv2.circle(annotated_frame, (int(x_center), int(y_center)), 5, (0, 0, 255), -1)

                # --- SPEED MATH ---
                cm_per_pixel = 6.0 / width 

                if track_id in previous_positions:
                    prev_x, prev_y = previous_positions[track_id]
                    pixel_distance = math.sqrt((x_center - prev_x)**2 + (y_center - prev_y)**2)
                    real_distance_cm = pixel_distance * cm_per_pixel
                    
                    # Calculate speed using the exact video FPS
                    speed_cm_s = real_distance_cm * video_fps

                    speed_text = f"{speed_cm_s:.1f} cm/s"
                    cv2.putText(annotated_frame, speed_text, (int(x_center) - 20, int(y_center) - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                previous_positions[track_id] = (x_center, y_center)

        # --- 3. SAVE THE CUSTOM FRAME ---
        # Write the modified frame containing your custom text to the video file
        video_writer.write(annotated_frame)
        # --------------------------------

        display_frame = cv2.resize(annotated_frame, (window_width, window_height))
        cv2.imshow("Custom Detection Window", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Detection stopped early by user.")
            break

    # --- 4. CLEAN UP ---
    video_writer.release() # Very important! This actually saves the final video file.
    cv2.destroyAllWindows()

    print("\nProcessing complete!")
    print(f"A saved copy of the tracked video with speeds is located at: {save_path}")
