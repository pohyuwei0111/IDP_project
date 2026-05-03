import cv2
import torch
import torch.serialization
from ultralytics import YOLO

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
    # 1. Load the custom model
    model = YOLO("car_trained.pt")

    # 2. Set the path to your clipped video
    video_path = "steady.mp4" 

    print(f"Loading {video_path} and starting tracking...")

    # 3. UPGRADE TO TRACKING
    # Changed 'predict' to 'track' and added 'persist=True' to remember cars across frames
    results = model.track(
        source=video_path, 
        show=False, 
        save=True, 
        stream=True,
        persist=True 
    )

    window_width = 960  
    window_height = 540 

    # 4. Process the video frame-by-frame
    for r in results:
        annotated_frame = r.plot()

        # --- 5. EXTRACT COORDINATES HERE ---
        # Check if the tracker actually found any IDs in this frame
        if r.boxes.id is not None: 
            # Extract the Center X, Center Y, Width, and Height of every box
            boxes = r.boxes.xywh.cpu() 
            
            # Extract the unique tracking IDs
            track_ids = r.boxes.id.int().cpu().tolist()

            # Loop through every tracked car in the current frame
            for box, track_id in zip(boxes, track_ids):
                x_center, y_center, width, height = box
                
                # Print the data to your terminal so you can see it working
                print(f"Car ID: {track_id} | X: {int(x_center)}, Y: {int(y_center)}")

                # Draw a red dot directly in the center of the car
                cv2.circle(annotated_frame, (int(x_center), int(y_center)), 5, (0, 0, 255), -1)
        # -----------------------------------

        # Resize the frame *only* for the display window 
        display_frame = cv2.resize(annotated_frame, (window_width, window_height))

        # Show the resized frame on your laptop screen
        cv2.imshow("Custom Detection Window", display_frame)

        # Wait 1ms for the next frame, and check if 'q' was pressed to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Detection stopped early by user.")
            break

    # Clean up the display window when the video finishes
    cv2.destroyAllWindows()

    print("\nProcessing complete!")
    print("A saved copy of the tracked video is located in your 'runs/detect/track' folder.")
