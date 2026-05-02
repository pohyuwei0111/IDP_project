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
    video_path = "test_vid_1.mov" 

    print(f"Loading {video_path} and starting detection...")

    # 3. Turn OFF default 'show' and turn ON 'stream'
    # stream=True processes the video frame-by-frame (saves RAM)
    # save=True lets YOLO handle saving the full-resolution video in the background
    # conf=0.60: Ignores any predictions with less than 60% confidence
    # iou=0.30: If two boxes overlap by more than 30%, merge them and keep the most confident one
    results = model.predict(
        source=video_path, 
        show=False, 
        save=True, 
        stream=True 
    )

    # 4. Set your custom display window size (Width, Height)
    # 960x540 is exactly half of 1080p, which is a perfect medium size
    window_width = 960  
    window_height = 540 

    # 5. Process the video frame-by-frame
    for r in results:
        # Grab the raw frame with the bounding boxes drawn on it
        annotated_frame = r.plot()

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
    print("A saved copy of the full-resolution video is located in your 'runs/detect/predict' folder.")
