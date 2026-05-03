# Daily Progress Report: Anti-Phantom Traffic System

**Date:** May 3, 2026  
**Module:** Kinematics, Shockwave Detection & Speed Harmonization  
**Prepared By:** Poh Yu Wei  

---

## 1. Executive Summary
Today's development successfully transitioned the computer vision module from static object detection to dynamic, real-time kinematics tracking. The system can now calculate the real-world speed of individual physical blocks, detect rapid deceleration events (shockwaves), and calculate the optimal pacing speed required to mitigate phantom traffic jams. Additionally, the data logging and video saving pipelines were robustly upgraded.

## 2. Technical Progress & Implementations

### A. Dynamic Object Tracking & Coordinate Extraction
* **Tracking Upgrade:** Transitioned the YOLOv8 inference from `predict` to `track` mode (`persist=True`), allowing the system to assign persistent unique IDs to each vehicle across continuous frames.
* **Spatial Mapping:** Successfully extracted raw `(x, y)` center coordinates and bounding box dimensions for every detected object, enabling frame-to-frame distance tracking.

### B. Real-World Kinematics ($cm/s$)
* **Pixel-to-Physical Calibration:** Engineered a dynamic conversion ratio leveraging the known physical length of the test vehicles (6 cm). By dividing the physical length by the bounding box pixel width (`6.0 / width`), the system automatically adapts to perspective shifts.
* **Velocity Calculation:** Implemented distance formula math $\sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$ to calculate pixel displacement. Multiplied by the calibration ratio and the video's exact FPS to output highly accurate real-world speed in $cm/s$.

### C. Shockwave Detection (Rapid Deceleration)
* **Acceleration Math:** Added frame-to-frame speed memory to calculate acceleration ($a = \frac{\Delta v}{\Delta t}$).
* **Event Triggers:** Established a tunable `DECEL_THRESHOLD` (currently 50.0 $cm/s^2$). When a vehicle's negative acceleration exceeds this threshold, the system flags the vehicle and triggers a visual "RAPID DECEL!" UI alert on the monitoring feed.

### D. Speed Harmonization Algorithm
* **Bottleneck Analysis:** Programmed a memory state to continuously track the velocity of vehicles that trigger a deceleration event.
* **Target Speed Generation:** The algorithm dynamically isolates the absolute minimum speed reached during a shockwave (`target_safe_speed`) and broadcasts it to the UI as the "Suggested Pacing Speed" to enforce upstream traffic smoothing. 

### E. Pipeline Data Operations
* **Custom OpenCV Integration:** Bypassed YOLO's default video saver to preserve custom UI overlays (speed text, warnings).
* **Dynamic File Management:** Implemented the `datetime` module to inject unique timestamps into the output filenames (e.g., `steady_with_speed_20260503_130800.mp4`). This prevents overwriting data and creates an organized archive of test runs within the `runs/detect/custom_track` directory.

## 3. Next Steps
With the visual tracking and core harmonization math successfully running locally, the system is ready for the hardware/IoT integration phase:
* **IoT Data Transmission:** Begin publishing the generated `target_safe_speed` and shockwave alerts via MQTT to the Node-RED dashboard.
* **Hardware Feedback Loop:** Link the Node-RED dashboard logic to the in-car devices to physically alert the trailing vehicles to begin gentle deceleration.
* **Multi-Vehicle Stress Test:** Run a physical test with 3+ cars to visually verify that the upstream pacing speed successfully absorbs the shockwave before the trailing cars are forced to stop.
