# Engineering Progress Report: Anti-Phantom Traffic System

**Date:** May 26, 2026

**Module:** Custom Object Tracking, Dynamic Harmonization & Hardware Pipeline

**Prepared By:** Poh Yu Wei

---

## 1. Executive Summary

This report details the transition of the Anti-Phantom Traffic testbed from using basic physical blocks to a fully custom-trained computer vision model detecting actual toy cars. The system has evolved from simple speed calculations into a dynamic, self-correcting traffic management algorithm. Recent efforts have heavily focused on overcoming physical hardware bottlenecks (camera frame rates, AI jitter) to establish a high-speed, real-time data pipeline using smartphone telemetry.

---

## 2. Phase 1: Custom Object Detection (The Toy Car Upgrade)

The system was upgraded to recognize specific physical testbed vehicles rather than generic shapes.

* **Dataset Generation:** Refactored the `vid2img.py` script into a flexible command-line interface (CLI) tool using `argparse`, allowing rapid extraction of 1-FPS video frames into categorized datasets.
* **Model Training:** Utilized Roboflow for automated bounding-box annotation and exported the dataset in YOLOv8 format.
* **Cloud Processing:** Successfully trained a custom weights file (`car_trained.pt`) using a Google Colab T4 GPU instance. The local OpenCV pipeline was updated to load this highly accurate, use-case-specific model.

---

## 3. Phase 2: Advanced Kinematics & Traffic Logic

The core mathematical model was completely overhauled to handle the entire lifecycle of a traffic jam—from the initial shockwave to full traffic recovery.

* **Dynamic Target Pacing:** The system now recalculates the `current_frame_safe_speed` continuously. Instead of locking onto a historical minimum speed, the UI dynamically updates the suggested pacing speed as the jammed cars begin to accelerate.
* **Shockwave Recovery Logic:** Engineered a memory-clearing threshold (`RECOVERY_SPEED = 30.0` cm/s). Once a car involved in a rapid deceleration event accelerates past this threshold, the system automatically removes it from the `shockwave_cars` memory and switches the UI back to a green "TRAFFIC NORMAL" status.

---

## 4. Phase 3: Hardware Constraints & Signal Filtering

Transitioning to live video tracking revealed several real-world physics and hardware limitations that were successfully debugged:

* **Bounding Box Jitter (The "Static Movement" Bug):** Identified an issue where AI micro-wiggles caused perfectly still cars to register at 0.1 to 0.2 cm/s. Implemented a spatial deadzone filter; if pixel displacement is `< 2.0` between frames, the system forces the calculated speed to an absolute `0.0` cm/s.
* **Hardware Frame Rate Caps:** Conducted hardware diagnostics on the local laptop camera and external HQ-USB-1080HD module. Confirmed a physical hardware cap of 30 FPS across both devices (even at compressed MJPG / 240p resolutions).

---

## 5. Phase 4: The High-Speed Virtual Camera Pipeline (Current Status)

To bypass the 30 FPS hardware limitation and prevent motion blur on fast-moving toy cars, the tracking pipeline was rerouted through a modern smartphone camera sensor.

* **ADB Telemetry:** Bypassed the missing native Android 13 USB-Webcam functionality by enabling Android Developer Options and activating **USB Debugging**.
* **Virtual Driver Integration:** Deployed Iriun/Camo as a software bridge. This routes the smartphone's superior camera sensor (capable of true 60 FPS) directly over the high-speed USB data cable.
* **OpenCV Handshake:** The Python script is now configured (`video_source = 1`) to ingest this virtual feed, granting the YOLOv8 model a high-framerate, low-latency video stream for perfectly accurate distance and acceleration math.

---

## 6. Immediate Next Steps

1. **Verify the 60 FPS Pipeline:** Run the newly developed `test_fps.py` diagnostic script to confirm the virtual camera is successfully transmitting a stable ~60 frames per second over the USB bridge.
2. **Live Track Calibration:** Run the main tracking script (`test_model.py`) using the live phone feed and adjust the `DECEL_THRESHOLD` and `RECOVERY_SPEED` variables based on the smoother 60 FPS data.
3. **IoT Dashboard Integration:** Begin formatting the `target_safe_speed` output to be published via MQTT to the Node-RED dashboard.
