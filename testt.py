import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO
# from picamera2 import Picamera2 # Replaced by CameraHandler
from camera_handler import CameraHandler # Import the new handler
from sound import make_sound
from alert_owner import send_alert_to_owner # Import the alert function
import os # For creating output directory
import time # For human detection timer


# Load YOLO model
try:
    model = YOLO('yolov8n.pt')
    print("YOLOv8n model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    sys.exit(1)


def preprocess_frame(frame):
    # Reduce blue channel for Noir camera
    b, g, r = cv2.split(frame)
    b = (b * 0.7).astype(np.uint8)
    frame = cv2.merge((b, g, r))
    # Denoise
    frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
    # Histogram equalization for contrast (on Y channel)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.equalizeHist(y)
    ycrcb = cv2.merge((y, cr, cb))
    frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return frame

def identify_entities(img_array):
    # Run YOLOv8 inference with confidence threshold 0.35
    results = model(img_array, conf=0.35)

    detected_humans = 0
    detected_animals = 0
    animal_classes = {
        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse',
        18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
        22: 'zebra', 23: 'giraffe'
    }

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            if cls_id == 0:
                detected_humans += 1
            elif cls_id in animal_classes:
                detected_animals += 1
                print(f"Detected animal: {animal_classes[cls_id]}")

    print(f"Humans detected: {detected_humans} | Animals detected: {detected_animals}")

    # Annotated frame with bounding boxes and labels
    annotated_frame = results[0].plot()
    return annotated_frame, detected_humans, detected_animals


if __name__ == "__main__":
    print("Farm Alert - Animal and Human Identification System")
    print("---------------------------------------------------")

    # --- Configuration ---
    CAMERA_TYPE = 'ip'  # 'pi' or 'ip'. For 'pi', ensure Picamera2 is installed and hardware connected.
    IP_CAMERA_URL = "http://192.168.0.3:8080/video" # Replace with your IP camera's stream URL
    PI_CAMERA_CONFIG = {'size': (1280, 1280), 'format': 'RGB888'} # Example config for Pi camera
    OUTPUT_DIR = "detection_outputs"

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # Initialize camera using CameraHandler
    try:
        if CAMERA_TYPE == 'pi':
            # For Pi camera, ensure picamera2 library is installed and camera is connected
            # You might need to adjust pi_camera_config based on your Pi Camera version and needs
            camera = CameraHandler(camera_type='pi', pi_camera_config=PI_CAMERA_CONFIG)
        elif CAMERA_TYPE == 'ip':
            camera = CameraHandler(camera_type='ip', ip_camera_url=IP_CAMERA_URL)
        else:
            raise ValueError("Invalid CAMERA_TYPE configured. Choose 'ip' or 'pi'.")
        print(f"{CAMERA_TYPE.upper()} camera initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize {CAMERA_TYPE} camera: {e}")
        if CAMERA_TYPE == 'pi':
            print("Ensure Picamera2 library is installed, camera is connected, and drivers are configured.")
        sys.exit(1)




    human_detected_start_time = None
    human_alert_threshold = 10 # seconds
    human_alert_triggered_for_current_detection = False

    print(f"Starting detection loop. Press 'q' to quit.")

    try:
        while True:
            start_time = time.time()

            # Capture frame
            try:
                frame = camera.get_frame()
                if frame is None:
                    print("Failed to get frame. Retrying...")
                    time.sleep(1) # Wait a bit before retrying
                    continue
            except Exception as e:
                print(f"Error capturing frame: {e}")
                time.sleep(1)
                continue

            original_frame_for_alert = frame.copy() # Keep a copy of the original frame for alerts
            annotated_frame, detected_humans, detected_animals = identify_entities(frame)

            if detected_animals > 0:
                make_sound() # Play sound if animal is detected
                print("Animal detected, sound played.")

            if detected_humans > 0:
                if human_detected_start_time is None:
                    human_detected_start_time = time.time()
                    human_alert_triggered_for_current_detection = False # Reset trigger for new detection period
                    print(f"Human detected. Starting timer ({human_alert_threshold}s threshold).")
                elif not human_alert_triggered_for_current_detection and (time.time() - human_detected_start_time > human_alert_threshold):
                    print(f"Human detected for over {human_alert_threshold} seconds. Triggering alert.")
                    send_alert_to_owner(original_frame_for_alert) # Send original frame
                    human_alert_triggered_for_current_detection = True # Ensure alert is sent only once per continuous detection
            else: # No humans detected
                if human_detected_start_time is not None:
                    print("Human no longer detected. Resetting timer.")
                human_detected_start_time = None # Reset timer if no human is detected
                human_alert_triggered_for_current_detection = False

            # Display results (optional, can be kept commented out)
            # cv2.imshow("YOLO Detections", annotated_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # Save annotated frame to disk in the specified output directory
            
            # timestamp_val = int(time.time())
            # output_filename = f"detection_{timestamp_val}.jpg"
            # output_path = os.path.join(OUTPUT_DIR, output_filename)
            # cv2.imwrite(output_path, annotated_frame)
            # print(f"Saved detection result to {output_path}")

            # Minimal delay to prevent high CPU usage if processing is very fast
            # and to allow other processes to run.
            # This can be adjusted or removed if maximum possible throughput is critical
            # and CPU usage is not a concern.
            time.sleep(0.01) # Sleep for 10ms

    except KeyboardInterrupt:
        print("User interrupted, stopping...")

    finally:
        if 'camera' in locals() and camera: # Ensure camera object exists
            camera.release()
        # cv2.destroyAllWindows() # Already called in camera.release()
        print("Camera resources released. Exiting.")
