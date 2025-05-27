import cv2
import numpy as np
import onnxruntime as ort
import os
import time
import threading
from sound import make_sound

# Import your CameraHandler class (adjust path if needed)
# Make sure camera_handler.py is in the same directory or accessible via PYTHONPATH
try:
    from camera_handler import CameraHandler
except ImportError:
    print("ERROR: Could not import CameraHandler. Make sure camera_handler.py is accessible.")
    # A dummy class to allow the script to run further for debugging other parts,
    # but you'll need a real CameraHandler.
    class CameraHandler:
        def __init__(self, camera_type=None, ip_camera_url=None, camera_index=0):
            print("WARNING: Using dummy CameraHandler. No real frames will be captured.")
            self.camera_type = camera_type
            self.frame = np.zeros((480, 640, 3), dtype=np.uint8) # Dummy frame
            if camera_type == 'ip':
                 print(f"Dummy IP Camera URL: {ip_camera_url}")
            elif camera_type == 'pi':
                 print(f"Dummy Pi Camera")
            else:
                 print(f"Dummy USB Camera Index: {camera_index}")


        def get_frame(self):
            # In a real scenario, this would return None if a frame isn't available.
            # For testing, we return a dummy frame.
            # time.sleep(0.1) # Simulate frame capture delay
            return self.frame.copy() # Return a copy

        def release(self):
            print("Dummy CameraHandler released.")

# === Config ===
MODEL_PATH = "yolov8n.onnx"  # Make sure this path is correct
ALERT_SOUND = "alert.mp3"    # Make sure this file exists and mpg123 is installed
INPUT_SIZE = (640, 640)      # Must match the input size your ONNX model expects
CONF_THRESH = 0.50           # Initial confidence threshold (can be adjusted)
NMS_THRESH = 0.45            # Non-Maximum Suppression threshold

# === Class labels ===
COCO_CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
                "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

ANIMAL_CLASSES = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}

# Define TARGET_CLASSES - This is crucial!
TARGET_CLASSES = ANIMAL_CLASSES.union({"person"})
print(f"Targeting classes: {TARGET_CLASSES}")

# === Load ONNX model ===
session = None
input_name = None
output_names = None
try:
    print(f"Loading ONNX model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider']) # Explicitly use CPU
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    print(f"Model loaded. Input name: {input_name}, Output names: {output_names}")
    print(f"Input details: {session.get_inputs()[0]}")
    for i, output_node in enumerate(session.get_outputs()):
        print(f"Output {i} details: {output_node}")
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    exit()

# === Globals ===
latest_frame = None
lock = threading.Lock()
running = True

# === Frame capture thread function using your CameraHandler ===
def capture_frames(camera_handler):
    global latest_frame, running
    frame_count = 0
    while running:
        frame = camera_handler.get_frame()
        if frame is not None:
            if frame_count % 100 == 0: # Print shape every 100 frames
                 print(f"[Capture Thread] Captured frame shape: {frame.shape}, dtype: {frame.dtype}")
            # Convert Pi Camera RGB to BGR for OpenCV if needed
            if camera_handler.camera_type == 'pi' and len(frame.shape) == 3 and frame.shape[2] == 3: # Basic check
                # Assuming picamera gives RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            with lock:
                latest_frame = frame.copy()
            frame_count += 1
        else:
            print("[Capture Thread] Warning: No frame captured.")
        time.sleep(0.01)  # Adjust sleep time as needed (e.g., 1/fps)

# === Preprocessing for ONNX model ===
def preprocess(image):
    original_shape = image.shape[:2] # (height, width)
    
    # Resize image with letterboxing to maintain aspect ratio
    img_h, img_w = original_shape
    net_h, net_w = INPUT_SIZE # (640, 640)

    scale = min(net_w / img_w, net_h / img_h)
    new_unpad_w, new_unpad_h = int(round(img_w * scale)), int(round(img_h * scale))
    
    # Calculate padding
    dw, dh = (net_w - new_unpad_w) / 2, (net_h - new_unpad_h) / 2
    
    if (img_w, img_h) != (new_unpad_w, new_unpad_h):
        resized_image = cv2.resize(image, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized_image = image

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    letterboxed_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(letterboxed_image, cv2.COLOR_BGR2RGB)
    
    # Normalize to 0-1 and transpose from HWC to CHW
    normalized_image = rgb_image.astype(np.float32) / 255.0
    blob = np.transpose(normalized_image, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0) # Add batch dimension: (1, C, H, W)
    
    # Return original shape, new unpadded shape, and padding info for postprocessing
    return blob, original_shape, (new_unpad_w, new_unpad_h), (dw, dh)


# === Postprocessing ONNX outputs for YOLOv8 ===
# YOLOv8 output is typically [batch, 84, 8400] where 84 = 4 (box) + 80 (COCO classes)
# The box is [cx, cy, w, h]
def postprocess(outputs, original_shape, new_unpad_shape, padding_info):
    # `outputs` is a list of output tensors. For YOLOv8, usually one output.
    # Expected shape: (1, 84, N) where N is number of proposals (e.g., 8400)
    # 84 = 4 (cx, cy, w, h) + 80 (class confidences for COCO)
    
    predictions = outputs[0][0] # Shape: (84, N)
    
    # Transpose to (N, 84) to iterate over detections
    predictions = predictions.T # Shape: (N, 84)
    
    img_h, img_w = original_shape # Original image height, width
    net_h, net_w = INPUT_SIZE     # Network input height, width (e.g., 640, 640)
    unpad_w, unpad_h = new_unpad_shape # Size after resize, before padding
    dw, dh = padding_info         # Padding added (width_padding/2, height_padding/2)

    scale_w = img_w / unpad_w
    scale_h = img_h / unpad_h

    boxes, confidences, class_ids = [], [], []
    # print(f"[Postprocess] Number of raw proposals: {len(predictions)}")

    detection_idx = 0
    for pred in predictions:
        # pred shape: (84,)
        # First 4 elements are bbox: cx, cy, w, h
        # Next 80 elements are class confidences
        
        box_params = pred[:4]  # cx, cy, w, h
        class_scores = pred[4:] # 80 class confidences for COCO

        class_id = np.argmax(class_scores)
        score = class_scores[class_id]

        if detection_idx < 5 or score > 0.1: # Print first 5 or high score preds
             #print(f"[Postprocess DEBUG {detection_idx}] Score: {score:.4f}, Class ID: {class_id}, Raw Box: {box_params}")
             pass
        detection_idx += 1
        label = COCO_CLASSES[class_id]


        if score < CONF_THRESH:
            continue

        # Ensure class_id is valid for COCO_CLASSES
        if class_id >= len(COCO_CLASSES):
            #print(f"[Postprocess] Warning: class_id {class_id} is out of bounds for COCO_CLASSES (len {len(COCO_CLASSES)}). Skipping.")
            continue

        if label not in TARGET_CLASSES:
            #print(f"[Postprocess] Label '{label}' not in TARGET_CLASSES. Skipping.")
            continue
        
        #print(f"[Postprocess] Potential Detection: Label: {label}, Score: {score:.2f}, Class ID: {class_id}")

        # Convert box from network input space (cx,cy,w,h) to original image space
        cx, cy, w, h = box_params
        
        # Adjust for padding and scaling
        # 1. Scale to letterbox_image dimensions (before padding removal)
        # These are relative to INPUT_SIZE (e.g., 640x640)
        # cx_net, cy_net, w_net, h_net = cx, cy, w, h
        
        # 2. Convert to x1, y1, x2, y2 relative to letterbox_image (still padded)
        x1_padded = (cx - w / 2) 
        y1_padded = (cy - h / 2) 
        x2_padded = (cx + w / 2) 
        y2_padded = (cy + h / 2) 

        # 3. Remove padding offset
        # (dw and dh are the padding amounts on each side for width and height)
        x1_unpad = x1_padded - dw
        y1_unpad = y1_padded - dh
        x2_unpad = x2_padded - dw
        y2_unpad = y2_padded - dh
        
        # 4. Scale to original image dimensions
        x1_orig = int(x1_unpad * scale_w)
        y1_orig = int(y1_unpad * scale_h)
        x2_orig = int(x2_unpad * scale_w)
        y2_orig = int(y2_unpad * scale_h)

        # Clip to image boundaries
        x1_orig = max(0, x1_orig)
        y1_orig = max(0, y1_orig)
        x2_orig = min(img_w -1, x2_orig)
        y2_orig = min(img_h -1, y2_orig)

        width_orig = x2_orig - x1_orig
        height_orig = y2_orig - y1_orig

        if width_orig <=0 or height_orig <=0:
            continue

        boxes.append([x1_orig, y1_orig, width_orig, height_orig]) # NMSBoxes expects [x, y, w, h]
        confidences.append(float(score))
        class_ids.append(class_id)

    if not boxes:
        #print("[Postprocess] No boxes passed confidence and target class checks.")
        return []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, NMS_THRESH)

    results = []
    if len(indices) > 0:
        #print(f"[Postprocess] NMS kept {len(indices)} boxes out of {len(boxes)}.")
        for i in indices.flatten():
            results.append((boxes[i], class_ids[i], confidences[i]))
            print(f"  -> Final Detection: {COCO_CLASSES[class_ids[i]]} with conf {confidences[i]:.2f} at {boxes[i]}")
    #else:
        #print("[Postprocess] NMS removed all boxes.")
    return results


def play_alert():
    print("Attempting to play alert sound...")
    # Ensure mpg123 is installed: sudo apt-get install mpg123
    # Ensure ALERT_SOUND file exists
    if os.path.exists(ALERT_SOUND):
        # Using os.system in a non-blocking way. Redirecting output to hide mpg123 messages.
        os.system(f"mpg123 -q {ALERT_SOUND} >/dev/null 2>&1 &")
        print(f"Played {ALERT_SOUND} (if mpg123 is installed and sound works).")
    else:
        print(f"Warning: Alert sound file '{ALERT_SOUND}' not found.")

# === Main program ===
if __name__ == "__main__":
    if session is None:
        print("ONNX session not initialized. Exiting.")
        exit()

    print("Starting detection with threaded camera capture...")
    print(f"Attempting to use IP Camera URL: {os.getenv('IP_CAMERA_URL', 'http://192.168.0.2:8080/video')}")

    # Initialize camera (choose 'pi', 'ip', or 'usb')
    # For IP camera, ensure the URL is correct.
    # For USB camera, camera_index might be 0, 1, etc.
    # For Pi camera, ensure picamera library and camera are set up.
    IP_CAMERA_URL = os.getenv('IP_CAMERA_URL', "http://192.168.0.2:8080/video") # Replace with your IP camera's stream URL or use env var
    # camera_handler = CameraHandler(camera_type='usb', camera_index=0)
    # camera_handler = CameraHandler(camera_type='ip', ip_camera_url=IP_CAMERA_URL)
    camera_handler = CameraHandler(camera_type='pi')


    # Start frame capture thread
    thread = threading.Thread(target=capture_frames, args=(camera_handler,))
    thread.daemon = True # Allow main program to exit even if thread is running
    thread.start()

    print("Frame capture thread started. Waiting for first frame...")
    time.sleep(2) # Give camera and thread some time to initialize

    frame_display_time = time.time()
    detection_count = 0

    try:
        while True:
            current_frame_for_processing = None
            with lock:
                if latest_frame is not None:
                    current_frame_for_processing = latest_frame.copy()
                else:
                    # If latest_frame is None even after startup, camera_handler might have issues
                    # print("[Main Loop] Waiting for frame from capture thread...")
                    time.sleep(0.1) # Wait a bit longer if no frame yet
                    continue # Skip processing if no frame

            if current_frame_for_processing is not None:
                if current_frame_for_processing.size == 0:
                    print("[Main Loop] Error: Captured frame is empty.")
                    time.sleep(0.1)
                    continue
                
                # print(f"[Main Loop] Processing frame of shape: {current_frame_for_processing.shape}")
                
                # Preprocess
                input_tensor, original_shape, new_unpad_shape, padding_info = preprocess(current_frame_for_processing)
                # print(f"[Main Loop] Input tensor shape: {input_tensor.shape}, Original shape: {original_shape}")

                # Run inference
                outputs = session.run(output_names, {input_name: input_tensor})
                # print(f"[Main Loop] Model outputs count: {len(outputs)}")
                # print(f"[Main Loop] Model output 0 shape: {outputs[0].shape}")


                # Postprocess
                detections = postprocess(outputs, original_shape, new_unpad_shape, padding_info)

                # Add these imports at the top
                from face_recognition_handler import FaceRecognitionHandler
                from alert_owner import send_email_with_attachment_from_frame

                # Add after other global variables
                face_recognition_handler = None
                try:
                    face_recognition_handler = FaceRecognitionHandler()
                    print("Face recognition system initialized successfully")
                except Exception as e:
                    print(f"Warning: Could not initialize face recognition: {e}")

                # In the main loop, modify the detection handling section:
                human_count = 0
                intruder_detected = False
                processed_frame_for_display = current_frame_for_processing.copy()

                for box, class_id, score in detections:
                    label = COCO_CLASSES[class_id]
                    x, y, w, h = box
                    
                    if label == "person":
                        human_count += 1
                        # Check if this person is the owner
                        person_frame = processed_frame_for_display[y:y+h, x:x+w]
                        if face_recognition_handler is not None:
                            faces = face_recognition_handler.identify_faces(person_frame)
                            if faces:
                                is_owner = any(face['is_owner'] for face in faces)
                                if not is_owner:
                                    intruder_detected = True
                                    cv2.rectangle(processed_frame_for_display, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for intruder
                                else:
                                    cv2.rectangle(processed_frame_for_display, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for owner
                            else:
                                # No faces detected clearly, mark as potential intruder
                                intruder_detected = True
                                cv2.rectangle(processed_frame_for_display, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow for unclear
                    elif label in ANIMAL_CLASSES:
                        animal_count += 1
                        cv2.rectangle(processed_frame_for_display, (x, y), (x + w, y + h), (255, 165, 0), 2)  # Orange for animals
                    
                    text = f"{label}: {score:.2f}"
                    cv2.putText(processed_frame_for_display, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Only alert if an intruder is detected
                if intruder_detected:
                    print("ALERT: Intruder detected!")
                    try:
                        send_email_with_attachment_from_frame(processed_frame_for_display)
                        print("Intruder alert email sent successfully")
                    except Exception as e:
                        print(f"Failed to send intruder alert: {e}")

                # Continue with animal detection alerts as before
                if animal_count >= 1:
                    detection_count += 1
                    print(f"ALERT: Animals detected: {animal_count}")
                    make_sound()

                else:
                    print("No target humans or animals detected in this frame.")

                # Display the frame (optional, can be slow)
                # cv2.imshow("Detections", processed_frame_for_display)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break
                
                # Limit processing rate slightly if needed, or rely on camera frame rate
                time.sleep(0.01) # Small sleep to yield CPU

            else:
                # This case should be rare if capture_frames is working and thread is given time to start
                print("[Main Loop] No frame available for processing. Retrying...")
                time.sleep(0.1) # Wait if no frame was available from the thread

    except KeyboardInterrupt:
        print("Exiting gracefully...")
    finally:
        running = False
        if 'thread' in locals() and thread.is_alive():
            print("Waiting for frame capture thread to join...")
            thread.join(timeout=5) # Wait for 5 seconds
            if thread.is_alive():
                print("Warning: Frame capture thread did not join in time.")
        if 'camera_handler' in locals():
            camera_handler.release()
        cv2.destroyAllWindows()
        print("Resources released. Exiting program.")