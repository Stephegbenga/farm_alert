import os
import sys

# Import necessary libraries
import cv2
# import tensorflow as tf # Placeholder for TensorFlow or PyTorch
from ultralytics import YOLO

def identify_entities(input_source):
    """
    Processes the input_source (image or video) to identify animals and humans.

    Args:
        input_source (str): Path to an image file or video source.
    """
    print(f"Processing input: {input_source}")

    # 1. Load the input (image or video frame)
    try:
        img = cv2.imread(input_source)
        if img is None:
            print(f"Error: Could not read image from {input_source}")
            return
        print(f"Successfully loaded image: {input_source} with dimensions: {img.shape}")
        # Display the image (optional, for testing)
        # cv2.imshow("Loaded Image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error loading image {input_source}: {e}")
        return

    # 2. Preprocessing is handled by the YOLO model internally for common image formats.

    # 3. Load the pre-trained YOLO model
    try:
        model = YOLO('yolov8n.pt')  # Load a pre-trained YOLOv8n model
        print("Successfully loaded YOLOv8n model.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # 4. Perform inference/prediction
    try:
        results = model(img)  # Perform detection
        print("Inference completed.")
    except Exception as e:
        print(f"Error during model inference: {e}")
        return

    # 5. Post-process the results
    detected_humans = 0
    detected_animals = 0
    animal_classes = {
        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse',
        18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
        22: 'zebra', 23: 'giraffe'
    } # COCO class IDs for some animals

    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls)
            if class_id == 0: # COCO class ID for 'person'
                detected_humans += 1
            elif class_id in animal_classes:
                detected_animals += 1
                print(f"Detected animal: {animal_classes[class_id]}")

    # 6. Output the results
    print(f"--- Detection Summary ---")
    print(f"Detected Humans: {detected_humans}")
    print(f"Detected Animals: {detected_animals}")
    print("-------------------------")

    # Optional: Display the image with detections
    # try:
    #     annotated_frame = results[0].plot() # Plot results on the image
    #     cv2.imshow("YOLO Detections", annotated_frame)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # except Exception as e:
    #     print(f"Error displaying annotated image: {e}")

if __name__ == "__main__":
    print("Farm Alert - Animal and Human Identification System")
    print("---------------------------------------------------")

    # Example usage: Replace with actual input source later
    # For now, we can use a placeholder or command-line argument
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        identify_entities(input_path)
    else:
        print("Please provide an image or video path as a command-line argument.")
        print("Example: python main.py path/to/your/image.jpg")

    # Future enhancements:
    # - Configuration file for settings (model paths, thresholds, etc.)
    # - Logging
    # - More robust input handling (camera streams, directories of images)