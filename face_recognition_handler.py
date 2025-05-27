import face_recognition
import numpy as np
import cv2
import os

class FaceRecognitionHandler:
    def __init__(self):
        # Load owner's face encoding
        owner_image_path = os.path.join('assets', 'owner.jpeg')
        owner_image = face_recognition.load_image_file(owner_image_path)
        self.owner_encoding = face_recognition.face_encodings(owner_image)[0]

    def identify_person(self, frame):
        # Convert BGR (OpenCV) to RGB (face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame for better face detection (smaller faces)
        height, width = frame.shape[:2]
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=2, fy=2)  # Double the size
        
        # Find all faces in the frame with more sensitive parameters
        face_locations = face_recognition.face_locations(
            small_frame,
            model="hog",  # Use HOG model which is better for CPU
            number_of_times_to_upsample=2  # Increase sensitivity
        )
        print(f"Number of faces detected: {len(face_locations)}")
        
        if not face_locations:
            print("No faces detected in frame")
            return None
        
        # Adjust face locations back to original frame size
        face_locations = [(int(top/2), int(right/2), int(bottom/2), int(left/2)) 
                         for top, right, bottom, left in face_locations]
        
        # Draw rectangles around detected faces for debugging
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        print(f"Number of face encodings generated: {len(face_encodings)}")
        
        if not face_encodings:
            print("No face encodings could be generated")
            return None
        
        # Compare each face with owner's face
        for face_encoding in face_encodings:
            # Compare with owner's face encoding with more tolerance
            matches = face_recognition.compare_faces([self.owner_encoding], face_encoding, tolerance=0.5)  # Increased tolerance
            print(f"Face match result: {matches[0]}")
            if matches[0]:
                return True  # Owner found
        
        return False  # No match found - intruder detected