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
        
        # Find all faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # If no faces found, return None
        if not face_encodings:
            return None
        
        # Compare each face with owner's face
        for face_encoding in face_encodings:
            # Compare with owner's face encoding
            matches = face_recognition.compare_faces([self.owner_encoding], face_encoding, tolerance=0.6)
            if matches[0]:
                return True  # Owner found
        
        return False  # No match found - intruder detected