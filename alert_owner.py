import requests
import base64
import json
import cv2

def send_email_with_attachment_from_frame(frame):
    # Encode OpenCV frame to PNG in memory
    success, encoded_image = cv2.imencode('.png', frame)
    if not success:
        raise ValueError("Failed to encode image.")

    # Convert to base64 string
    img_bytes = encoded_image.tobytes()
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    data_uri = f"data:image/png;base64,{base64_str}"

    # Payload
    payload = {
        "email": "adegbamiyestephen2018@gmail.com",
        "subject": "Security Alert: Human Detected",
        "message": "<b>A human has been detected multiple times in the surveillance area.</b>",
        "attachmentBase64": data_uri,
        "attachmentName": "detection_alert.png"
    }

    # POST request
    url = "https://script.google.com/macros/s/AKfycbyDMIcmdplU_cGEjFggdqMHaatkYPet6Bc3uD2oQjTaH3QUm0KFpEUq2uP62jD3gMJk/exec"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=json.dumps(payload), headers=headers)

    print("Status:", response.status_code)
    print("Response:", response.text)
