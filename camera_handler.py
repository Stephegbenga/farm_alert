import cv2
import time

# Try to import Picamera2, but don't fail if it's not available (e.g., running on a non-Pi system)
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    print("Picamera2 library not found. Raspberry Pi camera functionality will be unavailable.")

class CameraHandler:
    def __init__(self, camera_type='ip', ip_camera_url="http://192.168.1.100:8080/video", pi_camera_config=None):
        """
        Initializes the camera handler.
        :param camera_type: 'ip' for IP camera, 'pi' for Raspberry Pi camera.
        :param ip_camera_url: URL for the IP camera stream.
        :param pi_camera_config: Configuration dictionary for PiCamera2. 
                                 Example: {'size': (1280, 720), 'format': 'RGB888'}
        """
        self.camera_type = camera_type
        self.ip_camera_url = ip_camera_url
        self.pi_camera_config = pi_camera_config if pi_camera_config else {'size': (1280, 1280), 'format': 'RGB888'}
        self.cap = None
        self.picam2 = None

        if self.camera_type == 'ip':
            self._initialize_ip_camera()
        elif self.camera_type == 'pi':
            if PICAMERA_AVAILABLE:
                self._initialize_pi_camera()
            else:
                raise RuntimeError("Picamera2 library is not available, cannot initialize Raspberry Pi camera.")
        else:
            raise ValueError("Invalid camera_type. Choose 'ip' or 'pi'.")

    def _initialize_ip_camera(self):
        print(f"Initializing IP camera at {self.ip_camera_url}")
        self.cap = cv2.VideoCapture(self.ip_camera_url)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open IP camera stream at {self.ip_camera_url}")
        print("IP camera initialized successfully.")

    def _initialize_pi_camera(self):
        print("Initializing Raspberry Pi camera...")
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": self.pi_camera_config['size'], "format": self.pi_camera_config['format']}
        )
        self.picam2.configure(config)
        self.picam2.start()
        # Allow some time for camera to initialize
        time.sleep(5) 
        print("Raspberry Pi camera initialized successfully.")

    def get_frame(self):
        """Captures a frame from the initialized camera."""
        if self.camera_type == 'ip':
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("Warning: Failed to grab frame from IP camera.")
                    return None
                return frame
            else:
                print("Error: IP camera not initialized or already released.")
                return None
        elif self.camera_type == 'pi':
            if self.picam2:
                frame = self.picam2.capture_array()
                return frame # Picamera2 captures in RGB by default if format is RGB888
            else:
                print("Error: Raspberry Pi camera not initialized.")
                return None
        return None

    def release(self):
        """Releases the camera resources."""
        print("Releasing camera resources...")
        if self.camera_type == 'ip' and self.cap:
            self.cap.release()
            print("IP camera released.")
        elif self.camera_type == 'pi' and self.picam2:
            self.picam2.stop()
            print("Raspberry Pi camera stopped.")
        cv2.destroyAllWindows()

# Example Usage (optional, for testing this module directly):
if __name__ == '__main__':
    # Test IP Camera
    try:
        print("Testing IP Camera...")
        ip_cam = CameraHandler(camera_type='ip')
        for _ in range(5): # Capture 5 frames
            frame = ip_cam.get_frame()
            if frame is not None:
                pass
            else:
                print("Failed to get frame from IP camera.")
                break
        ip_cam.release()
    except Exception as e:
        print(f"Error testing IP camera: {e}")

    # Test Pi Camera (only if available)
    if PICAMERA_AVAILABLE:
        try:
            print("\nTesting Raspberry Pi Camera...")
            # You might need to adjust pi_camera_config based on your Pi Camera version and needs
            pi_cam_config = {'size': (640, 480), 'format': 'RGB888'}
            pi_cam = CameraHandler(camera_type='pi', pi_camera_config=pi_cam_config)
            for _ in range(5):
                frame = pi_cam.get_frame()
                if frame is not None:
                    # PiCamera2 captures in RGB, OpenCV expects BGR for imshow
                    # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    # cv2.imshow("Pi Camera Test", frame) # If format is XRGB888 or similar, direct show might work
                    pass
                else:
                    print("Failed to get frame from Pi camera.")
                    break
            pi_cam.release()
        except Exception as e:
            print(f"Error testing Pi camera: {e}")
    else:
        print("\nSkipping Raspberry Pi camera test as Picamera2 library is not available.")

    cv2.destroyAllWindows()