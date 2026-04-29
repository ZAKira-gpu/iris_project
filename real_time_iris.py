import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# 1. Model Loading Implementation
def load_feature_extractor(model_path='iris_feature_extractor.h5'):
    """
    Loads the pre-saved local DenseNet-201 feature extractor.
    This ensures no internet connection is required and the model is shipped with the code.
    """
    if not os.path.exists(model_path):
        raise IOError(f"Cannot find the AI model file: {model_path}. Ensure it is in the directory.")
    
    print(f"Loading local AI model from {model_path}...")
    # Load the entire model architecture and weights from the local file
    model = tf.keras.models.load_model(model_path)
    return model

# 2. Pipeline Components
class IrisService:
    def __init__(self):
        # Load eye detection cascade
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        if self.eye_cascade.empty():
            raise IOError("Cannot load haarcascade_eye.xml. Ensure it exists in the root directory.")
        
        # Load the local model
        self.model = load_feature_extractor()
        
        # Constants from original repository
        self.ROI_SIZE = (200, 150)
        self.MODEL_INPUT_SIZE = (70, 70)
        self.KERNEL = np.ones((5, 5), np.uint8)

    def detect_eye(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize for faster detection as per original logic
        resized_gray = cv2.resize(gray, (400, 300))
        eyes = self.eye_cascade.detectMultiScale(resized_gray, 1.1, 5)
        
        if len(eyes) == 0:
            return None, None
            
        # Select the largest eye found
        max_area = 0
        best_eye = None
        for (x, y, w, h) in eyes:
            if w * h > max_area:
                max_area = w * h
                best_eye = (x, y, w, h)
        
        # Scale coordinates back to original frame size
        orig_h, orig_w = gray.shape
        scale_x = orig_w / 400
        scale_y = orig_h / 300
        
        x, y, w, h = best_eye
        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)
        
        return (x, y, w, h), gray[y:y+h, x:x+w]

    def segment_iris(self, eye_roi):
        """
        Maintains the core segmentation math using Hough Circles and ROI extraction.
        """
        # Enhance contrast for better circle detection
        eye_roi_colored = cv2.cvtColor(eye_roi, cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(eye_roi, cv2.HOUGH_GRADIENT, 1, 50,
                                  param1=100, param2=30, minRadius=10, maxRadius=100)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Find the best circle (using average intensity as per original logic)
            min_avg = 255
            best_circle = None
            
            for (x, y, r) in circles:
                # Check bounds
                if x-r > 0 and y-r > 0 and x+r < eye_roi.shape[1] and y+r < eye_roi.shape[0]:
                    roi = eye_roi[y-r:y+r, x-r:x+r]
                    avg = np.average(roi)
                    if avg < min_avg:
                        min_avg = avg
                        best_circle = (x, y, r)
            
            if best_circle:
                cx, cy, r = best_circle
                # Extract iris ROI (using radius 40 as per original script)
                fixed_radius = 40
                
                # Check bounds for fixed radius extraction
                if cy-fixed_radius > 0 and cx-fixed_radius > 0 and \
                   cy+fixed_radius < eye_roi.shape[0] and cx+fixed_radius < eye_roi.shape[1]:
                    iris_roi = eye_roi[cy-fixed_radius:cy+fixed_radius, cx-fixed_radius:cx+fixed_radius]
                else:
                    # Fallback to center extraction if bounds are tight
                    mid_y, mid_x = eye_roi.shape[0]//2, eye_roi.shape[1]//2
                    iris_roi = eye_roi[mid_y-fixed_radius:mid_y+fixed_radius, mid_x-fixed_radius:mid_x+fixed_radius]
                
                iris_roi = cv2.resize(iris_roi, self.ROI_SIZE)
                return iris_roi, (cx, cy, r)
                
        return None, None

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Real-time Iris Service Started. Press 'q' to exit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display frame
            display_frame = frame.copy()
            
            try:
                # 1. Eye Detection
                eye_coords, eye_roi = self.detect_eye(frame)
                
                if eye_coords is not None:
                    ex, ey, ew, eh = eye_coords
                    cv2.rectangle(display_frame, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
                    
                    # 2. Iris Segmentation
                    iris_roi, circle = self.segment_iris(eye_roi)
                    
                    if iris_roi is not None:
                        cx, cy, r = circle
                        # Draw circle on display frame (relative to eye box)
                        cv2.circle(display_frame, (ex + cx, ey + cy), r, (0, 255, 0), 2)
                        
                        # 3. Feature Extraction
                        # Convert to 3-channel and resize for model
                        iris_input = cv2.cvtColor(iris_roi, cv2.COLOR_GRAY2RGB)
                        iris_input = cv2.resize(iris_input, self.MODEL_INPUT_SIZE)
                        iris_input = iris_input.astype('float32') / 255.0
                        iris_input = np.expand_dims(iris_input, axis=0)
                        
                        # Get vector
                        vector = self.model.predict(iris_input, verbose=0)[0]
                        
                        # 4. Console Output (Real-time)
                        print(f"\rIris Vector [First 10]: {vector[:10]}", end="")
                else:
                    # Self-healing: Skip frame if no eye detected
                    pass
                    
            except Exception as e:
                # Self-healing: Catch unexpected errors and continue
                print(f"\nError processing frame: {e}")
                continue

            cv2.imshow('Real-time Iris Vector Extraction', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        print("\nService stopped.")

if __name__ == "__main__":
    service = IrisService()
    service.run()
