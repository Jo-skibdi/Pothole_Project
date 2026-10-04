import cv2
from ultralytics import YOLO

def detect_live():
    # Load the downloaded weights (make sure best.pt is in the same folder as this script)
    model = YOLO('best.pt')
    
    # Open your laptop's default webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    print("Starting live detection. Press 'q' to quit.")
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Run YOLO inference
        results = model.predict(source=frame, conf=0.5, verbose=False)
        
        # Draw bounding boxes
        annotated_frame = results[0].plot()
        
        # Show the video feed
        cv2.imshow("Pothole Detection - Live", annotated_frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_live()