import cv2
import torch

# Load the trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt')

# Open DroidCam stream (replace with 1 for USB or the URL for Wi-Fi)
cap = cv2.VideoCapture(1)  # or use url = "http://<ip_address>:4747/video" for Wi-Fi

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Render results on the frame
    frame = results.render()[0]

    # Display the frame with detection
    cv2.imshow('YOLOv5 Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
