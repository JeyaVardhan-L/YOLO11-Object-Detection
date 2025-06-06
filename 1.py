# radar_cam.py (YOLOv8 + radar interface with checkline background)
from ultralytics import YOLO
import cv2
import numpy as np


model = YOLO(r"C:\Users\Phanisri Matta\OneDrive\Dokumen\Desktop\airborne_detection\YOLO11-Object-Detection-master\runs\detect\train\weights\best.pt")

# Start webcam
cap = cv2.VideoCapture(1)

# Radar window size
radar_size = 600
center = (radar_size // 2, radar_size // 2)
radius = radar_size // 2 - 10

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for consistency
    frame = cv2.resize(frame, (radar_size, radar_size))

    # Run YOLOv8 detection
    results = model.predict(source=frame, verbose=False)

    # Draw annotated frame
    annotated_frame = frame.copy()
    radar_frame = np.zeros((radar_size, radar_size, 3), dtype=np.uint8)

    # Radar grid
    for r in range(50, radius + 1, 50):
        cv2.circle(radar_frame, center, r, (0, 100, 0), 1)
    for x in range(0, radar_size, 50):
        cv2.line(radar_frame, (x, 0), (x, radar_size), (0, 60, 0), 1)
    for y in range(0, radar_size, 50):
        cv2.line(radar_frame, (0, y), (radar_size, y), (0, 60, 0), 1)
    cv2.line(radar_frame, (center[0], 0), (center[0], radar_size), (0, 255, 0), 1)
    cv2.line(radar_frame, (0, center[1]), (radar_size, center[1]), (0, 255, 0), 1)

    for result in results:
        frame_h, frame_w = frame.shape[:2]

        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            obj_center_x = (x1 + x2) // 2
            obj_center_y = (y1 + y2) // 2

            # Radar translation
            dx = (obj_center_x - frame_w / 2) / (frame_w / 2)
            dy = (obj_center_y - frame_h / 2) / (frame_h / 2)
            radar_x = int(center[0] + dx * radius)
            radar_y = int(center[1] + dy * radius)

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Radar blip
            cv2.circle(radar_frame, (radar_x, radar_y), 6, (0, 0, 255), -1)

    combined_view = np.hstack((annotated_frame, radar_frame))
    cv2.imshow("YOLO + Radar View", combined_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()