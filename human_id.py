import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

model = YOLO("yolov8n.pt")
tracker = Sort()

cap = cv2.VideoCapture("football.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    detections = []

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls == 0:  # person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append([x1, y1, x2, y2, 0.9])  # dummy conf

    dets_np = np.array(detections)

    tracks = tracker.update(dets_np)

    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Person {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Human ID Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

