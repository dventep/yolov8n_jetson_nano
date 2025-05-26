import cv2
import time
import threading
from ultralytics import YOLO

model_path = 'Model/trash_yolov8n_640_v1.pt'
model = YOLO(model_path)

cap = cv2.VideoCapture(0)

# Tiempo del último análisis
last_detection_time = 0
detection_interval = 0.3  # segundos

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    current_time = time.time()

    # Solo detectar si han pasado 0.3 segundos desde la última detección
    if current_time - last_detection_time >= detection_interval:
        last_detection_time = current_time

        # Ejecutar detección
        results = model.predict(source=frame, conf=0.6, stream=False)
        annotated_frame = results[0].plot()
    else:
        # En caso contrario, seguir mostrando el último frame anotado
        annotated_frame = annotated_frame if 'annotated_frame' in locals() else frame

    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()