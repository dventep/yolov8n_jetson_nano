import cv2
import time
import threading
from ultralytics import YOLO
import platform

# Detectar si es Jetson (aarch64)
IS_JETSON = platform.machine() == "aarch64"

if IS_JETSON:
    import Jetson.GPIO as GPIO
    GPIO.setmode(GPIO.BOARD)
    SERVO_PIN = 33
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    pwm = GPIO.PWM(SERVO_PIN, 50)
    pwm.start(0)

def mover_servo_angulo(angulo):
    if IS_JETSON:
        duty = 2 + (angulo / 18)
        pwm.ChangeDutyCycle(duty)
        time.sleep(0.5)
        pwm.ChangeDutyCycle(0)
    else:
        print(f"[PC Simulación] Servo movería a {angulo}°")


# Estados de los botes
botes_estado = {
    "BotePlastico": False,
    "BotePapel": False,
    "BoteBateria": False,
    "BoteVidrio": False
}

# Mapeo clase → Bote
clase_a_bote = {
    "HojaPapel": "BotePapel",
    "TazaPapel": "BotePapel",
    "BotellaPlastico": "BotePlastico",
    "TapaPlastico": "BotePlastico",
    "BotellaVidrio": "BoteVidrio",
    "Bateria": "BoteBateria"
}

botes_timers = {
    "BotePlastico": None,
    "BotePapel": None,
    "BoteBateria": None,
    "BoteVidrio": None
}

bote_a_angulo = {
    "BotePlastico": 90,
    "BotePapel": 45,
    "BoteVidrio": 135,
    "BoteBateria": 0
}

def desactivar_bote(bote):
    botes_estado[bote] = False
    botes_timers[bote] = None
    print(f"{bote} DESACTIVADO")

def activar_bote(bote):
    if not botes_estado[bote]:
        botes_estado[bote] = True
        print(f"{bote} ACTIVADO")
        mover_servo_angulo(bote_a_angulo[bote])
    else:
        print(f"{bote} ya activo, reiniciando temporizador.")

    # Cancelar temporizador anterior si existe
    if botes_timers[bote]:
        botes_timers[bote].cancel()

    # Reiniciar temporizador de 5s
    t = threading.Timer(5, desactivar_bote, args=[bote])
    botes_timers[bote] = t
    t.start()

model_path = 'Model/trash_yolov8n_640_v1.pt'
model = YOLO(model_path)

# Tiempo del último análisis
last_detection_time = 0
detection_interval = 0.3  # segundos

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    if current_time - last_detection_time >= detection_interval:
        last_detection_time = current_time
        results = model.predict(source=frame, conf=0.6, stream=False)
        annotated_frame = results[0].plot()

        # Leer predicciones
        for box in results[0].boxes:
            cls_index = int(box.cls[0].item())
            nombre_clase = model.names[cls_index]
            bote = clase_a_bote.get(nombre_clase)

            if bote:
                activar_bote(bote)
    else:
        annotated_frame = annotated_frame if 'annotated_frame' in locals() else frame

    cv2.imshow("YOLOv8 Object Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if IS_JETSON:
    pwm.stop()
    GPIO.cleanup()