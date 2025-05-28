import cv2
import time
import threading
from ultralytics import YOLO
import platform

# Detectar si es Jetson (aarch64)
IS_JETSON = platform.machine() == "aarch64"

# Pines GPIO por cada bote (ajusta si usas otros)
botes_a_pines = {
    "BotePlastico": 33,
    # "BotePapel": 35,
    # "BoteBateria": 37,
    "BoteVidrio": 32
}

# PWM de cada servo
pwm_servos = {}

if IS_JETSON:
    import Jetson.GPIO as GPIO
    GPIO.setmode(GPIO.BOARD)

    for bote, pin in botes_a_pines.items():
        GPIO.setup(pin, GPIO.OUT)
        pwm = GPIO.PWM(pin, 50)  # 50Hz
        pwm.start(0)
        pwm_servos[bote] = pwm
        
# Estados y temporizadores
botes_estado = {bote: False for bote in botes_a_pines}
botes_timers = {bote: None for bote in botes_a_pines}

# Mapeo clase Bote
clase_a_bote = {
    "HojaPapel": "BotePapel",
    "TazaPapel": "BotePapel",
    "BotellaPlastico": "BotePlastico",
    "TapaPlastico": "BotePlastico",
    "BotellaVidrio": "BoteVidrio",
    "Bateria": "BoteBateria"
}

# Angulo cerrado por defecto (todos se abren a 0 y cierran a 90)
ANGULO_ABIERTO = 0
ANGULO_CERRADO = 90

def mover_servo(bote, angulo):
    if IS_JETSON:
        duty = 2 + (angulo / 18)
        pwm_servos[bote].ChangeDutyCycle(duty)
        time.sleep(0.5)
        pwm_servos[bote].ChangeDutyCycle(0)
    else:
        print(f"[PC Simulacion] {bote} moveria a {angulo} grados")

def desactivar_bote(bote):
    botes_estado[bote] = False
    botes_timers[bote] = None
    mover_servo(bote, ANGULO_CERRADO)
    print(f"{bote} DESACTIVADO (cerrado)")

def activar_bote(bote):
    if "BotePlastico" in bote or bote == "BoteVidrio":
        if not botes_estado[bote]:
            botes_estado[bote] = True
            print(f"{bote} ACTIVADO (abierto)")
            mover_servo(bote, ANGULO_ABIERTO)
        else:
            print(f"{bote} ya activo, reiniciando temporizador.")

        # Cancelar y reiniciar temporizador
        if botes_timers[bote]:
            botes_timers[bote].cancel()
        t = threading.Timer(5, desactivar_bote, args=[bote])
        botes_timers[bote] = t
        t.start()

model_path = 'Model/trash_yolov8n_640_v1.pt'
model = YOLO(model_path)

# Tiempo del ultimo analisis
last_detection_time = 0
detection_interval = 0.3  # segundos

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la camara.")
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

# Limpieza
if IS_JETSON:
    for pwm in pwm_servos.values():
        pwm.stop()
    GPIO.cleanup()
