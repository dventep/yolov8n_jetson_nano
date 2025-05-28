import Jetson.GPIO as GPIO
import time

# Configurar el pin
servo_pin = 33  # GPIO13 en modo BOARD
GPIO.setmode(GPIO.BOARD)
GPIO.setup(servo_pin, GPIO.OUT)

# Crear la señal PWM en 50Hz (frecuencia típica de servos)
pwm = GPIO.PWM(servo_pin, 50)
pwm.start(0)  # Comenzar con duty cycle en 0%

# Función para mover el servo a un ángulo (duty cycle)
def set_angle(angle):
    duty = 2 + (angle / 18)
    GPIO.output(servo_pin, True)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)
    GPIO.output(servo_pin, False)
    pwm.ChangeDutyCycle(0)

try:
    while True:
        print("Moviendo a 0 grados")
        set_angle(0)
        time.sleep(1)

        print("Moviendo a 90 grados")
        set_angle(90)
        time.sleep(1)

        print("Moviendo a 180 grados")
        set_angle(180)
        time.sleep(1)

except KeyboardInterrupt:
    print("Finalizando programa")

finally:
    pwm.stop()
    GPIO.cleanup()