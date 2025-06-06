# ultrasonic.py (Main script to listen for object detection and launch radar + webcam)

import serial
import time
import subprocess

# Adjust COM port if needed
COM_PORT = 'COM3'
BAUD_RATE = 9600

try:
    arduino = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Let the serial connection settle
    print(f"Connected to Arduino on {COM_PORT}")
except Exception as e:
    print(f"Error connecting to Arduino: {e}")
    exit()

print("Listening for object detection...")

while True:
    try:
        data = arduino.readline().decode().strip()
        if not data:
            continue

        print(f"Arduino: {data}")

        if data == "DETECT":
            print("Object detected! Launching YOLO + Radar Interface...")

            # Close serial before launching radar_cam.py to avoid COM conflict
            arduino.close()

            # Run YOLO and radar visualization
            subprocess.run(["python", "radar_ultra.py"])

            # Reopen serial after subprocess completes
            arduino = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
            time.sleep(2)

            print("Ready for next detection...")

    except Exception as e:
        print(f"Error: {e}")