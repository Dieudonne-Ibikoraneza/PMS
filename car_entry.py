import cv2
from ultralytics import YOLO
import pytesseract
import os
import time
import serial
import serial.tools.list_ports
import csv
from collections import Counter

# Load YOLOv8 model
model = YOLO('./best.pt')

# Plate save directory
save_dir = 'plates'
os.makedirs(save_dir, exist_ok=True)

# CSV log file
csv_file = 'plates_log.csv'
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Plate Number', 'Payment Status', 'Timestamp'])

# ===== Auto-detect Arduino Serial Port =====
def detect_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        if "usbmodem" in port.device or "wchusbmodem" in port.device or "ttyUSB" in port.device or "ttyACM" in port.device:
            return port.device
    return None

arduino_port = detect_arduino_port()
if arduino_port:
    print(f"[CONNECTED] Arduino on {arduino_port}")
    arduino = serial.Serial(arduino_port, 9600, timeout=1)
    time.sleep(2)
else:
    print("[ERROR] Arduino not detected.")
    arduino = None

# ===== Ultrasonic Sensor Setup =====
import random
def mock_ultrasonic_distance(state):
    if state == "approaching":
        return random.randint(10, 40)
    return random.randint(60, 150)

# Initialize webcam
cap = cv2.VideoCapture(0)
plate_buffer = []
entry_cooldown = 300  # 5 minutes
last_saved_plate = None
last_entry_time = 0
vehicle_state = "away"  # Track vehicle presence
last_detection_time = 0
detection_timeout = 10  # Stop detecting after 10 seconds of continuous detection

print("[SYSTEM] Ready. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # Simulate vehicle approaching and leaving
    if vehicle_state == "away" and random.random() < 0.1:  # 10% chance to approach
        vehicle_state = "approaching"
    elif vehicle_state == "approaching" and random.random() < 0.2:  # 20% chance to leave
        vehicle_state = "away"

    distance = mock_ultrasonic_distance(vehicle_state)
    print(f"[SENSOR] Distance: {distance} cm, State: {vehicle_state}")

    if distance <= 50:
        last_detection_time = current_time  # Update detection time
        results = model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_img = frame[y1:y2, x1:x2]

                # Plate Image Processing
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                # OCR Extraction
                plate_text = pytesseract.image_to_string(
                    thresh, config='--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                ).strip().replace(" ", "")
                print(f"[OCR] Raw Plate Text: {plate_text}")

                # Plate Validation
                if "RA" in plate_text:
                    start_idx = plate_text.find("RA")
                    plate_candidate = plate_text[start_idx:]
                    if len(plate_candidate) >= 7:
                        plate_candidate = plate_candidate[:7]
                        prefix, digits, suffix = plate_candidate[:3], plate_candidate[3:6], plate_candidate[6]
                        if (prefix.isalpha() and prefix.isupper() and
                            digits.isdigit() and suffix.isalpha() and suffix.isupper()):
                            print(f"[VALID] Plate Detected: {plate_candidate}")
                            plate_buffer.append(plate_candidate)

                            # Process buffer after 3 detections
                            if len(plate_buffer) >= 3:
                                most_common = Counter(plate_buffer).most_common(1)[0][0]
                                print(f"[BUFFER] Most Common Plate: {most_common}")

                                if (most_common != last_saved_plate or
                                    (current_time - last_entry_time) > entry_cooldown):
                                    with open(csv_file, 'a', newline='') as f:
                                        writer = csv.writer(f)
                                        writer.writerow([most_common, 0, time.strftime('%Y-%m-%d %H:%M:%S')])
                                    print(f"[SAVED] {most_common} logged to CSV.")

                                    if arduino:
                                        try:
                                            arduino.write(b'1')
                                            print("[GATE] Opening gate (sent '1')")
                                            time.sleep(15)
                                            arduino.write(b'0')
                                            print("[GATE] Closing gate (sent '0')")
                                        except serial.SerialException:
                                            print("[ERROR] Arduino communication failed.")

                                    last_saved_plate = most_common
                                    last_entry_time = current_time
                                else:
                                    print("[SKIPPED] Duplicate within 5 min window.")

                                plate_buffer.clear()
                                vehicle_state = "away"  # Simulate vehicle leaving

                cv2.imshow("Plate", plate_img)
                cv2.imshow("Processed", thresh)
                time.sleep(0.5)

    # Stop detection if vehicle is present too long
    if vehicle_state == "approaching" and (current_time - last_detection_time) > detection_timeout:
        print("[TIMEOUT] No new plates, resetting detection.")
        vehicle_state = "away"
        plate_buffer.clear()

    annotated_frame = results[0].plot() if distance <= 50 else frame
    cv2.imshow('Webcam Feed', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if arduino:
    arduino.close()
cv2.destroyAllWindows()