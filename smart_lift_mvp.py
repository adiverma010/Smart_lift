import cv2
from ultralytics import YOLO
import time

# Load YOLOv8 model (downloads automatically first time)
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

button_pressed = False
lift_full = False
decision = "WAITING..."

last_press_time = 0
delay_seconds = 3  # wait after button press

print("\nControls:")
print("Press 'b' -> simulate button press")
print("Press 'f' -> toggle lift full")
print("Press 'q' -> quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    person_detected = False

    # Run detection
    results = model(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":
                person_detected = True

                # Draw box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Decision logic after delay
    if button_pressed and (time.time() - last_press_time > delay_seconds):
        if person_detected:
            if not lift_full:
                decision = "STOP"
            else:
                decision = "SKIP (FULL)"
        else:
            decision = "SKIP (NO PERSON)"
        button_pressed = False  # reset

    # Display status
    cv2.putText(frame, f"Decision: {decision}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(frame, f"Lift Full: {lift_full}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Smart Lift MVP", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('b'):
        button_pressed = True
        last_press_time = time.time()
        decision = "WAITING FOR VALIDATION..."

    elif key == ord('f'):
        lift_full = not lift_full

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()