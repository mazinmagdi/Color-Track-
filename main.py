import cv2 
import numpy as np

def detect_and_draw_boxes(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    lower_red1 = np.array([0, 120, 120])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 120])
    upper_red2 = np.array([180, 255, 255])

    lower_green = np.array([40, 50, 50])
    upper_green = np.array([90, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 | mask_red2

    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    mask_combined = mask_red | mask_green

    contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            if np.any(mask_red[y:y+h, x:x+w]):
                color = (0, 0, 255)
            elif np.any(mask_green[y:y+h, x:x+w]):
                color = (255, 0, 0)
            else:
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    return frame

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open the camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture video.")
        break

    frame = cv2.resize(frame, (640, 480))
    output_frame = detect_and_draw_boxes(frame)
    cv2.imshow("Color Detection with Bounding Boxes", output_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
