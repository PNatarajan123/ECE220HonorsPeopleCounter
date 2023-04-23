import cv2
import time

# capture frames from a video
cap = cv2.VideoCapture('people.mp4')

#haarcascade
pedestrian_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")

people_entered = 0

#width and height of frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#center of frame
center_line = frame_width // 2

threshold = 5

#determine if person cross center
person_crossed_center = False

#delay to avoid double counting
delay = 0.6

#track last detection
last_detection_time = time.time()

#Output Video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 25, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    pedestrians = pedestrian_cascade.detectMultiScale(gray, 1.1, 1)

    # Draw a rectangle in each pedestrian
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if x < center_line - threshold and x + w > center_line:
            if not person_crossed_center and time.time() - last_detection_time >= delay:
                people_entered += 1
                person_crossed_center = True
                last_detection_time = time.time()
        else:
            person_crossed_center = False

    cv2.line(frame, (center_line, 0), (center_line, frame_height), (0, 0, 255), 2)

    cv2.putText(frame, f'People in the room: {people_entered}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Pedestrian Detection', frame)

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
