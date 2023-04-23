import cv2
import time

# capture frames from a video
cap = cv2.VideoCapture('people.mp4')

# Trained XML classifiers describes some features of some object we want to detect
pedestrian_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")

# initialize variables to keep track of people entering the room
people_entered = 0

# get the width and height of the input video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# define the center line of the frame
center_line = frame_width // 2

# define the threshold about the center
threshold = 5

# define a flag to keep track of whether a person has already crossed the center line or not
person_crossed_center = False

# define a delay (in seconds) to avoid double counting
delay = 0.6

# define a variable to keep track of the time of the last detection
last_detection_time = time.time()

# loop runs if capturing has been initialized.
while cap.isOpened():
    # reads frames from a video
    ret, frame = cap.read()
    if not ret:
        break

    # convert to gray scale of each frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect pedestrians of different sizes in the input image
    pedestrians = pedestrian_cascade.detectMultiScale(gray, 1.1, 1)

    # To draw a rectangle in each pedestrian
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # check if the person passes through the center line with the threshold
        if x < center_line - threshold and x + w > center_line:
            # increment people_entered only if the person has not already crossed the center line and the delay time has elapsed
            if not person_crossed_center and time.time() - last_detection_time >= delay:
                people_entered += 1
                person_crossed_center = True
                last_detection_time = time.time()
        else:
            # reset the flag if the person is not crossing the center line
            person_crossed_center = False

    # draw the center line on the frame
    cv2.line(frame, (center_line, 0), (center_line, frame_height), (0, 0, 255), 2)

    # Display the number of people in the room
    cv2.putText(frame, f'People in the room: {people_entered}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display frames in a window
    cv2.imshow('People Detection', frame)

    # press 'q' to exit the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
