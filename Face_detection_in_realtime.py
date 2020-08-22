import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

cap= cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    # Find all the faces in the current frame of video
    gray_scaled = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    face_locations = face_cascade.detectMultiScale(gray_scaled)

    for top, right, bottom, left in face_locations:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0,255), 2)
    # Display the resulting image
    cv2.imshow('Video', frame)


    # Wait for Enter key to stop
    if cv2.waitKey(25) == 13:
        break

cap.release()
cv2.destroyAllWindows()
