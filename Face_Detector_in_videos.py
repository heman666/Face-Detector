import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

video = cv2.VideoCapture("video.mp4")

while True:
    bool_ret,frame = video.read()

    # rgb_frame = frame[:,:,::-1]
    frame2 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame2 = cv2.resize(frame2,(0,0), fx=0.25, fy=0.25, interpolation = cv2.INTER_LINEAR)

    faces = face_cascade.detectMultiScale(frame2)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame2,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("Video Frame",frame2)

    if cv2.waitKey(25) == 13:
        break

video.release()
cv2.destroyAllWindows()
print("Code Executed Successfully")
