import cv2

# the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Start video capture(0=default webcam)
cap = cv2.VideoCapture(0)

while True:
#Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

#Convert frame to grayscale    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#Draw rectangles around faces
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

#Show the frame
    cv2.imshow('Face Detection', frame)

#Exit on Pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()