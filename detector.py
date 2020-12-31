import cv2

# Load the cascade for database
front_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cat_face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')

# Capture video from webcam
video = cv2.VideoCapture(0)

while True:
    # Read the video info
    conection, frame = video.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = front_face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 2)
    cat_faces = cat_face_cascade.detectMultiScale(gray, 1.3, 3)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    for (x, y, w, h) in eyes:
        # cv2.circle(frame, (int((x+w)/2),int((y+h)/2)), int((w+h)/2), (0, 255, 0))

        cv2.circle(frame, (int(x+(w/2)),int(y+(h/2))), int(w/4), (0, 255, 0))

    for (x, y, w, h) in cat_faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display
    cv2.imshow('face detection by gustavo pimenta', frame)

    # Stop if ESC is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
        
# Release the VideoCapture object
video.release()
