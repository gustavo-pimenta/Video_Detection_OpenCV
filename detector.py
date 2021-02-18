#########################################################################################
#   Name: Video Detection with OpenCV                                                   #
#   Version : 1.0                                                                       #
#                                                                                       #
#   Made by: Gustavo Pimenta - Computer Engineering / Game Developer / Python Coder     #
#            gustavopimenta.gp@gmail.com                                                #
#########################################################################################

import cv2

# load the cascades for image detection
front_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
prof_face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cat_face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

color_image = 0 # initial var for colored image frame
gray_image = 0 # initial var for grayscale image frame
out_image = 0 # initial var for the image that will be showed to user

# var to control what will be detected by the cascade system
FACES = True
EYES = True
CAT_FACES = False
BODY = False

class video_input():

    def __init__(self): 
        # Capture video from webcam
        self.video = cv2.VideoCapture(0)

    def update(self):
        global gray_image, color_image, out_image

        # Read the video info
        conection, frame = self.video.read()

        # actual video frame with colors
        color_image = frame

        # image that will be showed to user
        out_image = frame

        # actual video frame in grayscale
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def image_write():
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale = 1
    fontColor = (255,255,255)
    lineType = 2

    #cv2.putText(img,'Hello World!', 
    #     bottomLeftCornerOfText, 
    #     font, 
    #     fontScale,
    #     fontColor,
    #     lineType)

def cascade_detect():
    global FACES, EYES, CAT_FACES, BODY, gray_image, out_image

    # Detect images
    if FACES==True: # to detect human faces by frontal and profile

        frontal_faces = front_face_cascade.detectMultiScale(gray_image, 1.1, 5) # detect frontal faces
        profile_faces = prof_face_cascade.detectMultiScale(gray_image, 1.3, 3) # detect profile faces
        
        faces=[] # list with all the faces
        for i in frontal_faces: faces.append(i) # add frontal faces in the list
        for i in profile_faces: faces.append(i) # add profile faces in the list
        
        for (x, y, w, h) in faces:
            cv2.rectangle(out_image, (x, y), (x+w, y+h), (255, 0, 0), 2) # Draw the rectangle around each face

            if EYES==True: # to detect eyes
                gray_face_image = gray_image[y:y+h, x:x+w] # get a image of the actual face
                eyes = eye_cascade.detectMultiScale(gray_face_image) # detect eyes in this face
                for (ex, ey, ew, eh) in eyes:
                    cv2.circle(out_image, (int(x+ex+(ew/2)),int(y+ey+(eh/2))), int(ew/4), (0, 255, 0)) # draw a circle around each eye

    if CAT_FACES==True: # to detect cat faces
        cat_faces = cat_face_cascade.detectMultiScale(gray_image, 1.3, 5) # detect cat faces
        for (x, y, w, h) in cat_faces:
            cv2.rectangle(out_image, (x, y), (x+w, y+h), (0, 0, 255), 2) # draw a red rectangle in the cat face
    
    if BODY==True: # to detect bodys
        bodys = body_cascade.detectMultiScale(gray_image, 1.3, 5) # detect bodys
        for (x, y, w, h) in bodys:
            cv2.rectangle(out_image, (x, y), (x+w, y+h), (0, 0, 255), 2)


video = video_input()

while True:
    video.update()
    cascade_detect()

    # Display
    cv2.imshow('video detection by gustavo pimenta', out_image)

    # Stop if ESC is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
        
# Release the VideoCapture object
video.release()
