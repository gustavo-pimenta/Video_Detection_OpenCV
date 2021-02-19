#########################################################################################
#   Name: Video Detection with OpenCV                                                   #
#   Version : 1.0                                                                       #
#                                                                                       #
#   Made by: Gustavo Pimenta - Computer Engineering / Game Developer / Python Coder     #
#            gustavopimenta.gp@gmail.com                                                #
#########################################################################################

import cv2
import numpy as np
import math

# load the cascades for image detection
front_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
prof_face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cat_face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# control what will be detected by the cascade system
FACES = True
EYES = False
CAT_FACES = False
BODY = False

# informations to the skin color detector
ROI = True # define if the region of interest will be used 
lower_skin = np.array([0,20,70], dtype=np.uint8) # define lower skin color (also can use [0, 48, 80])
upper_skin = np.array([20,255,255], dtype=np.uint8) # define upper skin color

# class to recieve and update the video input
class video_input(): 

    def __init__(self): 
        # capture video from webcam
        self.video = cv2.VideoCapture(0)

    def update(self):
        global gray_image, color_image, out_image

        # Read the video info
        conection, self.frame = self.video.read()

        # actual video frame with colors
        self.color_image = self.frame

        # actual video frame in grayscale
        self.gray_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

    def hand_detector(self):
        global lower_skin, upper_skin

        # define the region of interest
        cv2.rectangle(self.frame, (100, 100), (300, 300), (0, 255, ), 2)
        roi_image = self.frame[100:300, 100:300]
        
        # apply gaussian blur on the ROI
        blur = cv2.GaussianBlur(roi_image, (3, 3), 0)

        # convert the ROI image from BGR to HSV color 
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # create a binary image where white will be skin colors and rest is black
        mask2 = cv2.inRange(hsv, lower_skin, upper_skin)

        # create the kernel for morphological transformation
        kernel = np.ones((5, 5))

        # apply morphological transformations to filter out the background noise
        dilation = cv2.dilate(mask2, kernel, iterations=1)
        erosion = cv2.erode(dilation, kernel, iterations=1)

        # apply gaussian blur and threshold
        filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
        ret, thresh = cv2.threshold(filtered, 127, 255, 0)

        # find contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        try:
            # find the max area of the contour
            contour = max(contours, key=lambda x: cv2.contourArea(x))

            # draw a rectangle around the contour
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(roi_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

            # find convex hull
            hull = cv2.convexHull(contour)

            # draw the contour
            drawing = np.zeros(roi_image.shape, np.uint8)
            
            cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
            cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

            # Find convexity defects
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull)
            count_defects = 0

            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])

                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

                # if the angle > 90 draw a circle at the far point
                if angle <= 90:
                    count_defects += 1
                    cv2.circle(roi_image, far, 1, [0, 0, 255], -1)

                cv2.line(roi_image, start, end, [0, 255, 0], 2)
        
            # save the number of fingers
            if count_defects == 0:
                self.fingers = 1
            elif count_defects == 1:
                self.fingers = 2
            elif count_defects == 2:
                self.fingers = 3
            elif count_defects == 3:
                self.fingers = 4
            elif count_defects == 4:
                self.fingers = 5
            else: self.fingers = 0

            # create and save images to be showed 
            self.frame = self.frame
            self.roi_image = np.hstack((drawing, roi_image))
            self.roi_blackwhite = thresh

        except: print('DETECT FINGERS ERROR')
        
    def show_images(self):
        try:
            cv2.imshow('frame',self.frame)
            cv2.imshow('roi image',self.roi_image)
            cv2.imshow('roi black white',self.roi_blackwhite)
        except: pass
        
    def image_write(self, text, position=(100,100), color=(255,255,255), scale=1, linetype=2):
        font = cv2.FONT_HERSHEY_SIMPLEX

        image = self.frame

        cv2.putText(
            image,
            text, 
            position, 
            font, 
            scale,
            color,
            linetype
        )

    def cascade_detector(self):
        global FACES, EYES, CAT_FACES, BODY

        # Detect images
        if FACES==True: # to detect human faces by frontal and profile

            frontal_faces = front_face_cascade.detectMultiScale(self.gray_image, 1.1, 5) # detect frontal faces
            profile_faces = prof_face_cascade.detectMultiScale(self.gray_image, 1.3, 3) # detect profile faces
            
            faces=[] # list with all the faces
            for i in frontal_faces: faces.append(i) # add frontal faces in the list
            for i in profile_faces: faces.append(i) # add profile faces in the list
            
            for (x, y, w, h) in faces:
                cv2.rectangle(self.frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Draw the rectangle around each face

                if EYES==True: # to detect eyes
                    gray_face_image = self.gray_image[y:y+h, x:x+w] # get a image of the actual face
                    eyes = eye_cascade.detectMultiScale(gray_face_image) # detect eyes in this face
                    for (ex, ey, ew, eh) in eyes:
                        cv2.circle(self.frame, (int(x+ex+(ew/2)),int(y+ey+(eh/2))), int(ew/4), (0, 255, 0)) # draw a circle around each eye

        if CAT_FACES==True: # to detect cat faces
            cat_faces = cat_face_cascade.detectMultiScale(self.gray_image, 1.3, 5) # detect cat faces
            for (x, y, w, h) in cat_faces:
                cv2.rectangle(self.frame, (x, y), (x+w, y+h), (0, 0, 255), 2) # draw a red rectangle in the cat face
        
        if BODY==True: # to detect bodys
            bodys = body_cascade.detectMultiScale(self.gray_image, 1.3, 5) # detect bodys
            for (x, y, w, h) in bodys:
                cv2.rectangle(self.frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

video = video_input() # capture video from webcam

while True:
    video.update()
    
    video.cascade_detector()
    if ROI==True: video.hand_detector()

    video.show_images()

    # break when ESC is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
        
# release the video capture object
video.video.release()
cv2.destroyAllWindows()
