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

color_image = 0 # initial var for colored image frame
gray_image = 0 # initial var for grayscale image frame
out_image = 0 # initial var for the image that will be showed to user

# control what will be detected by the cascade system
FACES = True
EYES = True
CAT_FACES = False
BODY = False

# control what will be detected by colors
ROI = True # define if the region of interest will be used (if false, the ROI will be the full image)
lower_skin = np.array([0,20,70], dtype=np.uint8) # define lower skin color
lower_skin = np.array([0, 48, 80], dtype=np.uint8) # define lower skin color
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
        color_image = self.frame

        # image that will be showed to user
        out_image = self.frame

        # actual video frame in grayscale
        gray_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        

    def hand_detector(self): 
        global lower_skin, upper_skin, out_image

        # define the region of interest
        self.roi = self.frame[100:300, 100:300]

        # draw a rectangle around the ROI
        cv2.rectangle(out_image,(100,100),(300,300),(0,0,255),3)

        # blur the ROI image
        blur = cv2.GaussianBlur(self.roi, (3, 3), 0)

        # convert the ROI image from BGR to HSV color 
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Create a binary image with where white will be skin colors and rest is black
        mask2 = cv2.inRange(hsv, lower_skin, upper_skin)

        # kernel for morphological transformation
        kernel = np.ones((5, 5))

        # Apply morphological transformations to filter out the background noise
        dilation = cv2.dilate(mask2, kernel, iterations=1)
        erosion = cv2.erode(dilation, kernel, iterations=1)

        # Apply Gaussian Blur and Threshold
        filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
        ret, thresh = cv2.threshold(filtered, 127, 255, 0)

        
        cv2.imshow("Thresholded Hemal", thresh)

        # Find contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        try:
        
            # Find contour with maximum area
            contour = max(contours, key=lambda x: cv2.contourArea(x))

            
            

            # Find convex hull
            hull = cv2.convexHull(contour)

            # Draw contour
            drawing = np.zeros(out_image.shape, np.uint8)
            
            cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
            cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

            # Find convexity defects
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull)

            # Use cosine rule to find angle of the far point from the start and end point i.e. the convex points (the finger
            # tips) for all defects
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

                # if angle > 90 draw a circle at the far point
                if angle <= 90:
                    count_defects += 1
                    cv2.circle(out_image, far, 1, [0, 0, 255], -1)

                cv2.line(out_image, start, end, [0, 255, 0], 2)

            # Print number of fingers
            if count_defects == 0:
                cv2.putText(out_image, "ONE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
            elif count_defects == 1:
                cv2.putText(out_image, "TWO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
            elif count_defects == 2:
                cv2.putText(out_image, "THREE", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
            elif count_defects == 3:
                cv2.putText(out_image, "FOUR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
            elif count_defects == 4:
                cv2.putText(out_image, "FIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
            else:
                pass
        except:
            pass




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

video = video_input() # capture video from webcam

while True:
    video.update()
    cascade_detect()
    if ROI==True: video.hand_detector()

    # display the image on screen
    cv2.imshow('video detection by gustavo pimenta', out_image)

    # Stop if ESC is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
        
# release the video capture object
video.release()
