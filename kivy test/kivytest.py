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
from kivymd.app import MDApp
from kivymd.uix.label import MDLabel
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.core.window import Window

FPS = 60.0

class MainApp(MDApp):

    def build(self):

        layout = MDBoxLayout(orientation='vertical')

        self.image = Image() # create Image object
        layout.add_widget(self.image) # add the Image Widget on screen
       
        self.video_capture = VideoCapture()

        Clock.schedule_interval(self.controller, 1.0/FPS) # interval to call update function
        
        return layout
        
    def controller(self, *args): # controll the functions in use 

        self.video_capture.update()
        self.video_capture.duplicate_image()
        
        self.image_on_screen()
    
    def image_on_screen(self): # show the mais image in the screen
        
        self.frame = self.video_capture.get_frame()

        buffer = cv2.flip(self.frame, 0).tostring()
        texture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

class VideoCapture():

    def __init__(self): 
        self.video_capture = cv2.VideoCapture(0) # start video capture

    def update(self):
        conection, self.frame = self.video_capture.read() # Read the video info
        # self.color_image = self.frame # actual video frame with colors

    def get_frame(self):
        return self.frame

    def grayscale(self):
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY) # turn image into grayscale
        self.frame = cv2.flip(self.frame, 0).tobytes()
    
    def duplicate_image(self): # make a duplicate side by side version of an image
        self.frame = np.hstack((self.frame, self.frame))
    
        
MainApp().run()






# while True:

#     video.update()
#     video.show_images()

#     # break when ESC is pressed
#     k = cv2.waitKey(30) & 0xff
#     if k==27:
#         break
        
# release the video capture object
# video.video.release()
cv2.destroyAllWindows()