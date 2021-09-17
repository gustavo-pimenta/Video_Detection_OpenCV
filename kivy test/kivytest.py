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


FPS = 60.0

class MainApp(MDApp):

    def build(self):

        layout = MDBoxLayout(orientation='vertical')
        self.image = Image()
        self.label = MDLabel()
        layout.add_widget(self.image)
        layout.add_widget(self.label)

        self.save_img_button = MDRaisedButton(
            text="CLICK HERE",
            pos_hint={'center_x': .5, 'center_y': .5},
            size_hint=(None, None))
        self.save_img_button.bind(on_press = self.show_images)
        layout.add_widget(self.save_img_button)

        self.video_capture = cv2.VideoCapture(0) # start video capture
        Clock.schedule_interval(self.update_video_capture, 1.0/FPS) # interval to call update function
        
        return layout

    def update_video_capture(self, *args):
        
        conection, frame = self.video_capture.read() # Read the video info

        self.color_image = frame # actual video frame with colors

        

        buffer = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

    def grayscale(image):

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # turn image into grayscale
        return gray_image

    def show_images(self, *args):
        try:
            cv2.imshow('frame',self.frame)
            cv2.imshow('roi image',self.roi_image)
            cv2.imshow('roi black white',self.roi_blackwhite)
        except: pass

    
    

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