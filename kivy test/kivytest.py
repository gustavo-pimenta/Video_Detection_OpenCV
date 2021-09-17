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

        # layout.add_widget(self.label)
        # self.label = MDLabel()

        layout.add_widget(self.image)
        

        self.save_img_button = MDRaisedButton(
            text="CLICK HERE",
            pos_hint={'center_x': .5, 'center_y': .5},
            size_hint=(None, None))
        self.save_img_button.bind(on_press = self.update_video_capture)
        layout.add_widget(self.save_img_button)

        self.video_capture = cv2.VideoCapture(0) # start video capture
        Clock.schedule_interval(self.controller, 1.0/FPS) # interval to call update function
        
        return layout

    def update_video_capture(self):
        
        conection, self.frame = self.video_capture.read() # Read the video info
        # self.color_image = self.frame # actual video frame with colors

    def print_image(self): # show the mais image in the screen
        
        buffer = cv2.flip(self.frame, 0).tostring()
        texture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

    def grayscale(self):
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY) # turn image into grayscale
        self.frame = cv2.flip(self.frame, 0).tobytes()
    
    def duplicate_image(self): # make a duplicate side by side version of an image

        self.frame = np.hstack((self.frame, self.frame))
        
    
    def controller(self, *args): # controll the functions in use 

        self.update_video_capture()
        # self.grayscale()
        self.duplicate_image()
        self.print_image()
        

    
    

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