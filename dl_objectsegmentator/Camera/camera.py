#
# Created on Feb, 2018
#
# @author: alexandre2r
#
# Class which abstracts a Camera from a proxy (created by ICE/ROS),
# and provides the methods to keep it constantly updated. It delivers it to the neural network,
# which returns the segmented objects.
#
#
# Based on @naxvm code:
# https://github.com/JdeRobot/dl-objectdetector
#

import traceback
import threading

import numpy as np
import cv2


class Camera:

    def __init__(self, cam):
        ''' Camera class gets images from live video
        in order to segment the objects in the image.
        '''

        self.cam = cam
        self.lock = threading.Lock()

        try:
            if self.cam.hasproxy():
                self.im = self.cam.getImage()
                self.im_height = self.im.height
                self.im_width = self.im.width
                print('Image size: {0}x{1} px'.format(
                        self.im_width, self.im_height))
            else:
                print("Interface camera not connected")

        except:
            traceback.print_exc()
            exit()

    def getImage(self):
        ''' Gets the image from the webcam and returns the original
        image that we're going to use to make the segmentation.
        '''
        if self.cam:
            self.lock.acquire()
            im = np.frombuffer(self.im.data, dtype=np.uint8)
            im = self.transformImage(im)
            im = np.reshape(im, (540, 404, 3))

            self.lock.release()

            return im

    def transformImage(self, im):
        im_resized = np.reshape(im, (self.im_height, self.im_width, 3))
        im_resized = cv2.resize(im_resized, (404, 540))
        return im_resized

    def update(self):
        ''' Updates the camera every time the thread changes. '''
        if self.cam:
            self.lock.acquire()

            self.im = self.cam.getImage()
            self.im_height = self.im.height
            self.im_width = self.im.width

            self.lock.release()