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

import sys
import traceback
import threading

import numpy as np
import comm
import config


class Camera:


    def __init__(self):
        ''' Camera class gets images from live video
        in order to segment the objects in the image.
        '''

        # Creation of the camera through the comm-ICE proxy.
        try:
            cfg = config.load(sys.argv[1])
        except IndexError:
            raise SystemExit('Missing YML file. Usage: python2 objectsegmentator.py objectsegmentator.yml')

        jdrc = comm.init(cfg, 'ObjectSegmentator')

        self.lock = threading.Lock()

        try:
            self.cam = jdrc.getCameraClient('ObjectSegmentator.Camera')
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
            #im = np.zeros((self.im_height, self.im_width, 3), np.uint8)
            im = np.frombuffer(self.im.data, dtype=np.uint8)
            im.shape = self.im_height, self.im_width, 3

            self.lock.release()

            return im

    def update(self):
        ''' Updates the camera every time the thread changes. '''
        if self.cam:
            self.lock.acquire()

            self.im = self.cam.getImage()
            self.im_height = self.im.height
            self.im_width = self.im.width

            self.lock.release()