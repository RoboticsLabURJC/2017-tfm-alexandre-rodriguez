#
# Created on Feb, 2018
#
# @author: alexandre2r
#
# Class which abstracts a Camera from a proxy (created by ICE/ROS),
# and provides the methods to keep it constantly updated.
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

    def __init__(self, cam, gui_cfg):
        ''' Camera class gets images from live video. '''

        self.cam = cam
        self.count = 0
        self.gui_cfg = gui_cfg
        self.lock = threading.Lock()
        self.im = None
        self.buffer_cam = []

        if self.cam.hasproxy():
            original_image = self.cam.getImage()
            self.im_height = original_image.height
            self.im_width = original_image.width

            print('Image size: {0}x{1} px'.format(
                self.im_width, self.im_height))
        else:
            raise SystemExit("Interface camera not connected")

        if self.gui_cfg == 'off':
            print('GUI not set')
            self.detection = None
            self.count2 = 0
            #self.buffer = []

    def getImage(self):
        ''' Gets the image from the webcam and returns the original image. '''

        if self.cam:
            im = self.cam.getImage()
            im = np.frombuffer(im.data, dtype=np.uint8)
            im = self.transformImage(im)
            im = np.reshape(im, (540, 404, 3))
            self.count += 1
            cv2.putText(im, str(self.count), (340, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=2)  # numerate frames
            return im

    def transformImage(self, im):
        im_resized = np.reshape(im, (self.im_height, self.im_width, 3))
        im_resized = cv2.resize(im_resized, (404, 540))
        return im_resized

    def setGUI(self, gui):
        self.gui = gui

    def setNetwork(self, network, t_network):
        ''' Declares the Network object and its corresponding control thread. '''
        self.network = network
        self.t_network = t_network

    def setTracker(self, tracker):
        self.tracker = tracker

    def toggleNetwork(self):
        #self.toggleMode() sobra?
        self.network.toggleNetwork()
        self.tracker.activated = False

    def update(self):
        ''' Updates the camera every time the thread changes. '''
        if self.cam:
            self.lock.acquire()
            self.im = self.getImage()
            self.lock.release()

        if self.gui_cfg == 'on':
            self.buffer_cam.append(self.im)

        else:  # gui off

            im = self.getImage() # use self.im directly ?

            try:

                if self.count2 == 0:
                    self.network.setInputImage(im)  # segment first frame
                    self.count2 += 1

                im_segmented = self.network.getOutputImage()[0]
                zeros = self.network.getOutputImage()[1]

                if zeros:  # first frames

                    self.buffer_cam.append(im)

                else:

                    self.buffer_cam.append(im)

                    if not self.tracker.activated and not self.network.activated:  # segmentation

                        self.network.setInputImage(self.buffer_cam[len(self.buffer_cam) - 1])  # segment last frame in buffer
                        self.network.toggleNetwork()  # network on
                        # segmentada
                        cv2.imwrite(str(self.count) + '.jpg', im_segmented)  # BGR

                        # tracking configuration
                        detection = self.network.getOutputDetection()
                        label = self.network.getOutputLabel()
                        color_list = self.network.getColorList()
                        self.tracker.setInputDetection(detection, True)
                        self.tracker.setInputLabel(label)
                        self.tracker.setColorList(color_list)
                        self.tracker.setBuffer(self.buffer_cam)
                        self.buffer_cam = []  # new buffer
                        self.tracker.activated = True  # tracker on

                    elif self.tracker.activated:  # tracking output
                        im_detection = self.tracker.getOutputImage()
                        self.tracker.checkProgress()
                        if im_detection is not None:
                            cv2.imwrite(str(self.count) + '.jpg', im_detection)  # BGR

            except AttributeError:
                pass