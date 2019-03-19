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

import numpy as np
import cv2
from PyQt5 import QtGui


class Camera:
    def __init__(self, cam, gui_cfg):
        ''' Camera class gets images from live video. '''

        # initialize Camera instance attributes
        self.cam = None
        self.source = None
        self.im_height = None
        self.im_width = None
        self.gui_cfg = gui_cfg
        self.im = None
        self.buffer = []
        self.im_once_set = False
        self.im_segmented = None
        self.frame_counter = 0
        self.frame_to_process = None
        self.network_framework = None
        self.filename_offset = 0
        self.last_frames_video = False

        # image source: camera stream (ICE/ROS)
        if hasattr(cam, 'hasproxy'):
            self.cam = cam
            self.source = 'stream_camera'
            self.im = self.getImage()
            self.im_height = self.im.height  #ToDo: change to shape[0] and shape[1] when using ROS comm
            self.im_width = self.im.width

        # image source: local camera (OpenCV)
        elif isinstance(cam, int):
            self.cam = cv2.VideoCapture(cam)
            self.source = 'local_camera'
            if not self.cam.isOpened():
                print("%d is not a valid device index in this machine." % cam)
                raise SystemExit("Please check your camera id (hint: ls /dev)")
            self.im_width = int(self.cam.get(3))
            self.im_height = int(self.cam.get(4))

        # image source: local video (OpenCV)
        elif isinstance(cam, str):
            self.source = 'local_video'
            from os import path
            video_path = path.expanduser(cam)
            if not path.isfile(video_path):
                raise SystemExit('%s does not exists. Please check the path.' % video_path)
            self.cam = cv2.VideoCapture(video_path)
            if not self.cam.isOpened():
                print("%s is not a valid video path." % video_path)
            self.im_width = int(self.cam.get(3))
            self.im_height = int(self.cam.get(4))

        else:
            raise SystemExit("Interface camera not connected")

        print('Image size: {0}x{1} px'.format(
            self.im_width, self.im_height))

        if self.gui_cfg == 'off':
            print('GUI not set')

    def getImage(self):
        ''' Gets the image from the source and returns it resized and tagged with the frame number. '''

        im = None

        if self.cam and self.source == 'stream_camera':
            im = self.cam.getImage()
            im = np.frombuffer(im.data, dtype=np.uint8)
        elif self.cam and (self.source == 'local_camera' or self.source == 'local_video'):
            _, frame = self.cam.read()
            if frame is not None:
                im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im = self.resizeImage(im)
                im = np.reshape(im, (416, 416, 3))
                self.frame_counter += 1
                cv2.putText(im, str(self.frame_counter), (370, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                            thickness=2)  # numerate frames to debug, REMOVE in final version
        return im

    def resizeImage(self, im):
        ''' Resizes the image. '''
        im_resized = np.reshape(im, (self.im_height, self.im_width, 3))
        im_resized = cv2.resize(im_resized, (416, 416), cv2.INTER_NEAREST)
        return im_resized

    def setGUI(self, gui):
        ''' Declares the GUI object. '''
        self.gui = gui

    def setNetwork(self, network, t_network):
        ''' Declares the Network object and its corresponding control thread. '''
        self.network = network
        self.t_network = t_network
        self.network_framework = network.framework

    def setTracker(self, tracker):
        ''' Declares the Tracker object. '''
        self.tracker = tracker
        self.tracker.network_framework = self.network_framework  # indicates the neural network type to the tracker

    def toggleNetworkAndTracker(self):
        ''' Network and Tracker off. '''
        self.network.toggleNetwork()
        self.tracker.activated = False

    def trackerConfiguration(self):
        ''' Initializes tracker parameters. '''
        detection = self.network.getOutputDetection()
        label = self.network.getOutputLabel()
        color_list = self.network.getColorList()
        self.tracker.setInputDetection(detection, True)
        self.tracker.setInputLabel(label)
        self.tracker.setColorList(color_list)
        self.tracker.setBuffer(self.buffer)
        self.filename_offset = len(self.buffer) + 1
        self.buffer = []  # new buffer, reset buffer
        self.tracker.activated = True  # tracker on

    def update(self):
        ''' Main function: controls the flow of the application. '''
        if self.cam:
            self.im = self.getImage()

        if self.im is not None or len(self.buffer) > 0:

            if self.gui_cfg == 'on' and self.im is not None:  # control GUI from camera thread
                im = QtGui.QImage(self.im, self.im.shape[1], self.im.shape[0],
                                  QtGui.QImage.Format_RGB888)
                im_scaled = im.scaled(self.gui.im_label.size())
                self.gui.im_label.setPixmap(QtGui.QPixmap.fromImage(im_scaled))  # show live images

            if self.gui.mode == 'continuous':

                try:

                    if self.source == 'local_video' and self.im is not None:
                        self.buffer.append(self.im)  # allows processing last frames in buffer
                    elif self.source == 'local_camera' or self.source == 'stream_camera':
                        self.buffer.append(self.im)

                    if self.gui.count == 0:
                        self.network.setInputImage(self.im, self.frame_counter)  # segment first frame
                        self.frame_to_process = self.frame_counter
                        self.gui.count += 1

                    processed_frame = self.network.getProcessedFrame()

                    if processed_frame == self.frame_to_process:
                        self.im_segmented = self.network.getOutputImage()[0]

                    if not self.tracker.activated and not self.network.activated:  # segmentation
                        self.network.setInputImage(
                            self.buffer[len(self.buffer) - 1], self.frame_counter)  # segment last frame in buffer
                        self.frame_to_process = self.frame_counter
                        self.network.toggleNetwork()  # network on

                        if self.gui_cfg == 'on':
                            #  show segmented image
                            im_segmented_qimage = QtGui.QImage(self.im_segmented.data, self.im_segmented.shape[1],
                                                               self.im_segmented.shape[0],
                                                               QtGui.QImage.Format_RGB888)
                            im_segmented_scaled = im_segmented_qimage.scaled(self.gui.im_combined_label.size())
                            self.gui.im_combined_label.setPixmap(QtGui.QPixmap.fromImage(im_segmented_scaled))
                            self.gui.im_segmented_label.setPixmap(QtGui.QPixmap.fromImage(im_segmented_scaled))
                        else:  # gui off, save image
                            cv2.imwrite(str(self.frame_counter - self.filename_offset) + '.jpg', self.im_segmented)  # in BGR

                        # tracking configuration
                        self.trackerConfiguration()

                    elif self.tracker.activated:  # get tracker output
                        im_detection = self.tracker.getOutputImage()
                        self.tracker.checkProgress()
                        if im_detection is not None:
                            if self.gui_cfg == 'on':
                                im_detection = QtGui.QImage(im_detection.data, im_detection.shape[1],
                                                            im_detection.shape[0],
                                                            QtGui.QImage.Format_RGB888)
                                im_detection_scaled = im_detection.scaled(self.gui.im_segmented_label.size())
                                self.gui.im_combined_label.setPixmap(QtGui.QPixmap.fromImage(im_detection_scaled))
                                self.gui.im_tracked_label.setPixmap(QtGui.QPixmap.fromImage(im_detection_scaled))
                            else:  # gui off, save image
                                saved_image = cv2.cvtColor(im_detection, cv2.COLOR_BGR2RGB)
                                cv2.imwrite(str(self.frame_counter - self.filename_offset) + '.jpg', saved_image)  # in RGB

                except AttributeError:
                    pass

            else:  # 'once' mode

                if not self.im_once_set:  # initial sets
                    self.network.setInputImage(self.im, self.frame_counter)
                    self.frame_to_process = self.frame_counter
                    self.im_once_set = True
                if not self.network.getOutputImage()[1]:  # get processed frame
                    processed_frame = self.network.getProcessedFrame()
                    if processed_frame == self.frame_to_process:
                        self.im_segmented = self.network.getOutputImage()[0]
                if np.any(self.im_segmented.data):  # set segmented frame
                    if self.gui_cfg == 'on':
                        im_segmented_qimage = QtGui.QImage(self.im_segmented.data, self.im_segmented.shape[1],
                                                           self.im_segmented.shape[0],
                                                           QtGui.QImage.Format_RGB888)
                        im_segmented_scaled = im_segmented_qimage.scaled(self.gui.im_segmented_label.size())
                        self.gui.im_combined_label.setPixmap(QtGui.QPixmap.fromImage(im_segmented_scaled))
                        self.gui.im_segmented_label.setPixmap(QtGui.QPixmap.fromImage(im_segmented_scaled))
                    else:  # gui off, save image
                        saved_image = cv2.cvtColor(self.im_segmented, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(str(self.frame_counter - self.filename_offset) + '.jpg', saved_image)
                    self.im_once_set = False
                    self.buffer = self.buffer[self.frame_to_process:len(self.buffer)]

        elif self.source == 'local_video':  # allows processing last frames in buffer
            if not self.last_frames_video:
                self.frame_counter -= self.filename_offset  # avoids overwriting filename in last frames
                self.last_frames_video = not self.last_frames_video
            elif self.last_frames_video:
                self.frame_counter += 1
            im_detection = self.tracker.getOutputImage()  #ToDo: put this in function trackerOut?
            self.tracker.checkProgress()
            if im_detection is not None:
                if self.gui_cfg == 'on':
                    im_detection = QtGui.QImage(im_detection.data, im_detection.shape[1],
                                                im_detection.shape[0],
                                                QtGui.QImage.Format_RGB888)
                    im_detection_scaled = im_detection.scaled(self.gui.im_segmented_label.size())
                    self.gui.im_combined_label.setPixmap(QtGui.QPixmap.fromImage(im_detection_scaled))
                    self.gui.im_tracked_label.setPixmap(QtGui.QPixmap.fromImage(im_detection_scaled))
                else:  # gui off, save image
                    saved_image = cv2.cvtColor(im_detection, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(str(self.frame_counter) + '.jpg', saved_image)  # in RGB
