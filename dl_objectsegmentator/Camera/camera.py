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

        self.frame_counter = 0
        self.gui_cfg = gui_cfg
        self.im = None
        self.buffer = []
        self.im_once_set = False
        self.im_segmented = None
        self.frame_to_process = None

        if hasattr(cam, 'hasproxy'):
            self.cam = cam
            self.source = 'stream_camera'
            self.im = self.getImage()
            self.im_height = self.im.height
            self.im_width = self.im.width

        elif isinstance(cam, int):
            self.cam = cv2.VideoCapture(cam)
            self.source = 'local_camera'
            if not self.cam.isOpened():
                print("%d is not a valid device index in this machine." % cam)
                raise SystemExit("Please check your camera id (hint: ls /dev)")
            self.im_width = int(self.cam.get(3))
            self.im_height = int(self.cam.get(4))

        elif isinstance(cam, str):
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
            self.detection = None
            self.count2 = 0
            # self.buffer = []

    def getImage(self):
        ''' Gets the image from the source and returns it resized and tagged with the frame number. '''

        if self.cam and self.source == 'stream_camera':
            im = self.cam.getImage()
            im = np.frombuffer(im.data, dtype=np.uint8)
        elif self.cam and (self.source == 'local_camera' or self.source == 'local_video'):
            _, frame = self.cam.read()
            im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        im = self.resizeImage(im)
        im = np.reshape(im, (540, 404, 3))
        self.frame_counter += 1
        cv2.putText(im, str(self.frame_counter), (340, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                    thickness=2)  # numerate frames
        return im

    def resizeImage(self, im):
        ''' Resizes the image. '''
        im_resized = np.reshape(im, (self.im_height, self.im_width, 3))
        im_resized = cv2.resize(im_resized, (404, 540), cv2.INTER_NEAREST)
        return im_resized

    def setGUI(self, gui):
        ''' Declares the GUI object. '''
        self.gui = gui

    def setNetwork(self, network, t_network):
        ''' Declares the Network object and its corresponding control thread. '''
        self.network = network
        self.t_network = t_network

    def setTracker(self, tracker):
        ''' Declares the Tracker object. '''
        self.tracker = tracker

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
        self.buffer = []  # new buffer
        self.tracker.activated = True  # tracker on

    def update(self):
        ''' Main function: controls the flow of the application. '''
        if self.cam:
            self.im = self.getImage()

        if self.gui_cfg == 'on':  # control GUI from camera thread

            if self.im is not None:

                im = QtGui.QImage(self.im, self.im.shape[1], self.im.shape[0],
                                  QtGui.QImage.Format_RGB888)
                im_scaled = im.scaled(self.gui.im_label.size())
                self.gui.im_label.setPixmap(QtGui.QPixmap.fromImage(im_scaled))  # show live images

                if self.gui.mode == 'continuous':

                    try:

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

                            #  show segmented image
                            im_segmented_qimage = QtGui.QImage(self.im_segmented.data, self.im_segmented.shape[1],
                                                               self.im_segmented.shape[0],
                                                               QtGui.QImage.Format_RGB888)
                            im_segmented_scaled = im_segmented_qimage.scaled(self.gui.im_combined_label.size())
                            self.gui.im_combined_label.setPixmap(QtGui.QPixmap.fromImage(im_segmented_scaled))
                            self.gui.im_segmented_label.setPixmap(QtGui.QPixmap.fromImage(im_segmented_scaled))

                            # tracking configuration
                            self.trackerConfiguration()

                        elif self.tracker.activated:  # get tracker output

                            im_detection = self.tracker.getOutputImage()
                            self.tracker.checkProgress()
                            if im_detection is not None:
                                im_detection = QtGui.QImage(im_detection.data, im_detection.shape[1],
                                                            im_detection.shape[0],
                                                            QtGui.QImage.Format_RGB888)
                                im_detection_scaled = im_detection.scaled(self.gui.im_segmented_label.size())
                                self.gui.im_combined_label.setPixmap(QtGui.QPixmap.fromImage(im_detection_scaled))
                                self.gui.im_tracked_label.setPixmap(QtGui.QPixmap.fromImage(im_detection_scaled))

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
                        im_segmented_qimage = QtGui.QImage(self.im_segmented.data, self.im_segmented.shape[1],
                                                           self.im_segmented.shape[0],
                                                           QtGui.QImage.Format_RGB888)
                        im_segmented_scaled = im_segmented_qimage.scaled(self.gui.im_segmented_label.size())
                        self.gui.im_combined_label.setPixmap(QtGui.QPixmap.fromImage(im_segmented_scaled))
                        self.gui.im_segmented_label.setPixmap(QtGui.QPixmap.fromImage(im_segmented_scaled))
                        self.im_once_set = False
                        self.buffer = self.buffer[self.frame_to_process:len(self.buffer)]

            else:
                self.frame_counter = 0

        else:  # gui off, writes results in .jpg <- review code!! change according to gui on

            im = self.getImage()

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

                        self.network.setInputImage(
                            self.buffer_cam[len(self.buffer_cam) - 1])  # segment last frame in buffer
                        self.network.toggleNetwork()  # network on
                        # segmentada
                        cv2.imwrite(str(self.frame_counter) + '.jpg', im_segmented)  # BGR

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
                            cv2.imwrite(str(self.frame_counter) + '.jpg', im_detection)  # BGR

            except AttributeError:
                pass
