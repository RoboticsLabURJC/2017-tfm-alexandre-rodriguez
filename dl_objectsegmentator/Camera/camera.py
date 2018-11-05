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

import threading
import numpy as np
import cv2
from PyQt5 import QtGui


class Camera:

    def __init__(self, cam, gui_cfg):
        ''' Camera class gets images from live video. '''

        self.cam = cam
        self.count = 0
        self.gui_cfg = gui_cfg
        self.lock = threading.Lock()
        self.im = None
        self.buffer = []
        self.im_once_set = False
        self.im_segmented = None
        self.frame_to_process = None

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

        if self.gui_cfg == 'on':  # control GUI from camera thread

            if self.im is not None:

                im_prev = self.im

                im = QtGui.QImage(im_prev, im_prev.shape[1], im_prev.shape[0],
                              QtGui.QImage.Format_RGB888)
                im_scaled = im.scaled(self.gui.im_label.size())
                self.gui.im_label.setPixmap(QtGui.QPixmap.fromImage(im_scaled))  # show live images

                if self.gui.mode == 'continuous':

                    try:

                        self.buffer.append(self.im)

                        if self.gui.count == 0:
                            self.network.setInputImage(im_prev, self.count)  # segment first frame
                            self.frame_to_process = self.count
                            self.gui.count += 1

                        processed_frame = self.network.getProcessedFrame()
                        if processed_frame == self.frame_to_process:
                            self.im_segmented = self.network.getOutputImage()[0]

                        #self.gui.network_not_finished = self.network.getOutputImage()[1]

                        if not self.tracker.activated and not self.network.activated:  # segmentation

                            self.network.setInputImage(
                                self.buffer[len(self.buffer) - 1], self.count)  # segment last frame in buffer
                            self.frame_to_process = self.count
                            self.network.toggleNetwork()  # network on
                            self.gui.last_segmented = self.im_segmented

                            #  show segmented image
                            im_segmented_qimage = QtGui.QImage(self.im_segmented.data, self.im_segmented.shape[1],
                                                               self.im_segmented.shape[0],
                                                               QtGui.QImage.Format_RGB888)
                            im_segmented_scaled = im_segmented_qimage.scaled(self.gui.im_combined_label.size())
                            self.gui.im_combined_label.setPixmap(QtGui.QPixmap.fromImage(im_segmented_scaled))
                            self.gui.im_segmented_label.setPixmap(QtGui.QPixmap.fromImage(im_segmented_scaled))

                            # tracking configuration
                            detection = self.network.getOutputDetection()
                            label = self.network.getOutputLabel()
                            color_list = self.network.getColorList()
                            self.tracker.setInputDetection(detection, True)
                            self.tracker.setInputLabel(label)
                            self.tracker.setColorList(color_list)
                            self.tracker.setBuffer(self.buffer)
                            self.buffer = []  # new buffer
                            self.tracker.activated = True  # tracker on

                        elif self.tracker.activated:  # tracker output

                            im_detection = self.tracker.getOutputImage()
                            self.tracker.checkProgress()
                            if im_detection is not None:
                                im_detection = QtGui.QImage(im_detection.data, im_detection.shape[1], im_detection.shape[0],
                                                                QtGui.QImage.Format_RGB888)
                                im_detection_scaled = im_detection.scaled(self.gui.im_segmented_label.size())
                                self.gui.im_combined_label.setPixmap(QtGui.QPixmap.fromImage(im_detection_scaled))
                                self.gui.im_tracked_label.setPixmap(QtGui.QPixmap.fromImage(im_detection_scaled))

                        # elif not self.tracker.activated and self.network.activated:  # tracker ends but no result from network -> discard frame
                        #     #print('descartei!')
                        #     no_track = self.buffer.pop(0)
                        #     im_detection = QtGui.QImage(no_track.data, no_track.shape[1], no_track.shape[0],
                        #                                 QtGui.QImage.Format_RGB888)
                        #     im_detection_scaled = im_detection.scaled(self.gui.im_segmented_label.size())
                        #     self.gui.im_combined_label.setPixmap(QtGui.QPixmap.fromImage(im_detection_scaled))  # show discarded frames

                    except AttributeError:
                        pass

                else:  # 'once' mode

                    if not self.im_once_set:  # initial sets
                        self.network.setInputImage(im_prev, self.count)
                        self.frame_to_process = self.count
                        self.im_once_set = True
                    if not self.network.getOutputImage()[1]:  # get processed frame
                        processed_frame = self.network.getProcessedFrame()
                        if processed_frame == self.frame_to_process:
                            self.im_segmented = self.network.getOutputImage()[0]
                    if np.any(self.im_segmented.data):  # set segmented frame
                        im_segmented_qimage = QtGui.QImage(self.im_segmented.data, self.im_segmented.shape[1], self.im_segmented.shape[0],
                                                           QtGui.QImage.Format_RGB888)
                        im_segmented_scaled = im_segmented_qimage.scaled(self.gui.im_segmented_label.size())
                        self.gui.im_combined_label.setPixmap(QtGui.QPixmap.fromImage(im_segmented_scaled))
                        self.gui.im_segmented_label.setPixmap(QtGui.QPixmap.fromImage(im_segmented_scaled))
                        self.im_once_set = False
                        self.buffer = self.buffer[self.frame_to_process:len(self.buffer)]



            else:
                self.count = 0

        else:  # gui off, writes results in .jpg <- review code!!

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