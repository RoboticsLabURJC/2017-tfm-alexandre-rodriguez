#
# Created on Feb, 2018
#
# @author: alexandre2r
#
# Based on @nuriaoyaga code:
# https://github.com/RoboticsURJC-students/2016-tfg-nuria-oyaga/blob/
#     master/gui/gui.py
#

from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets
import numpy as np
import time
import cv2

class GUI(QtWidgets.QWidget):

    updGUI = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        ''' GUI class creates the GUI that we're going to use to
        preview the live video as well as the results of the segmentation.
        '''

        QtWidgets.QWidget.__init__(self, parent)
        self.setWindowTitle("Object Segmentator (Keras-based Mask R-CNN trained with a COCO database)")
        self.resize(1200, 1100)
        self.move(150, 50)
        self.updGUI.connect(self.update)

        # Original image labels
        self.im_label = QtWidgets.QLabel(self)
        self.im_label.resize(480, 320)
        self.im_label.move(70, 40)
        self.im_label.show()
        self.im_label_txt = QtWidgets.QLabel(self)
        self.im_label_txt.resize(50, 40)
        self.im_label_txt.move(290, 10)
        self.im_label_txt.setText('Input')
        self.im_label_txt.show()

        # Black image at the beginning
        zeros = np.zeros((480, 320), dtype=np.int32)
        zeros = QtGui.QImage(zeros.data, zeros.shape[1], zeros.shape[0],
                                    QtGui.QImage.Format_RGB888)
        zeros_scaled = zeros.scaled(self.im_label.size())

        # Combined image labels
        self.im_combined_label = QtWidgets.QLabel(self)
        self.im_combined_label.resize(480, 320)
        self.im_combined_label.move(610, 40)
        self.im_combined_label.setPixmap(QtGui.QPixmap.fromImage(zeros_scaled))
        self.im_combined_label.show()
        self.im_combined_label_txt = QtWidgets.QLabel(self)
        self.im_combined_label_txt.resize(90, 40)
        self.im_combined_label_txt.move(820, 10)
        self.im_combined_label_txt.setText('Combined')
        self.im_combined_label_txt.show()

        # Segmented image labels
        self.im_segmented_label = QtWidgets.QLabel(self)
        self.im_segmented_label.resize(480, 320)
        self.im_segmented_label.move(70, 410)
        self.im_segmented_label.setPixmap(QtGui.QPixmap.fromImage(zeros_scaled))
        self.im_segmented_label.show()
        self.im_segmented_label_txt = QtWidgets.QLabel(self)
        self.im_segmented_label_txt.resize(50, 40)
        self.im_segmented_label_txt.move(290, 380)
        self.im_segmented_label_txt.setText('Net')
        self.im_segmented_label_txt.show()

        # Tracked image labels
        self.im_tracked_label = QtWidgets.QLabel(self)
        self.im_tracked_label.resize(480, 320)
        self.im_tracked_label.move(610, 410)
        self.im_tracked_label.setPixmap(QtGui.QPixmap.fromImage(zeros_scaled))
        self.im_tracked_label.show()
        self.im_tracked_label_txt = QtWidgets.QLabel(self)
        self.im_tracked_label_txt.resize(50, 40)
        self.im_tracked_label_txt.move(830, 380)
        self.im_tracked_label_txt.setText('Tracker')
        self.im_tracked_label_txt.show()

        # Button for processing a single frame
        self.button_now = QtWidgets.QPushButton('Run now', self)
        self.button_now.move(1140, 60)
        self.button_now.clicked.connect(self.updateOnce)

        # Button for processing continuous frames
        self.button_continuous = QtWidgets.QPushButton('Run continuous', self)
        self.button_continuous.move(1120, 100)
        self.button_continuous.clicked.connect(self.toggleNetwork)
        self.button_continuous.setStyleSheet('QPushButton {color: green;}')

        self.mode = 'continuous'
        self.detection = None
        #self.last_segmented = None
        self.count = 0
        self.buffer = []
        self.network_not_finished = True
        #self.last_buf = []

    def setCamera(self, cam):
        ''' Declares the Camera object '''
        self.cam = cam

    def setNetwork(self, network, t_network):
        ''' Declares the Network object and its corresponding control thread. '''
        self.network = network
        self.t_network = t_network

    def setTracker(self, tracker):
        self.tracker = tracker

    def update(self):
        ''' Updates the GUI for every time the thread change '''

        #im_prev = self.cam.getImage()
        if self.cam.im is not None:
            im_prev = self.cam.im
            #print(self.mode)
            #print('Red: ' + str(self.network.activated))
            #print('Tracker: ' + str(self.tracker.activated))

            im = QtGui.QImage(im_prev, im_prev.shape[1], im_prev.shape[0],
                              QtGui.QImage.Format_RGB888)
            im_scaled = im.scaled(self.im_label.size())
            self.im_label.setPixmap(QtGui.QPixmap.fromImage(im_scaled))  # We get the original image and display it.

            # Display results
            if self.mode == 'continuous':

                try:

                    if self.count == 0:
                        self.network.setInputImage(im_prev)  # segment first frame
                        self.count += 1

                    im_segmented = self.network.getOutputImage()[0]
                    self.network_not_finished = self.network.getOutputImage()[1]

                    self.buffer = self.cam.buffer_cam

                    if not self.tracker.activated and not self.network.activated and self.mode == 'continuous':  # segmentation

                        self.network.setInputImage(self.buffer[len(self.buffer) - 1])  # segment last frame in buffer
                        self.network.toggleNetwork()  # network on

                        # segmentada
                        #cv2.imshow('image_to_net', im_segmented)
                        im_segmented_qimage = QtGui.QImage(im_segmented.data, im_segmented.shape[1], im_segmented.shape[0],
                                                           QtGui.QImage.Format_RGB888)
                        im_segmented_scaled = im_segmented_qimage.scaled(self.im_combined_label.size())
                        self.im_combined_label.setPixmap(QtGui.QPixmap.fromImage(im_segmented_scaled))
                        self.im_segmented_label.setPixmap(QtGui.QPixmap.fromImage(im_segmented_scaled))

                        # tracking configuration
                        detection = self.network.getOutputDetection()
                        label = self.network.getOutputLabel()
                        color_list = self.network.getColorList()
                        self.tracker.setInputDetection(detection, True)
                        self.tracker.setInputLabel(label)
                        self.tracker.setColorList(color_list)
                        self.tracker.setBuffer(self.buffer)
                        self.cam.buffer_cam = []  # new buffer
                        self.tracker.activated = True  # tracker on

                    elif self.tracker.activated and self.mode == 'continuous':  # tracking output

                        im_detection = self.tracker.getOutputImage()
                        self.tracker.checkProgress()
                        if im_detection is not None:
                            im_detection = QtGui.QImage(im_detection.data, im_detection.shape[1], im_detection.shape[0],
                                                            QtGui.QImage.Format_RGB888)
                            im_detection_scaled = im_detection.scaled(self.im_segmented_label.size())
                            self.im_combined_label.setPixmap(QtGui.QPixmap.fromImage(im_detection_scaled))
                            self.im_tracked_label.setPixmap(QtGui.QPixmap.fromImage(im_detection_scaled))

                    elif not self.tracker.activated and self.network.activated:  # tracker ends but no result from network -> discard frame
                        print('descartei!')
                        no_track = self.buffer.pop(0)
                        im_detection = QtGui.QImage(no_track.data, no_track.shape[1], no_track.shape[0],
                                                    QtGui.QImage.Format_RGB888)
                        im_detection_scaled = im_detection.scaled(self.im_segmented_label.size())
                        self.im_combined_label.setPixmap(QtGui.QPixmap.fromImage(im_detection_scaled))  # show discarded frames

                except AttributeError:
                    pass

            else:  # 'once' mode
                self.network.setInputImage(im_prev)
                im_segmented = self.network.getOutputImage()[0]
                im_segmented_qimage = QtGui.QImage(im_segmented.data, im_segmented.shape[1], im_segmented.shape[0],
                                                   QtGui.QImage.Format_RGB888)
                im_segmented_scaled = im_segmented_qimage.scaled(self.im_segmented_label.size())
                self.im_combined_label.setPixmap(QtGui.QPixmap.fromImage(im_segmented_scaled))
        else:
            self.cam.count = 0

    def toggleMode(self):
        if self.mode == 'continuous':
            self.mode = 'once'
        else:
            self.mode = 'continuous'
            self.count = 0  # initialize buffer settings to zero
            self.buffer = []
            self.tracker.buffer_in = []
            self.tracker.buffer_out = []
            self.tracker.new_detection = False

    def toggleNetwork(self):
        self.toggleMode()
        self.network.toggleNetwork()
        self.tracker.activated = False
        if self.network.activated:
            self.button_continuous.setStyleSheet('QPushButton {color: green;}')
        else:
            self.button_continuous.setStyleSheet('QPushButton {color: red;}')

    def updateOnce(self):
        self.t_network.runOnce()