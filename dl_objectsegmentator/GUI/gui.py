#
# Created on Feb, 2018
#
# @author: alexandre2r
#
# Based on @nuriaoyaga code:
# https://github.com/RoboticsURJC-students/2016-tfg-nuria-oyaga/blob/
#     master/gui/gui.py
#

import time
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets


class GUI(QtWidgets.QWidget):

    updGUI = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        ''' GUI class creates the GUI that we're going to use to
        preview the live video as well as the results of the segmentation.
        '''

        QtWidgets.QWidget.__init__(self, parent)
        self.setWindowTitle("Object Segmentator (Keras-based Mask R-CNN trained with a COCO database)")
        self.resize(1180, 500)
        self.move(150, 50)
        self.updGUI.connect(self.update)

        # Original image label
        self.im_label = QtWidgets.QLabel(self)
        self.im_label.resize(480, 320)
        self.im_label.move(70, 50)
        self.im_label.show()

        # Segmented image label
        self.im_segmented_label = QtWidgets.QLabel(self)
        self.im_segmented_label.resize(480, 320)
        self.im_segmented_label.move(610, 50)
        self.im_segmented_label.show()

        # Button for processing continuous frames
        self.button_continuous = QtWidgets.QPushButton('Run continuous', self)
        self.button_continuous.move(520, 440)
        self.button_continuous.clicked.connect(self.toggleNetwork)
        self.button_continuous.setStyleSheet('QPushButton {color: green;}')

        # Button for processing a single frame
        self.button_now = QtWidgets.QPushButton('Run now', self)
        self.button_now.move(540, 400)
        self.button_now.clicked.connect(self.updateOnce)

        self.detection = None
        self.last_segmented = None

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
        # We get the original image and display it.

        im_prev = self.cam.getImage()
        self.network.setInputImage(im_prev)
        self.tracker.setInputImage(im_prev)

        im = QtGui.QImage(im_prev.data, im_prev.shape[1], im_prev.shape[0],
                          QtGui.QImage.Format_RGB888)
        im_scaled = im.scaled(self.im_label.size())

        self.im_label.setPixmap(QtGui.QPixmap.fromImage(im_scaled))

        # Display segmentation results
        try:
            im_segmented = self.network.getOutputImage()
            if ((im_segmented == self.last_segmented).all()) and im_segmented is not None and self.network.activated:

                detection = self.network.getOutputDetection()
                label = self.network.getOutputLabel()
                color_list = self.network.getColorList()
                self.tracker.setInputDetection(detection)
                self.tracker.setInputLabel(label)
                self.tracker.setColorList(color_list)
                im_detection = self.tracker.getOutputImage()
                im_detection = QtGui.QImage(im_detection.data, im_detection.shape[1], im_detection.shape[0],
                                            QtGui.QImage.Format_RGB888)

                im_detection_scaled = im_detection.scaled(self.im_segmented_label.size())

                self.im_segmented_label.setPixmap(QtGui.QPixmap.fromImage(im_detection_scaled))
            else:

                self.last_segmented = im_segmented
                im_segmented = QtGui.QImage(im_segmented.data, im_segmented.shape[1], im_segmented.shape[0],
                                QtGui.QImage.Format_RGB888)

                im_segmented_scaled = im_segmented.scaled(self.im_segmented_label.size())

                self.im_segmented_label.setPixmap(QtGui.QPixmap.fromImage(im_segmented_scaled))

        except AttributeError:
            '''
            im_segmented = np.zeros((480, 320), dtype=np.int32)
            im_segmented = QtGui.QImage(im_segmented.data, im_segmented.shape[1], im_segmented.shape[0],
                                        QtGui.QImage.Format_RGB888)
            im_segmented_scaled = im_segmented.scaled(self.im_segmented_label.size())
            self.im_segmented_label.setPixmap(QtGui.QPixmap.fromImage(im_segmented_scaled))
            '''
            pass

    def toggleNetwork(self):
        self.network.toggleNetwork()
        self.tracker.toggleTracker()

        if self.network.activated:
            self.button_continuous.setStyleSheet('QPushButton {color: green;}')
        else:
            self.button_continuous.setStyleSheet('QPushButton {color: red;}')

    def updateOnce(self):
        self.t_network.runOnce()