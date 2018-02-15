#
# Created on Feb, 2018
#
# @author: alexandre2r
#
# Based on @nuriaoyaga code:
# https://github.com/RoboticsURJC-students/2016-tfg-nuria-oyaga/blob/
#     master/gui/gui.py
#

import numpy as np
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from Net.network import Segmentation_Network
from Net.threadnetwork import ThreadNetwork


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

        # Button for processing a single frame
        self.button_now = QtWidgets.QPushButton('Run now', self)
        self.button_now.move(540, 400)
        self.button_now.clicked.connect(self.updateOnce)

        # Button for processing continuous frames
        self.button_continuous = QtWidgets.QPushButton('Run continuous', self)
        self.button_continuous.setCheckable(True)
        self.button_continuous.move(520, 440)
        self.button_continuous.setAutoRepeat(True)
        self.button_continuous.setAutoRepeatDelay(0)
        self.button_continuous.setAutoRepeatInterval(30000)
        self.button_continuous.clicked.connect(self.updateContinuous)

        # Network initialization
        self.network = Segmentation_Network()
        self.t_network = ThreadNetwork(self.network)
        self.t_network.start()

    def setCamera(self, cam):
        ''' Declares the Camera object '''
        self.cam = cam

    def update(self):
        ''' Updates the GUI for every time the thread change '''
        # We get the original image and display it.
        im_prev = self.cam.getImage()
        self.network.input_image = im_prev

        im_segmented = self.network.output_image

        im = QtGui.QImage(im_prev.data, im_prev.shape[1], im_prev.shape[0],
                          QtGui.QImage.Format_RGB888)
        im_scaled = im.scaled(self.im_label.size())

        self.im_label.setPixmap(QtGui.QPixmap.fromImage(im_scaled))

        #print(self.button_continuous.isDown())
        #self.button_continuous.clicked.connect(self.printeer)

        # Display segmentation results
        if im_segmented is not None:
            im_segmented = QtGui.QImage(im_segmented.data, im_segmented.shape[1], im_segmented.shape[0],
                            QtGui.QImage.Format_RGB888)
            im_segmented_scaled = im_segmented.scaled(self.im_segmented_label.size())
            self.im_segmented_label.setPixmap(QtGui.QPixmap.fromImage(im_segmented_scaled))
        else:
            im_segmented = np.zeros((480, 320), dtype=np.int32)
            im_segmented = QtGui.QImage(im_segmented.data, im_segmented.shape[1], im_segmented.shape[0],
                                        QtGui.QImage.Format_RGB888)
            im_segmented_scaled = im_segmented.scaled(self.im_segmented_label.size())
            self.im_segmented_label.setPixmap(QtGui.QPixmap.fromImage(im_segmented_scaled))


    def updateOnce(self):
        self.t_network.activated = True
        self.t_network.runOnce()

    def updateContinuous(self):
        self.t_network.activated = True
        self.button_continuous.setChecked(True)
        self.button_continuous.setDown(True)
        self.t_network.runOnce()

    #def printeer(self):
        #print('clicked')