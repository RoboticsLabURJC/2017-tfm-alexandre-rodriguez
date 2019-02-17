#
# Created on Feb, 2018
#
# @author: alexandre2r
#
# Based on @nuriaoyaga code:
# https://github.com/RoboticsURJC-students/2016-tfg-nuria-oyaga/blob/
#     master/gui/gui.py
#

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QPushButton
import numpy as np
import sys

from Net.threadnetwork import ThreadNetwork as t_net
from Camera.threadcamera import ThreadCamera as t_cam
from Tracker.threadtracker import ThreadTracker as t_tracker
from GUI.threadgui import ThreadGUI as t_gui


class GUI(QtWidgets.QWidget):
    updGUI = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        ''' GUI class creates the GUI that we're going to use to
        preview the live video as well as the results of the application. '''

        self.app = QtWidgets.QApplication([])

        QtWidgets.QWidget.__init__(self, parent)
        self.setWindowTitle("Object Tracker")
        self.resize(1200, 1100)
        self.move(150, 50)
        self.updGUI.connect(self.update)

        # original image labels
        self.im_label = QtWidgets.QLabel(self)
        self.im_label.resize(480, 320)
        self.im_label.move(70, 40)
        self.im_label.show()
        self.im_label_txt = QtWidgets.QLabel(self)
        self.im_label_txt.resize(50, 40)
        self.im_label_txt.move(290, 10)
        self.im_label_txt.setText('Input')
        self.im_label_txt.show()

        # black image at the beginning
        zeros = np.zeros((480, 320), dtype=np.int32)
        zeros = QtGui.QImage(zeros.data, zeros.shape[1], zeros.shape[0],
                             QtGui.QImage.Format_RGB888)
        zeros_scaled = zeros.scaled(self.im_label.size())

        # combined image labels
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

        # neural network image labels
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

        # tracked image labels
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

        # button for processing a single frame
        self.button_now = QtWidgets.QPushButton('Run now', self)
        self.button_now.move(1140, 60)
        self.button_now.clicked.connect(self.updateOnce)

        # button for processing continuous frames
        self.button_continuous = QtWidgets.QPushButton('Run continuous', self)
        self.button_continuous.move(1120, 100)
        self.button_continuous.clicked.connect(self.buttonClicked)
        self.button_continuous.setStyleSheet('QPushButton {color: green;}')

        self.mode = 'continuous'
        self.count = 0
        self.buffer = []

    def setNetwork(self, network, t_network):
        ''' Declares the Network object and its corresponding control thread. '''
        self.network = network
        self.t_network = t_network

    def setTracker(self, tracker):
        ''' Declares the Tracker object. '''
        self.tracker = tracker

    def toggleMode(self):
        ''' Changes GUI mode. '''
        if self.mode == 'continuous':
            self.mode = 'once'
        else:
            self.mode = 'continuous'
            self.count = 0  # initialize buffer settings to zero
            self.buffer = []
            self.tracker.buffer_in = []
            self.tracker.buffer_out = []
            self.tracker.new_detection = False

    def buttonClicked(self):
        ''' Run continuous button is clicked. '''
        self.toggleMode()
        self.network.toggleNetwork()
        self.tracker.activated = False
        if self.network.activated:
            self.button_continuous.setStyleSheet('QPushButton {color: green;}')
        else:
            self.button_continuous.setStyleSheet('QPushButton {color: red;}')

    def updateOnce(self):
        ''' Run once mode. '''
        self.t_network.runOnce()
