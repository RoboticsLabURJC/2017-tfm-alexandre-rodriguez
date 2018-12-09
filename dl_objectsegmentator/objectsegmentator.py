#
# Created on Feb, 2018
#
# @author: alexandre2r
#
# It receives images from a live video and classify them into object classes using CNNs and Tracking,
# based on TensorFlow Deep Learning middleware.
# It shows the live video and the results in a GUI.
#
# Based on @nuriaoyaga code:
# https://github.com/RoboticsURJC-students/2016-tfg-nuria-oyaga/blob/
#     master/numberclassifier.py
# and @dpascualhe's:
# https://github.com/RoboticsURJC-students/2016-tfg-david-pascual/blob/
#     master/digitclassifier.py
#
#

import sys
import signal

from PyQt5 import QtWidgets

from Camera.camera import Camera
from Camera.threadcamera import ThreadCamera
from GUI.gui import GUI
from GUI.threadgui import ThreadGUI
from Net.network import Segmentation_Network
from Net.threadnetwork import ThreadNetwork
from Tracker.tracker import Tracker
from Tracker.threadtracker import ThreadTracker

import config
import comm

signal.signal(signal.SIGINT, signal.SIG_DFL)

if __name__ == '__main__':

    # Creation of the camera through the comm-ICE proxy.
    try:
        cfg = config.load(sys.argv[1])
    except IndexError:
        raise SystemExit('Missing YML file. Usage: python2 objectsegmentator.py objectsegmentator.yml on')

    try:
        gui_cfg = sys.argv[2]
    except IndexError:
        raise SystemExit('Missing GUI configuration. Usage: python2 objectsegmentator.py objectsegmentator.yml on')

    jdrc = comm.init(cfg, 'ObjectSegmentator')
    proxy = jdrc.getCameraClient('ObjectSegmentator.Camera')

    network = Segmentation_Network()
    # Threading Network
    t_network = ThreadNetwork(network)
    t_network.start()

    tracker = Tracker()
    # Threading Tracker
    t_tracker = ThreadTracker(tracker)
    t_tracker.start()

    if gui_cfg == 'on':
        app = QtWidgets.QApplication(sys.argv)
        window = GUI()
        cam = Camera(proxy, gui_cfg)
        cam.setGUI(window)
        cam.setNetwork(network, t_network)
        cam.setTracker(tracker)

        # Threading camera
        t_cam = ThreadCamera(cam)
        t_cam.start()
        window.setCamera(cam)
        window.setNetwork(network, t_network)
        window.setTracker(tracker)
        window.show()

        # Threading GUI
        t_gui = ThreadGUI(window)
        t_gui.start()

        sys.exit(app.exec_())

    else:  # gui off
        cam = Camera(proxy, gui_cfg)
        # Threading camera
        t_cam = ThreadCamera(cam)
        t_cam.start()
        cam.setNetwork(network, t_network)
        cam.setTracker(tracker)