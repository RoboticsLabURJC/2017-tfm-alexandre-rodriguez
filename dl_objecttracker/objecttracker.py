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
# https://github.com/RoboticsURJC-students/2016-tfg-nuria-oyaga/blob/master/numberclassifier.py
#
# @dpascualhe code:
# https://github.com/RoboticsURJC-students/2016-tfg-david-pascual/blob/master/digitclassifier.py
#
# and @naxvm code:
# https://github.com/JdeRobot/dl-objectdetector/blob/master/objectdetector.py

import sys
import signal
import yaml

from Camera.threadcamera import ThreadCamera
from GUI.gui import GUI
from GUI.threadgui import ThreadGUI
from Net.threadnetwork import ThreadNetwork
from Tracker.tracker import Tracker
from Tracker.threadtracker import ThreadTracker

signal.signal(signal.SIGINT, signal.SIG_DFL)


def selectVideoSource(cfg, gui_cfg):
    """
    @param cfg: configuration
    @return cam: selected camera
    @raise SystemExit in case of unsupported video source
    """
    source = cfg['ObjectTracker']['Source']
    from Camera.camera import Camera
    if source.lower() == 'local':
        cam_idx = cfg['ObjectTracker']['Local']['DeviceNo']
        print('  Chosen source: local camera (index %d)' % cam_idx)
        cam = Camera(cam_idx, gui_cfg)
    elif source.lower() == 'video':
        video_path = cfg['ObjectTracker']['Video']['Path']
        print('  Chosen source: local video (%s)' % video_path)
        cam = Camera(video_path, gui_cfg)
    elif source.lower() == 'stream':
        print('  Chosen source: stream using ROS')
        cam = Camera('stream', gui_cfg)
    else:
        raise SystemExit('%s not supported! Supported source: Local, Video, Stream' % source)

    return cam


def selectNetwork(cfg):
    """
    @param cfg: configuration
    @return net_prop, DetectionNetwork: network properties and Network class
    @raise SystemExit in case of invalid network
    """
    net_prop = cfg['ObjectTracker']['Network']
    framework = net_prop['Framework']
    if framework.lower() == 'tensorflow':
        from Net.TensorFlow.network import DetectionNetwork
    elif framework.lower() == 'keras':
        sys.path.append('Net/Keras')
        from Net.Keras.network import DetectionNetwork
    else:
        raise SystemExit('%s not supported! Supported frameworks: Keras, TensorFlow' % framework)
    return net_prop, DetectionNetwork


def selectTracker(cfg):
    """
    @param cfg: configuration
    @return net_prop: tracker properties
    @raise SystemExit in case of invalid tracker
    """
    tracker_prop = cfg['ObjectTracker']['Tracker']
    library = tracker_prop['Lib']
    if library.lower() == 'opencv' or library.lower() == 'dlib':
        print('Using ' + library.lower() + ' tracking.')
    else:
        raise SystemExit('%s not supported! Supported trackers of: OpenCV, dlib' % library)
    return tracker_prop, library.lower()


def readLoggerStatus(cfg):
    logger_status = cfg['ObjectTracker']['Logger']['Status']
    if logger_status:
        print('Logging activated.')
    else:
        print('Logging desactivated.')
    return logger_status


def readConfig():
    try:
        with open(sys.argv[1], 'r') as stream:
            return yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        raise SystemExit('Error: Cannot read/parse YML file. Check YAML syntax.')
    except:
        raise SystemExit('\n\tUsage: python2 objecttracker.py objecttracker.yml\n')


if __name__ == '__main__':

    # cfg to specify the use of GUI
    gui_cfg = None
    try:
        gui_cfg = sys.argv[2]
    except IndexError:
        raise SystemExit('Missing GUI configuration. Usage: python2 objecttracker.py objecttracker.yml on')

    cfg = readConfig()
    cam = selectVideoSource(cfg, gui_cfg)
    net_prop, DetectionNetwork = selectNetwork(cfg)
    tracker_prop, tracker_lib_prop = selectTracker(cfg)
    logger_status = readLoggerStatus(cfg)

    network = DetectionNetwork(net_prop)
    # Threading Network
    t_network = ThreadNetwork(network)
    t_network.setDaemon(True)  # setting daemon thread to exit
    t_network.start()

    tracker = Tracker(tracker_prop, tracker_lib_prop)
    # Threading Tracker
    t_tracker = ThreadTracker(tracker)
    t_tracker.setDaemon(True)
    t_tracker.start()

    window = GUI()
    cam.setGUI(window)
    cam.setNetwork(network, t_network)
    cam.setTracker(tracker)
    cam.setLogger(logger_status)

    # Threading camera
    t_cam = ThreadCamera(cam)
    t_cam.setDaemon(True)
    t_cam.start()
    window.setNetwork(network, t_network)
    window.setTracker(tracker)

    if gui_cfg == 'on':
        window.show()
        # Threading GUI
        t_gui = ThreadGUI(window)
        t_gui.setDaemon(True)
        t_gui.start()

    sys.exit(window.app.exec_())