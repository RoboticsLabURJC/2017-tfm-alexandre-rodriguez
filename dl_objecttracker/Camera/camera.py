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
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ROSCamera:

    def __init__(self):
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.callback)
        self.bridge = CvBridge()
        self.image = None
        self.im_height = None
        self.im_width = None

    def callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.im_height = self.image.shape[0]
        self.im_width = self.image.shape[1]

    def read(self):
        return self.image


class Camera:
    def __init__(self, cam, gui_cfg):
        ''' Camera class gets images from live video. '''

        # initialize Camera instance attributes
        self.cam = None
        self.source = None
        self.im_height = None
        self.im_width = None
        self.image_net_size = (None, None)
        self.gui_cfg = gui_cfg
        self.im = None
        self.buffer = []
        self.im_once_set = False
        self.im_net = None
        self.frame_counter = 0
        self.frame_to_process = None
        self.network_framework = None
        self.filename_offset = 0
        self.last_frames_video = False
        self.frame_tag = []
        self.logger_status = True  # logging on by default

        # image source: stream ROS
        if cam == 'stream':
            self.source = 'stream_camera'
            rospy.init_node('ROS_camera', anonymous=True)
            self.cam = ROSCamera()

        # image source: local camera (OpenCV)
        elif isinstance(cam, int):
            self.cam = cv2.VideoCapture(cam)
            self.source = 'local_camera'
            if not self.cam.isOpened():
                print("%d is not a valid device index in this machine." % cam)
                raise SystemExit("Please check your camera id (hint: ls /dev)")
            self.im_width = int(self.cam.get(3))
            self.im_height = int(self.cam.get(4))
            print('Image size: {0}x{1} px'.format(
                self.im_width, self.im_height))

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
            print('Image size: {0}x{1} px'.format(
                self.im_width, self.im_height))

        else:
            raise SystemExit("Interface camera not connected")

        if self.gui_cfg == 'off':
            print('GUI not set')

    def getImage(self):
        ''' Gets the image from the source and returns it resized and tagged with the frame number. '''

        im = None

        if self.cam and self.source == 'stream_camera':
            frame = self.cam.read()
            if frame is not None:
                self.im_height = frame.shape[0]
                self.im_width = frame.shape[1]
                im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im = self.resizeImage(im)
                im = np.reshape(im, (self.image_net_size[0], self.image_net_size[1], 3))
                self.frame_counter += 1
                cv2.putText(im, str(self.frame_counter), (int(self.image_net_size[0]/1.13), int(self.image_net_size[1]/1.06)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                            thickness=2)  # numerate frames to debug, REMOVE in final version
                self.frame_tag.append(self.frame_counter)

        elif self.cam and (self.source == 'local_camera' or self.source == 'local_video'):
            _, frame = self.cam.read()
            if frame is not None:
                im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im = self.resizeImage(im)
                im = np.reshape(im, (self.image_net_size[0], self.image_net_size[1], 3))
                self.frame_counter += 1
                cv2.putText(im, str(self.frame_counter), (int(self.image_net_size[0]/1.13), int(self.image_net_size[1]/1.06)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                            thickness=2)  # numerate frames to debug, REMOVE in final version
                self.frame_tag.append(self.frame_counter)
        return im

    def resizeImage(self, im):
        ''' Resizes the image. '''
        im_resized = np.reshape(im, (self.im_height, self.im_width, 3))
        self.network.image_scale = (float(self.im_width)/self.image_net_size[0], float(self.im_height)/self.image_net_size[1])
        self.tracker.image_scale = self.network.image_scale
        im_resized = cv2.resize(im_resized, self.image_net_size, cv2.INTER_NEAREST)  #ToDo: allow different input sizes
        return im_resized

    def setGUI(self, gui):
        ''' Declares the GUI object. '''
        self.gui = gui

    def setNetwork(self, network, t_network):
        ''' Declares the Network object and its corresponding control thread. '''
        self.network = network
        self.t_network = t_network
        self.network_framework = network.framework
        self.network.logger_status = self.logger_status

    def setTracker(self, tracker):
        ''' Declares the Tracker object. '''
        self.tracker = tracker
        self.tracker.network_framework = self.network_framework  # indicates the neural network type to the tracker
        self.tracker.logger_status = self.logger_status

    def toggleNetworkAndTracker(self):
        ''' Network and Tracker off. '''
        self.network.toggleNetwork()
        self.tracker.activated = False

    def setLogger(self, logger_status):
        if logger_status:
            self.logger_status = True
        else:
            self.logger_status = False

    def setNetworkParams(self, image_net_size, confidence):
        self.image_net_size = (image_net_size[0], image_net_size[1])
        self.network.confidence_threshold = confidence

    def trackerConfiguration(self):
        ''' Initializes tracker parameters. '''
        detection = self.network.getOutputDetection()
        label = self.network.getOutputLabel()
        color_list = self.network.getColorList()
        self.tracker.setInputDetection(detection, True)
        self.tracker.setInputLabel(label)
        self.tracker.setColorList(color_list)
        self.tracker.setBuffer(self.buffer)
        self.tracker.setFrameTags(self.frame_tag)
        self.filename_offset = len(self.buffer) + 1
        self.buffer = []  # new buffer, reset buffer
        self.frame_tag = []
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

                    if self.gui.count == 0:
                        self.network.setInputImage(self.im, self.frame_counter)  # process first frame
                        self.frame_to_process = self.frame_counter
                        self.gui.count += 1
                    else:
                        if self.source == 'local_video' and self.im is not None:
                            self.buffer.append(self.im)  # allows processing last frames in buffer
                        elif self.source == 'local_camera' or self.source == 'stream_camera':
                            self.buffer.append(self.im)

                    processed_frame = self.network.getProcessedFrame()

                    if processed_frame == self.frame_to_process and not self.network.getOutputImage()[1]:
                        self.im_net = self.network.getOutputImage()[0]

                    if not self.tracker.activated and not self.network.activated and not self.network.getOutputImage()[1]:  # added check if net output is not zeros
                        self.network.setInputImage(
                            self.buffer[len(self.buffer) - 1], self.frame_counter)  # process last frame in buffer
                        self.frame_to_process = self.frame_counter
                        self.network.toggleNetwork()  # network on

                        if self.gui_cfg == 'on':
                            #  show segmented image
                            im_net = QtGui.QImage(self.im_net.data, self.im_net.shape[1],
                                                               self.im_net.shape[0],
                                                               QtGui.QImage.Format_RGB888)
                            im_net_scaled = im_net.scaled(self.gui.im_net_label.size())
                            self.gui.im_combined_label.setPixmap(QtGui.QPixmap.fromImage(im_net_scaled))
                            self.gui.im_net_label.setPixmap(QtGui.QPixmap.fromImage(im_net_scaled))
                        else:  # gui off, save image
                            cv2.imwrite(str(self.frame_counter - self.filename_offset) + '.jpg', self.im_segmented)  # in BGR

                        # tracking configuration
                        self.trackerConfiguration()

                    elif self.tracker.activated:  # get tracker output
                        im_tracked = self.tracker.getOutputImage()
                        self.tracker.checkProgress()
                        if im_tracked is not None:
                            if self.gui_cfg == 'on':
                                im_tracked = QtGui.QImage(im_tracked.data, im_tracked.shape[1],
                                                            im_tracked.shape[0],
                                                            QtGui.QImage.Format_RGB888)
                                im_tracked_scaled = im_tracked.scaled(self.gui.im_tracked_label.size())
                                self.gui.im_combined_label.setPixmap(QtGui.QPixmap.fromImage(im_tracked_scaled))
                                self.gui.im_tracked_label.setPixmap(QtGui.QPixmap.fromImage(im_tracked_scaled))
                            else:  # gui off, save image
                                saved_image = cv2.cvtColor(im_tracked, cv2.COLOR_BGR2RGB)
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
                        self.im_net = self.network.getOutputImage()[0]
                if np.any(self.im_net.data):  # set segmented frame
                    if self.gui_cfg == 'on':
                        im_net = QtGui.QImage(self.im_net.data, self.im_net.shape[1],
                                                           self.im_net.shape[0],
                                                           QtGui.QImage.Format_RGB888)
                        im_net_scaled = im_net.scaled(self.gui.im_net_label.size())
                        self.gui.im_combined_label.setPixmap(QtGui.QPixmap.fromImage(im_net_scaled))
                        self.gui.im_net_label.setPixmap(QtGui.QPixmap.fromImage(im_net_scaled))
                    else:  # gui off, save image
                        saved_image = cv2.cvtColor(self.im_net, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(str(self.frame_counter - self.filename_offset) + '.jpg', saved_image)
                    self.im_once_set = False
                    self.buffer = self.buffer[self.frame_to_process:len(self.buffer)]

        elif self.source == 'local_video':  # allows processing last frames in buffer
            if not self.last_frames_video:
                self.frame_counter -= self.filename_offset  # avoids overwriting filename in last frames
                self.last_frames_video = not self.last_frames_video
            elif self.last_frames_video:
                self.frame_counter += 1
            im_tracked = self.tracker.getOutputImage()
            im_net = self.network.getOutputImage()[0]
            self.tracker.checkProgress()
            if im_tracked is not None:  # last tracked from Tracker
                if self.gui_cfg == 'on':
                    im_tracked = QtGui.QImage(im_tracked.data, im_tracked.shape[1],
                                                im_tracked.shape[0],
                                                QtGui.QImage.Format_RGB888)
                    im_tracked_scaled = im_tracked.scaled(self.gui.im_tracked_label.size())
                    self.gui.im_combined_label.setPixmap(QtGui.QPixmap.fromImage(im_tracked_scaled))
                    self.gui.im_tracked_label.setPixmap(QtGui.QPixmap.fromImage(im_tracked_scaled))
                else:  # gui off, save image
                    saved_image = cv2.cvtColor(im_tracked, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(str(self.frame_counter) + '.jpg', saved_image)  # in RGB
            if im_net is not None and im_tracked is None:  # last detection from Net
                if self.gui_cfg == 'on':
                    im_net = QtGui.QImage(im_net.data, im_net.shape[1],
                                                im_net.shape[0],
                                                QtGui.QImage.Format_RGB888)
                    im_net_scaled = im_net.scaled(self.gui.im_net_label.size())
                    self.gui.im_combined_label.setPixmap(QtGui.QPixmap.fromImage(im_net_scaled))
                    self.gui.im_net_label.setPixmap(QtGui.QPixmap.fromImage(im_net_scaled))  #ToDo: set functions de net/tracker image in GUI
                else:  # gui off, save image
                    saved_image = cv2.cvtColor(im_net, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(str(self.frame_counter) + '.jpg', saved_image)  # in RGB

            if self.tracker.logger_status and not self.network.activated and not self.tracker.activated:
                self.tracker.logTracking()
                self.network.logNetwork()

            if not self.network.activated and not self.tracker.activated:
                print('Finished processing video, please close the window...')