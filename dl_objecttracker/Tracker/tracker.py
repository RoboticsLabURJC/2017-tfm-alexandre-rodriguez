import cv2
import time
import dlib

class Tracker:
    def __init__(self):

        self.tracker = cv2.MultiTracker_create()
        print("Tracker created!")
        self.activated = False
        self.input_detection = None
        self.new_detection = False
        self.input_label = None
        self.output_image = None
        self.color_list = None
        self.buffer_in = []
        self.buffer_out = []
        self.image_counter = 0
        self.last_fps_buffer = [0 for _ in range(3)]
        self.avg_fps = 0
        self.tracker_slow = False
        self.tracker_fast = False
        self.counter_slow = 0
        self.counter_fast = 0
        self.len_buffer_in = 0
        self.first_image_to_track = True
        self.image = None
        self.network_framework = None
        self.trackers_dlib = [] #ToDo: rename this vars
        self.labels_dlib = []

    def imageToTrack(self):
        ''' Assigns a new image to track depending on certain conditions. '''
        if self.tracker_slow:
            if len(self.buffer_in) > 3:
                self.buffer_in.pop(0)
                self.buffer_in.pop(0)
                # self.buffer_in.pop(0)
                # self.buffer_in.pop(0)
                self.image = self.buffer_in.pop(0)  # jump frames
            elif len(self.buffer_in) > 0:
                self.image = self.buffer_in.pop(0)
        elif self.tracker_fast and self.counter_fast == 1 and len(self.buffer_in) > 0:
            self.image = self.buffer_in.pop(0)
        elif self.first_image_to_track:
            self.image = self.buffer_in.pop(0)
            self.image_counter += 1
        elif not self.tracker_slow and not self.tracker_fast and len(self.buffer_in) > 0:
            self.image = self.buffer_in.pop(0)

    def calculateFPS(self, start_time):
        ''' Calculates the average FPS. '''
        fps_rate = 1.0 / (time.time() - start_time)
        self.last_fps_buffer.pop(0)
        self.last_fps_buffer.append(fps_rate)
        self.avg_fps = sum(self.last_fps_buffer) / len(self.last_fps_buffer)
        #print('FPS avg: ' + str(self.avg_fps))
        return self.avg_fps

    def trackerSpeedMode(self, avg_fps):
        ''' Obtains the tracker speed. '''
        if not (0 in self.last_fps_buffer) and avg_fps < 10:  # tracker slow
            self.counter_slow += 1
            if self.counter_slow == 3:
                self.counter_slow = 0
                self.tracker_slow = True
        elif not (0 in self.last_fps_buffer) and 10 < avg_fps < 25:  # tracker normal
            self.tracker_slow = False
            self.tracker_fast = False
        elif avg_fps > 25 and self.counter_fast < 1:  # tracker fast
            self.counter_fast += 1
            self.tracker_fast = True

    def configureFirstTrack(self, detection):
        ''' Configures the tracker with the detections from the net. '''

        #dlib
        # if self.network_framework == "Keras":
        #     xmin = detection[1]
        #     ymin = detection[0]
        #     xmax = detection[3]
        #     ymax = detection[2]
        # elif self.network_framework == "TensorFlow":
        #     xmin = detection[0]
        #     ymin = detection[1]
        #     xmax = detection[2]
        #     ymax = detection[3]
        #
        # t = dlib.correlation_tracker()
        # rect = dlib.rectangle(xmin, ymin, xmax, ymax)
        # if not rect.is_empty():
        #     t.start_track(self.image, rect)
        #     self.trackers_dlib.append(t)

        #opencv
        if self.network_framework == "Keras":
            xmin = detection[1]
            ymin = detection[0]
            xmax = detection[3] - xmin
            ymax = detection[2] - ymin
        elif self.network_framework == "TensorFlow":
            xmin = detection[0]
            ymin = detection[1]
            xmax = detection[2] - xmin
            ymax = detection[3] - ymin

        self.tracker.add(cv2.TrackerMOSSE_create(), self.image, (
            xmin, ymin, xmax, ymax))

    def track(self):
        ''' The tracking function. '''

        detection = self.input_detection

        if detection is not None and self.new_detection:  # new detection from net

            self.tracker = cv2.MultiTracker_create()
            self.last_fps_buffer = [0 for _ in range(3)]
            self.first_image_to_track = True

        elif detection is None:  # no detection from net

            self.image = self.buffer_in.pop(0)
            self.buffer_out.append(self.image)

        if self.image_counter != self.len_buffer_in and detection is not None:

            start_time = time.time()
            self.imageToTrack()

            if self.activated:  # avoid to continue the loop if not activated

                # if self.new_detection: #dlib
                #     self.trackers_dlib = []
                #     self.labels_dlib = []
                for i in range(len(detection)):
                    if self.first_image_to_track:  # create multitracker only in the first frame of the buffer
                        self.configureFirstTrack(detection[i])
                        # self.labels_dlib = self.input_label #dlib

                self.first_image_to_track = False
                self.new_detection = False
                _, boxes = self.tracker.update(self.image)

                for i, newbox in enumerate(boxes):
                    p1 = (int(newbox[0]), int(newbox[1]))
                    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                    if len(self.input_label) == len(boxes):
                        cv2.rectangle(self.image, p1, p2, self.color_list[self.input_label[i]], thickness=2)
                        cv2.putText(self.image, self.input_label[i], (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.45,
                                    (0, 0, 0), thickness=2, lineType=2)
                        cv2.putText(self.image, 'FPS avg tracking: ' + str(self.avg_fps)[:5], (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.45,
                                    (255, 0, 0), thickness=1, lineType=1)

                # for i, (t, l) in enumerate(zip(self.trackers_dlib, self.labels_dlib)): #dlib
                #     # update the tracker and grab the position of the tracked object
                #     t.update(self.image)
                #     pos = t.get_position()
                #
                #     # unpack the position object
                #     p1 = (int(pos.left()), int(pos.top()))
                #     p2 = (int(pos.right()), int(pos.bottom()))
                #
                #     # draw the bounding box from the dlib tracker
                #     cv2.rectangle(self.image, p1, p2, self.color_list[self.input_label[i]], thickness=2)
                #     cv2.putText(self.image, l, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                #                 0.45,
                #                 (0, 0, 0), thickness=2, lineType=2)
                #     cv2.putText(self.image, 'FPS avg tracking: ' + str(self.avg_fps)[:5], (10, 20),
                #                 cv2.FONT_HERSHEY_SIMPLEX,
                #                 0.45,
                #                 (255, 0, 0), thickness=1, lineType=1)

                self.buffer_out.append(self.image)
                avg_fps = self.calculateFPS(start_time)
                self.trackerSpeedMode(avg_fps)

            else:
                self.image_counter = 0  # reset when toggling modes (activated=False)

    def setBuffer(self, buf):
        ''' Set buffer input of tracker. '''
        self.buffer_in = buf
        self.len_buffer_in = len(self.buffer_in)
        #print('New buffer with length ' + str(len(self.buffer_in)))

    def setInputDetection(self, bbox, state):
        ''' Set bboxes coordinates and state of new detection. '''
        self.input_detection = bbox
        self.new_detection = state

    def setInputLabel(self, label):
        ''' Set list of labels. '''
        self.input_label = label

    def setColorList(self, color_list):
        ''' Set list of colors of the bboxes. '''
        self.color_list = color_list

    def getOutputImage(self):
        ''' Get tracked image. '''
        if self.input_detection is not None and self.buffer_out and not self.tracker_slow and not self.tracker_fast:
            self.image_counter += 1
            return self.buffer_out.pop(0)  # returns last detection and deletes it
        elif self.buffer_out and self.tracker_slow:
            if len(self.buffer_in) > 3:
                self.image_counter += 3
                return self.buffer_out.pop(0)
            else:
                self.image_counter += 1
                return self.buffer_out.pop(0)
        elif self.buffer_out and self.tracker_fast:
            if self.counter_fast == 1:  # slow tracking a little, wait for network result
                self.counter_fast = 0
                self.image_counter += 1
                self.tracker_fast = False
                return self.buffer_out.pop(0)
        elif self.buffer_out and self.input_detection is None:
            self.image_counter += 1
            return self.buffer_out.pop(0)
        else:
            return None

    def checkProgress(self):
        ''' Checks tracker progress and resets it if tracking is done. '''
        if self.image_counter == self.len_buffer_in:
            self.toggleTracker()
            self.buffer_out = []
            self.image_counter = 0
            self.tracker_slow = False
            self.tracker_fast = False
            self.counter_slow = 0
            self.counter_fast = 0
            self.new_detection = False
            print('Tracking done!')

    def toggleTracker(self):
        ''' Toggles the tracker (on/off). '''
        self.activated = not self.activated
