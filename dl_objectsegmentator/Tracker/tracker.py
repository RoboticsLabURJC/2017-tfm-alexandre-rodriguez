import threading
import cv2
import time


class Tracker:
    def __init__(self):

        self.tracker = cv2.MultiTracker_create()
        print("Tracker created!")

        self.lock = threading.Lock()

        self.activated = False
        self.input_detection = None
        self.new_detection = False
        self.input_label = None
        self.output_image = None
        self.color_list = None
        self.buffer_in = []
        self.buffer_out = []
        self.image_counter = 0
        self.last_fps_buffer = [0 for _ in range(5)]
        self.tracker_slow = False
        self.tracker_fast = False
        self.counter_slow = 0
        self.counter_fast = 0
        self.len_buffer_in = 0
        self.first_image_to_track = True
        self.image = None

    def track(self):

        detection = self.input_detection

        if detection is not None and self.new_detection:  # new detection from net

            self.tracker = cv2.MultiTracker_create()
            self.last_fps_buffer = [0 for _ in range(5)]
            self.first_image_to_track = True

        elif detection is None:  # no detection from net

            self.image = self.buffer_in.pop(0)
            self.buffer_out.append(self.image)

        if self.image_counter != self.len_buffer_in and detection is not None:

            start_time = time.time()

            if self.tracker_slow:
                if len(self.buffer_in) > 5:
                    self.buffer_in.pop(0)
                    self.buffer_in.pop(0)
                    self.buffer_in.pop(0)
                    self.buffer_in.pop(0)
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
                self.counter_slow = 0  # reset counters when changing fps
                self.counter_fast = 0

            if self.activated:  # avoid to continue the loop if not activated
                for i in range(len(detection)):
                    if self.first_image_to_track:  # create multitracker only in the first frame of the buffer, same detections
                        self.tracker.add(cv2.TrackerTLD_create(), self.image, (
                            detection[i][1], detection[i][0], detection[i][3] - detection[i][1],
                            detection[i][2] - detection[i][0]))
                self.first_image_to_track = False
                self.new_detection = False
                _, boxes = self.tracker.update(self.image)

                for i, newbox in enumerate(boxes):
                    p1 = (int(newbox[0]), int(newbox[1]))
                    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                    if len(self.color_list) == len(boxes):
                        cv2.rectangle(self.image, p1, p2, self.color_list[i], thickness=2)
                        cv2.putText(self.image, self.input_label[i], (p1[0], p1[1] + 20), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.75,
                                    (0, 0, 0), thickness=2, lineType=2)

                self.buffer_out.append(self.image)
                fps_rate = 1.0 / (time.time() - start_time)
                self.last_fps_buffer.pop(0)
                self.last_fps_buffer.append(fps_rate)
                avg_fps = sum(self.last_fps_buffer) / len(self.last_fps_buffer)
                print('Average fps: ' + str(avg_fps))

                if not (0 in self.last_fps_buffer) and avg_fps < 2.5:
                    self.counter_slow += 1
                    if self.counter_slow == 5:
                        self.counter_slow = 0
                        self.tracker_slow = True
                elif not (0 in self.last_fps_buffer) and self.tracker_slow == True:
                    self.tracker_slow = False
                elif avg_fps > 15 and self.counter_fast <= 2:
                    self.counter_fast += 1
                    self.tracker_fast = True

            else:
                self.image_counter = 0  # reset when toggling modes (activated=False)

    def setBuffer(self, buf):
        self.buffer_in = buf  # fixes overlap segmentation+tracking in net result
        self.len_buffer_in = len(self.buffer_in)
        print('New buffer with length ' + str(len(self.buffer_in)))

    def setInputDetection(self, bbox, state):
        self.input_detection = bbox
        self.new_detection = state

    def setInputLabel(self, label):
        self.input_label = label

    def setColorList(self, color_list):
        self.color_list = color_list

    def getOutputImage(self):

        if self.input_detection is not None and self.buffer_out and not self.tracker_slow and not self.tracker_fast:  # check if list is not empty
            self.image_counter += 1
            return self.buffer_out.pop(0)  # returns last detection and deletes it
        elif self.buffer_out and self.tracker_slow:
            if len(self.buffer_in) > 5:
                self.image_counter += 5
                return self.buffer_out.pop(0)  # image counter incremented in tracking
            else:
                self.image_counter += 1
                return self.buffer_out.pop(0)
        elif self.buffer_out and self.tracker_fast:
            if self.counter_fast == 2:  # slow tracking a little, wait for network result
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
        print(self.image_counter)
        if self.image_counter == self.len_buffer_in:
            self.toggleTracker()
            self.buffer_out = []
            self.image_counter = 0
            self.tracker_slow = False
            self.new_detection = False
            print('Tracker OFF')

    def toggleTracker(self):
        ''' Toggles the tracker on/off '''
        self.activated = not self.activated
