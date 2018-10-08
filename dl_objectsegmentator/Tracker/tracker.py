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
        self.input_label = None
        self.output_image = None
        self.color_list = None
        self.buffer_in = []
        self.buffer_out = []
        self.image_counter = 0
        self.new_detection = False
        self.tracker_slow = False
        self.tracker_fast = False
        self.len_buffer_in = 0
        self.buffer_step = 0
        self.image_counter_fast = 0

    def track(self):

        detection = self.input_detection

        if detection is not None and self.new_detection:  # new detection

            self.tracker = cv2.MultiTracker_create()
            last_fps_buffer = [0 for _ in range(5)]
            counter = 0

            for idx, image in enumerate(self.buffer_in):
                print('Live buffer:'+str(len(self.buffer_in)))
                start_time = time.time()
                print(self.tracker_slow)
                if self.activated:  # avoid to continue the loop if not activated
                    for i in range(len(detection)):
                        if idx == 0:  # create multitracker only in the first frame, same detections
                            print('index cero buffer!')
                            self.tracker.add(cv2.TrackerTLD_create(), image, (
                                detection[i][1], detection[i][0], detection[i][3] - detection[i][1],
                                detection[i][2] - detection[i][0]))
                    _, boxes = self.tracker.update(image)
                    for i, newbox in enumerate(boxes):
                        p1 = (int(newbox[0]), int(newbox[1]))
                        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                        if len(self.color_list) == len(boxes):
                            cv2.rectangle(image, p1, p2, self.color_list[i], thickness=2)
                            cv2.putText(image, self.input_label[i], (p1[0], p1[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                        (0, 0, 0),
                                        thickness=2, lineType=2)
                    self.buffer_out.append(image)
                    fps_rate = 1.0 / (time.time() - start_time)
                    last_fps_buffer.pop(0)
                    last_fps_buffer.append(fps_rate)
                    avg_fps = sum(last_fps_buffer)/len(last_fps_buffer)
                    print(avg_fps)
                    if not(0 in last_fps_buffer) and avg_fps < 3:
                        counter += 1
                        if counter == 5:
                            print('Tracker slow!')
                            self.buffer_in = self.buffer_in[idx+20:len(self.buffer_in)]
                            if self.len_buffer_in - self.image_counter > self.buffer_step:
                                self.image_counter += self.buffer_step  # skip frames
                            else:
                                self.image_counter = self.len_buffer_in
                            counter = 0
                            self.tracker_slow = True
                    elif not(0 in last_fps_buffer) and self.tracker_slow == True:
                        self.tracker_slow = False

                    if avg_fps > 5.5:
                        self.tracker_fast = True

                else:
                    self.image_counter = 0  # reset when toggling modes (activated=False)

            self.new_detection = False
            if self.tracker_slow:  # ended tracking slow
                self.image_counter = self.len_buffer_in
            print('Tracking done!')

    def setBuffer(self, buf):
        self.buffer_in = buf[:-1]  # fixes overlap segmentation+tracking in net result
        self.len_buffer_in = len(self.buffer_in)
        self.buffer_step = int(self.len_buffer_in / 20)
        print('New buffer: ' +str(len(self.buffer_in)))

    def setInputDetection(self, bbox, state):
        self.input_detection = bbox
        self.new_detection = state

    def setInputLabel(self, label):
        self.input_label = label

    def setColorList(self, color_list):
        self.color_list = color_list

    def getOutputImage(self):

        if self.buffer_out and not self.tracker_slow and not self.tracker_fast:  # check if list is not empty
            self.image_counter += 1
            return self.buffer_out.pop(0)  # returns last detection and deletes it
        elif self.buffer_out and self.tracker_slow:
            return self.buffer_out.pop(0)  # image counter incremented in tracking
        elif self.buffer_out and self.tracker_fast:
            self.image_counter_fast += 1
            if self.image_counter_fast == 8:  # slow tracking a little, wait for network result
                self.image_counter_fast = 0
                self.image_counter += 1
                return self.buffer_out.pop(0)
            else:
                self.tracker_fast = False
        else:
            return None

    def checkProgress(self):
        print(self.image_counter)
        if (self.image_counter == self.len_buffer_in) or (len(self.buffer_in) == 0):
            self.toggleTracker()
            self.buffer_out = []
            self.image_counter = 0
            self.tracker_slow = False
            self.new_detection = False
            print('Tracker OFF')

    def toggleTracker(self):
        ''' Toggles the tracker on/off '''
        self.activated = not self.activated