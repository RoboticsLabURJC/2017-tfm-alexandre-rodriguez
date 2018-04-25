import threading
import cv2


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

    def track(self):

        detection = self.input_detection
        #print(self.new_detection)
        #print(len(self.buffer_in))
        if detection is not None and self.new_detection:  # new detection

            self.tracker = cv2.MultiTracker_create()

            for idx, image in enumerate(self.buffer_in):
                if self.activated:  # avoid to continue the loop if not activated
                    for i in range(len(detection)):
                        if idx == 0:  # create multitracker only in the first frame, same detections
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
                else:
                    self.image_counter = 0  # reset when toggling modes (activated=False)

            self.new_detection = False
            print('Tracking done!')


    def setBuffer(self, buf):
        self.buffer_in = buf
        print('New buffer: ' +str(len(buf)))

    def setInputDetection(self, bbox, state):
        self.input_detection = bbox
        self.new_detection = state

    def setInputLabel(self, label):
        self.input_label = label

    def setColorList(self, color_list):
        self.color_list = color_list

    def getOutputImage(self):
        if self.buffer_out:  # check if list is not empty
            self.image_counter += 1
            return self.buffer_out.pop(0)  # returns last detection and deletes it
        else:
            return None

    def checkProgress(self):
        if self.image_counter == len(self.buffer_in):
            self.toggleTracker()
            self.buffer_out = []
            self.image_counter = 0
            print('Tracker OFF')

    def toggleTracker(self):
        ''' Toggles the tracker on/off '''
        self.activated = not self.activated