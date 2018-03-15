import threading
import cv2


class Tracker:
    def __init__(self):

        self.tracker = cv2.MultiTracker_create()
        print("Tracker created!")

        self.lock = threading.Lock()

        self.activated = True

        self.input_detection = None
        self.input_image = None
        self.input_label = None
        self.output_image = None
        self.last_detection = None
        self.color_list = None

    def track(self):

        detection = self.input_detection
        image = self.input_image

        if detection is not None and ((detection == self.last_detection).all()): # detecciones coinciden, solo actualiza bbox
            _, boxes = self.tracker.update(image)
            for i, newbox in enumerate(boxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                if len(self.color_list) == len(boxes):
                    cv2.rectangle(image, p1, p2, self.color_list[i], thickness=2)
                    cv2.putText(image, self.input_label[i], (p1[0], p1[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 0, 0), thickness=2, lineType=2)
                else:
                    cv2.rectangle(image, p1, p2, (255, 0, 0), thickness=2)

        elif detection is not None and ((detection != self.last_detection).all()): # detecciones nuevas, reinicia multitracker
            self.tracker = cv2.MultiTracker_create()
            for i in range(len(detection)):
                self.tracker.add(cv2.TrackerTLD_create(), image, (
                detection[i][1], detection[i][0], detection[i][3] - detection[i][1], detection[i][2] - detection[i][0]))
            _, boxes = self.tracker.update(image)
            for i, newbox in enumerate(boxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(image, p1, p2, self.color_list[i], thickness=2)
                cv2.putText(image, self.input_label[i], (p1[0], p1[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0),
                            thickness=2, lineType=2)
            self.last_detection = detection

        self.output_image = image

    def setInputImage(self, im):
        self.input_image = im

    def setInputDetection(self, bbox):
        self.input_detection = bbox

    def setInputLabel(self, label):
        self.input_label = label

    def setColorList(self, color_list):
        self.color_list = color_list

    def getOutputImage(self):
        return self.output_image

    def toggleTracker(self):
        ''' Toggles the tracker on/off '''
        self.activated = not self.activated