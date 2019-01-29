import tensorflow as tf
import numpy as np
import cv2

from Net.utils import label_map_util

LABELS_DICT = {'voc': 'Net/labels/pascal_label_map.pbtxt',
               'coco': 'Net/labels/mscoco_label_map.pbtxt',
               'kitti': 'Net/labels/kitti_label_map.txt',
               'oid': 'Net/labels/oid_bboc_trainable_label_map.pbtxt',
               'pet': 'Net/labels/pet_label_map.pbtxt'}


class DetectionNetwork():
    def __init__(self, net_model):

        # attributes from dl-objecttracker network architecture
        self.input_image = None
        self.output_image = None
        self.activated = True
        self.detection = None
        self.label = None
        self.colors = None
        self.frame = None
        # new necessary attributes from dl-objectdetector network architecture
        self.original_height = None
        self.original_width = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.scale = 0.7
        COLORS = label_map_util.COLORS

        self.framework = "TensorFlow"
        self.net_has_masks = False

        labels_file = LABELS_DICT[net_model['Dataset'].lower()]
        label_map = label_map_util.load_labelmap(labels_file) # loads the labels map.
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes= 999999)
        category_index = label_map_util.create_category_index(categories)
        self.classes = {}
        # We build is as a dict because of gaps on the labels definitions
        for cat in category_index:
            self.classes[cat] = str(category_index[cat]['name'])

        # We create the color dictionary for the bounding boxes.
        self.net_classes = self.classes
        self.colors = {}
        idx = 0
        for _class in self.net_classes.values():
            self.colors[_class] = COLORS[idx]
            idx = + 1

        # Frozen inference graph, written on the file
        CKPT = 'Net/TensorFlow/' + net_model['Model']
        detection_graph = tf.Graph() # new graph instance.
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Set additional parameters for the TF session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config = tf.ConfigProto(gpu_options=gpu_options,
                                log_device_placement=False)
        self.sess = tf.Session(graph=detection_graph, config=config)
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # NCHW conversion. not possible
        #self.image_tensor = tf.transpose(self.image_tensor, [0, 3, 1, 2])
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        for op in detection_graph.get_operations():
            if op.name == 'detection_masks':
                self.net_has_masks = True
                self.detection_masks = detection_graph.get_tensor_by_name('detection_masks:0')

        self.scores = []
        self.predictions = []

        # Dummy initialization (otherwise it takes longer then)
        dummy_tensor = np.zeros((1,1,1,3), dtype=np.int32)
        if self.net_has_masks:
            self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections, self.detection_masks],
                feed_dict={self.image_tensor: dummy_tensor})
        else:
            self.sess.run(
                    [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                    feed_dict={self.image_tensor: dummy_tensor})

        self.confidence_threshold = 0.5

        print("Network ready!")

    def predict(self):
        input_image = self.input_image
        if input_image is not None:
            self.activated = False
            image_np_expanded = np.expand_dims(input_image, axis=0)
            if self.net_has_masks:
                (boxes, scores, predictions, _, masks) = self.sess.run(
                    [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections, self.detection_masks],
                    feed_dict={self.image_tensor: image_np_expanded})
                print('net masks')
                print(masks)
            else:
                (boxes, scores, predictions, _) = self.sess.run(
                    [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                    feed_dict={self.image_tensor: image_np_expanded})

            # We only keep the most confident predictions.
            conf = scores > self.confidence_threshold  # bool array
            boxes = boxes[conf]
            # aux variable for avoiding race condition while int casting
            tmp_boxes = np.zeros([len(boxes), 4]).astype(int)
            tmp_boxes[:, [0]] = boxes[:, [1]] * self.original_width
            tmp_boxes[:, [2]] = boxes[:, [3]] * self.original_width
            tmp_boxes[:, [3]] = boxes[:, [2]] * self.original_height
            tmp_boxes[:, [1]] = boxes[:, [0]] * self.original_height
            self.detection = tmp_boxes
            self.scores = scores[conf]
            predictions = predictions[conf].astype(int)
            self.label = []
            for pred in predictions:
                self.label.append(self.classes[pred])

            if self.net_has_masks:
                #from Net.utils import visualization_utils
                #print('draw mask')
                #visualization_utils.draw_mask_on_image_array(self.input_image, masks)
                #self.display_instances(self.input_image, self.detection, masks, self.label, self.scores)
                detected_image = self.renderModifiedImage()
            else:
                detected_image = self.renderModifiedImage()
            zeros = False
            print('Image segmented!')

        else:
            detected_image = np.array(np.zeros((480, 320), dtype=np.int32))
            zeros = True

        self.output_image = [detected_image, zeros]


    def renderModifiedImage(self): # from utils visualize of Tensorflow folder
        image_np = np.copy(self.input_image)

        detection_boxes = self.detection
        detection_classes = self.label
        detection_scores = self.scores

        for index in range(len(detection_classes)):
            _class = detection_classes[index]
            score = detection_scores[index]
            rect = detection_boxes[index]
            xmin = rect[0]
            ymin = rect[1]
            xmax = rect[2]
            ymax = rect[3]
            cv2.rectangle(image_np, (xmin, ymax), (xmax, ymin), self.colors[_class], 3)

            label = "{0} ({1} %)".format(_class, int(score*100))
            [size, base] = cv2.getTextSize(label, self.font, self.scale, 2)

            points = np.array([[[xmin, ymin + base],
                                [xmin, ymin - size[1]],
                                [xmin + size[0], ymin - size[1]],
                                [xmin + size[0], ymin + base]]], dtype=np.int32)
            cv2.fillPoly(image_np, points, (0, 0, 0))
            cv2.putText(image_np, label, (xmin, ymin), self.font, self.scale, (255, 255, 255), 2)

        return image_np

    def random_colors(self, N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        import colorsys
        import random
        brightness = 1.0 if bright else 0.7
        hsv = [(float(i) / float(N), 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def apply_mask(self, image, mask, color, alpha=0.5):
        """Apply the given mask to the image.
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image

    def display_instances(self, image, boxes, masks, class_names,
                          scores=None, title="",
                          figsize=(16, 16), ax=None):  # from mask rcnn visualize.py
        """
        boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
        masks: [height, width, num_instances]
        class_ids: [num_instances]
        class_names: list of class names of the dataset
        scores: (optional) confidence scores for each box
        figsize: (optional) the size of the image.
        """
        from skimage.measure import find_contours

        # Number of instances
        N = boxes.shape[0]

        # if not N:
        #     print("\n*** No instances to display *** \n")
        # else:
        #     assert boxes.shape[0] == masks.shape[-1]

        # Generate random colors
        colors = self.random_colors(N)
        color_list = []
        masked_image = image.astype(np.uint8).copy()
        for i in range(N):
            color = colors[i]

            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            cv2.rectangle(masked_image, (x1, y1), (x2, y2), thickness=2, lineType=8, color=(255, 255, 255))

            # Label
            score = scores[i] if scores is not None else None
            label = class_names[i]
            caption = "{} {:.3f}".format(label, score) if score else label
            cv2.putText(masked_image, caption, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2,
                        lineType=2)

            # Mask
            mask = masks[:, :, i]
            masked_image = self.apply_mask(masked_image, mask, color)

            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            edge_color = tuple([255 * x for x in color])
            for i, verts in enumerate(contours):
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                verts = np.array(verts).reshape(-1, 1, 2).astype(np.int32)
                cv2.drawContours(masked_image, verts, -1, edge_color)
            color_list.append(edge_color)

        return masked_image.astype(np.uint8), color_list

    def setInputImage(self, im, frame_number):
        ''' Sets the input image of the network. '''
        self.input_image = im
        self.frame = frame_number
        self.original_height = im.shape[0]
        self.original_width = im.shape[1]

    def getOutputImage(self):
        ''' Returns the image with the segmented objects on it. '''
        return self.output_image

    def getProcessedFrame(self):
        ''' Returns the index of the frame processed by the net. '''
        return self.frame

    def getOutputDetection(self):
        ''' Returns the bounding boxes. '''
        return self.detection

    def getOutputLabel(self):
        ''' Returns the labels. '''
        return self.label

    def getColorList(self):
        ''' Returns the colors for the bounding boxes. '''
        return self.colors

    def toggleNetwork(self):
        ''' Toggles the network (on/off). '''
        self.activated = not self.activated