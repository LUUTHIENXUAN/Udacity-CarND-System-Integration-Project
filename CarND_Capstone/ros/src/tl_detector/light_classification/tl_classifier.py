from styx_msgs.msg import TrafficLight
import os
import warnings
import tarfile
import six.moves.urllib as urllib
import numpy as np
import tensorflow as tf
from distutils.version import LooseVersion

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class TLClassifier(object):

    def __init__(self):

        self.traffic_light        = TrafficLight.UNKNOWN
        self.condition_check      = False
        self.maybe_download_model = False
        self.mode                 = "sim"

        """
        Pretrained model by TensorFlow
        """
        #base path where we will save our models
        self.PATH_TO_MODEL = '/home/udacity/Final_project/Traffic_light/traffic_light_model/'

        # Specify Model To Download. Obtain the model name from the object detection model zoo.
        self.MODEL_NAME    = 'ssd_inception_v2_coco_2017_11_17'
        self.MODEL_FILE    = self.MODEL_NAME + '.tar.gz'
        self.DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'


        # Load the Labels

        PATH_TO_LABELS = self.PATH_TO_MODEL+'color_label.pbtx'
        NUM_CLASSES    = 4

        label_map           = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories          = label_map_util.convert_label_map_to_categories(label_map,
                                                                             max_num_classes  = NUM_CLASSES,
                                                                             use_display_name = True)
        self.category_index = label_map_util.create_category_index(categories)

        ##### Build network

        # GPU
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # TODO: load classifier
        # http://insightsbot.com/blog/womeQ/tensorflow-object-detection-tutorial-on-images
        self.detection_graph = tf.Graph()

        if self.mode == "sim":
            PATH_TO_CKPT = './light_classification/frozen_inference_graph_sim.pb'
        else:
            PATH_TO_CKPT = './light_classification/frozen_inference_graph_real.pb'

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.sess           = tf.Session(graph=self.detection_graph, config=config)

        self.image_tensor   = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.boxes          = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.scores         = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes        = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')


    def check_condition(self):

        # Check TensorFlow Version
        assert LooseVersion(tf.__version__) != LooseVersion('1.3'), 'Please use Udacity default TensorFlow version 1.3.  You are using {}'.format(tf.__version__)
        print('TensorFlow Version: {}'.format(tf.__version__))

        # Check for a GPU
        if not tf.test.gpu_device_name():
            warnings.warn('No GPU found. Please use a GPU to train your neural network.')
        else:
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

        self.condition_check  = True

    def maybe_download_pretrained_vgg(self):

        """
        Download and extract pretrained model if it doesn't exist
        """
        DESTINATION_MODEL_TAR_PATH = self.PATH_TO_MODEL + self.MODEL_FILE

        # Check if model downloaded
        if not os.path.exists (self.PATH_TO_MODEL + self.MODEL_NAME + '/frozen_inference_graph.pb'):

            # Download model
            print 'Downloading pre-trained model...'
            opener = urllib.request.URLopener()
            opener.retrieve(self.DOWNLOAD_BASE + self.MODEL_FILE, DESTINATION_MODEL_TAR_PATH)

            # Extracting model
            print 'Extracting model...'
            tar_file = tarfile.open(DESTINATION_MODEL_TAR_PATH)

            for file in tar_file.getmembers():
                file_name = os.path.basename(file.name)
                if 'frozen_inference_graph.pb' in file_name:
                    tar_file.extract(file, self.PATH_TO_MODEL)

        self.maybe_download_model = True

    def get_classification(self, image):
        """
        Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        if not self.condition_check: self.check_condition()

        if not self.maybe_download_model: self.maybe_download_pretrained_vgg()

        # add dimension to feed classifier
        feeded_image = np.expand_dims(image, axis=0)

        with self.detection_graph.as_default():
            (boxes, scores, classes, num)= self.sess.run([self.boxes,
                                                          self.scores,
                                                          self.classes,
                                                          self.num_detections],
                                                          feed_dict={self.image_tensor: feeded_image})


        # remove unnessary dimension
        boxes   = np.squeeze(boxes)
        scores  = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        predicted_traffic_light = "UNKNOWN"

        highest_score = 0.0

        for i in range(boxes.shape[0]):
            # choose result with confidence over 60%
            if scores[i] > 0.6 and highest_score < scores[i]:
                # update traffic light
                predicted_traffic_light = self.category_index[classes[i]]['name']

        print predicted_traffic_light

        if predicted_traffic_light == "GREEN":
            self.traffic_light = TrafficLight.GREEN

        elif predicted_traffic_light == "YELLOW":
            self.traffic_light = TrafficLight.YELLOW

        elif predicted_traffic_light == "RED":
            self.traffic_light = TrafficLight.RED

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(image, boxes,
        classes, scores, self.category_index, use_normalized_coordinates=True,
        line_thickness=8)

        return self.traffic_light
