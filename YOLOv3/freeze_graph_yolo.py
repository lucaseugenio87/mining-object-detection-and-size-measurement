import numpy as np
import tensorflow as tf
from utils.misc_utils import parse_anchors, read_class_names
from model import yolov3
from utils.nms_utils import gpu_nms

output_node_names = ['output/box_output',
                     'output/score_output',
                     'output/label_output',
                     'input_data']

output_directory = "/home/lucas/TensorFlow/workspace/training_demo/YOLOv3_TensorFlow-master/frozen_model_dataset_VGR/"
output_graph = "/home/lucas/TensorFlow/workspace/training_demo/YOLOv3_TensorFlow-master/frozen_model_dataset_VGR/frozen_model.pb"
clear_devices = True
meta_path = '/home/lucas/TensorFlow/workspace/training_demo/YOLOv3_TensorFlow-master/checkpoint2/best_model_Epoch_28_step_2320_mAP_0.4297_loss_111.7316_lr_0.0001.meta' # Your .meta file
anchor_path = '/home/lucas/TensorFlow/workspace/training_demo/YOLOv3_TensorFlow-master/data/yolo_anchors.txt'
class_name_path = '/home/lucas/TensorFlow/workspace/training_demo/YOLOv3_TensorFlow-master/data/coco_dataset_VGR.names'


new_size = [416, 416]
anchors = parse_anchors(anchor_path)
classes = read_class_names(class_name_path)
num_class = len(classes)


def freeze_graph():
    with tf.Session() as sess:
        input_data = tf.placeholder(tf.float32, [1, new_size[1], new_size[0], 3], name='input_data')
        yolo_model = yolov3(num_class, anchors)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

        pred_scores = pred_confs * pred_probs

        boxes, scores, labels = gpu_nms(pred_boxes,
                                        pred_scores,
                                        num_class, 
                                        max_boxes=200, 
                                        score_thresh=0.3, 
                                        nms_thresh=0.45)

        saver = tf.train.Saver()
        saver.restore(sess, meta_path[:-5])


        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(),
            #input_graph_def,
            output_node_names,
        )

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
            
freeze_graph()
