import tensorflow as tf
import cv2
import numpy as np
import os


def load_image(image_name, image_height=416, image_width=416):
    #image = cv2.imread(image_name)
    image = cv2.resize(image_name, (image_height, image_width))[:, :, ::-1] / 255.
    #image = cv2.resize(image_name, (416, 416))[:, :, ::-1] / 255.
    image_exp = np.expand_dims(image, axis=0)
    #image_np = np.array(image, dtype=np.float32)
    return image_exp


def load_model(PATH_TO_FROZEN_GRAPH):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def run_inference(image_exp, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['output/box_output',
                        'output/score_output',
                        'output/label_output',
                        'input_data']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            image_tensor = tf.get_default_graph().get_tensor_by_name('input_data:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: image_exp})
            bbs = np.expand_dims(output_dict['output/box_output'], axis=0)
            scores = np.expand_dims(output_dict['output/score_output'], axis=0)
            classes = np.expand_dims(output_dict['output/label_output'], axis=0)
    return bbs, scores, classes

def draw_rectangles(image, bbs, scores, classes):

    a=0

    for i in classes:
        if scores[a] >= 0.5:
            xmin, ymin, xmax, ymax = bbs[a]
            (startX, startY, endX, endY) = (xmin, ymin, xmax, ymax)
            start_point = (int(startX), int(startY))
            end_point = (int(endX), int(endY))
            if i == 0:
                color = (0, 255, 0)
                cv2.rectangle(image, start_point, end_point, color, 2)
            '''if i == 1:
               color = (0, 0, 255)
            if i == 2:
                color = (255, 0, 0)
            if i == 3:
                color = (255, 255, 0)
            if i == 4:
                color = (0, 255, 255)'''
            #cv2.rectangle(image, start_point, end_point, color, 2)

        a=a+1

    return image

def draw_truth_rectangles(image, bbs):

    for box in bbs:
        xmin, ymin, xmax, ymax = box
        (startX, startY, endX, endY) = (xmin, ymin, xmax, ymax)
        start_point = (int(startX), int(startY))
        end_point = (int(endX), int(endY))
        color = (0, 0, 255)
        cv2.rectangle(image, start_point, end_point, color, 2)

    return image

def IoU(truth_boxes, predicted_boxes):

    boxes_truth = []
    boxes_predict = []

    true_positive_boxes = 0
    negative_boxes = 0

    for box_tuple in truth_boxes:
        xy = [(int(box_tuple[2]) + int(box_tuple[0])) / 2, (int(box_tuple[3]) + int(box_tuple[1])) / 2]
        wh = [(int(box_tuple[2]) - int(box_tuple[0])), (int(box_tuple[3]) - int(box_tuple[1]))]
        box = [xy, wh]
        boxes_truth.append(box)

    for p_box_tuple in predicted_boxes:
        pxy = [(int(p_box_tuple[2]) + int(p_box_tuple[0])) / 2, (int(p_box_tuple[3]) + int(p_box_tuple[1])) / 2]
        pwh = [(int(p_box_tuple[2]) - int(p_box_tuple[0])), (int(p_box_tuple[3]) - int(p_box_tuple[1]))]
        pbox = [pxy, pwh]
        boxes_predict.append(pbox)
        # boxes_truth.append(np.int0(box))

        #aux = aux + 1

    #cv2.rectangle(img_org, (box_tuple[0], box_tuple[1]), (box_tuple[2], box_tuple[3]), (0, 0, 255), 2)

    array_interArea = []
    array_IoU = []

    for t_box in boxes_truth:
        x_t_min = t_box[0][0] - (t_box[1][0] / 2)
        x_t_max = t_box[0][0] + (t_box[1][0] / 2)
        y_t_min = t_box[0][1] - (t_box[1][1] / 2)
        y_t_max = t_box[0][1] + (t_box[1][1] / 2)
        for p_box in boxes_predict:
            x_p_min = p_box[0][0] - (p_box[1][0] / 2)
            x_p_max = p_box[0][0] + (p_box[1][0] / 2)
            y_p_min = p_box[0][1] - (p_box[1][1] / 2)
            y_p_max = p_box[0][1] + (p_box[1][1] / 2)

            x_left = max(x_t_min, x_p_min)
            x_right = min(x_t_max, x_p_max)
            y_top = max(y_t_min, y_p_min)
            y_bottom = min(y_t_max, y_p_max)

            interArea = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)
            if interArea > 0:
                true_positive_boxes += 1

                array_interArea.append(interArea)

                fator_x_T = ((x_t_max - x_t_min) + 1.0)
                fator_y_T = ((y_t_max - y_t_min) + 1.0)

                fator_x_P = ((x_p_max - x_p_min) + 1.0)
                fator_y_P = ((y_p_max - y_p_min) + 1.0)

                boxTArea = ((x_t_max - x_t_min) + 1.0) * ((y_t_max - y_t_min) + 1.0)
                boxPArea = ((x_p_max - x_p_min) + 1.0) * ((y_p_max - y_p_min) + 1.0)

                IoU = interArea / float(boxTArea + boxPArea - interArea)
                array_IoU.append(IoU)

                #text_dim = "IoU: {:3.3}".format(IoU)
                #cv2.putText(img_org, text_dim, (int(x_t_min), int(y_t_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            #(0, 255, 0), 1)

                #print(text_dim)
            else:
                negative_boxes += 1

    return array_IoU, true_positive_boxes, negative_boxes

def read_boxes(file_path, path_img):

    count = 0
    truth_boxes = []
    with open(file_path) as file:
        #file_lines = file.readline()
        for line in file:
            boxes = line.split(' ')
            img_name = boxes[1].split(path_img)[1]
            truth_boxes.append(boxes)


    return truth_boxes

def boxes_array(boxes, path_img, img_name):

    boxes_coord = []

    for item in boxes:
        if img_name == item[1].split(path_img)[1]:
            for i in range(4,len(item),5):
                if int(item[i]) == 0:
                    coord = []
                    coord.append(item[i + 1])
                    coord.append(item[i + 2])
                    coord.append(item[i + 3])
                    coord.append(item[i + 4])
                    boxes_coord.append(coord)
            #index = index + 5
    return boxes_coord

def main():

    #file_path = "/home/lucas/TensorFlow/workspace/training_demo/YOLOv3_TensorFlow-master/data/my_data/test.txt"
    #path_img = "/home/lucas/TensorFlow/workspace/training_demo/image-segmentation-keras/label_quad/test_416/"
    #path_output = "/home/lucas/TensorFlow/workspace/training_demo/image-segmentation-keras/label_quad/IoU_416/"

    file_path = "/home/lucas/TensorFlow/workspace/training_demo/YOLOv3_TensorFlow-master/data/my_data/test_pedras.txt"
    path_img = "/home/lucas/TensorFlow/workspace/training_demo/YOLOv3_TensorFlow-master/data/my_data/images/test_416/"
    path_output = "/home/lucas/TensorFlow/workspace/training_demo/YOLOv3_TensorFlow-master/data/my_data/images/IoU_416/"

    all_truth_boxes = read_boxes(file_path, path_img)

    # Specify model file here
    #model_file = '/home/lucas/TensorFlow/workspace/training_demo/YOLOv3_TensorFlow-master/frozen_model_dataset_VGR/frozen_model.pb'
    model_file = '/home/lucas/TensorFlow/workspace/training_demo/YOLOv3_TensorFlow-master/frozen_model_dataset_bancada/frozen_model.pb'
    graph = load_model(model_file)

    IoU_full_array = []

    img_names = os.listdir(path_img)
    for name in img_names:
        frame = cv2.imread(path_img + name, cv2.IMREAD_UNCHANGED)
        image_exp = load_image(frame)
        bbs, scores, classes = run_inference(image_exp, graph)

        bbs_r = np.squeeze(bbs)
        scores_r = np.squeeze(scores)
        classes_r = np.squeeze(classes)

        boxes_xy = boxes_array(all_truth_boxes, path_img, name)

        IoU_array, TP_boxes, N_boxes = IoU(boxes_xy, bbs_r)
        img_IoU = [name, IoU_array]

        print("Image: {} - IoU: {:3.3} - True Positive: {:d}/{:d}".format(name, np.mean(IoU_array), TP_boxes, len(boxes_xy))) # IoU: {:3.3} True Positive: {:d}/{:d}".format(name, IoU, TP_boxes, len(boxes_xy)))

        IoU_full_array.append(img_IoU)

        images_with_boxes = draw_rectangles(frame, bbs_r, scores_r, classes_r)
        image_IoU = draw_truth_rectangles(images_with_boxes, boxes_xy)

        file_name = path_output + name

        cv2.imwrite(file_name, image_IoU)

    cv2.waitKey(0)

main()
#read_boxes()