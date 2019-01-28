import os
import config
import argparse
import numpy as np
import colorsys
import tensorflow as tf
from yolo_predict import yolo_predictor
from PIL import Image, ImageFont, ImageDraw
from utils import letterbox_image, load_weights

with tf.Session() as sess:
    image_path = "F:\\deeplearning_dataset\\new_ribbon\\split_imge\\1109_(98)_1.jpg"
    ##########数据准备阶段################
    image = Image.open(image_path)
    resize_image = letterbox_image(image, (416, 416))
    image_data = np.array(resize_image, dtype = np.float32)
    image_data /= 255.
    image_data = np.expand_dims(image_data, axis = 0)
    #####################################

    pb_file_path = 'F:\\github_working\\version_2_190114\\alsochen-tensorflow-yolo3-threeoutput\\tensorflow-yolo3\\pb_file\\model.pb'
    with tf.gfile.GFile(pb_file_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        pred_im_shape, pred_input_img, boxes, scores, classes = tf.graph_util.import_graph_def(graph_def, return_elements=['pred_im_shape:0', 'pred_input_img:0', 'predict/pred_boxes:0', 'predict/pred_scores:0', 'predict/pred_classes:0'])
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                pred_input_img: image_data,
                pred_im_shape: [image.size[1], image.size[0]]
            })
        print("class:\n")
        print(out_classes)
        print("class done\n")
        print([out_boxes, out_scores, out_classes])

