"""
该文件主要是将ckpt文件转化为pb文件
"""

"""
确定输出节点名称：
    yolohead()中的返回值： box_xy, box_wh, box_confidence, box_class_probs

尝试是否可以将预测后的结果--->nms---->letterbox，之后进行输出
"""
import os
import config
import argparse
import numpy as np
import tensorflow as tf
from yolo_predict import yolo_predictor
from PIL import Image, ImageFont, ImageDraw
from utils import letterbox_image, load_weights

"""
搭建图
"""
def detect(image_path, model_path, yolo_weights = None):
    """
    Introduction
    ------------
        加载模型，进行预测
    Parameters
    ----------
        model_path: 模型路径
        image_path: 图片路径
    """
    image = Image.open(image_path)
    resize_image = letterbox_image(image, (416, 416))
    image_data = np.array(resize_image, dtype = np.float32)
    image_data /= 255.
    image_data = np.expand_dims(image_data, axis = 0)

    pb_graph = tf.Graph()
    with pb_graph.as_default():
        input_image_shape = tf.placeholder(dtype = tf.int32, shape = (2,), name="pred_im_shape")
        input_image = tf.placeholder(shape = [None, 416, 416, 3], dtype = tf.float32, name='pred_input_img')
        predictor = yolo_predictor(config.obj_threshold, config.nms_iou_threshold, config.classes_path, config.anchors_path)
        boxes, scores, classes = predictor.predict(input_image, input_image_shape)
        print(input_image_shape)
        print(input_image)
        print(boxes)
        print(scores)
        print(classes)
    with tf.Session(graph=pb_graph) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        out_boxes, out_scores, out_classes = sess.run([boxes, scores, classes],
                                        feed_dict={ input_image: image_data,
                                                    input_image_shape: [image.size[1], image.size[0]]
                                                    }
                                                    )
        graph_def = tf.get_default_graph().as_graph_def()
        out_put_name_list = ['predict/pred_boxes', 'predict/pred_scores', 'predict/pred_classes']
        out_put_grah_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, out_put_name_list)
        pb_file_path = 'F:\\github_working\\version_2_190114\\alsochen-tensorflow-yolo3-threeoutput\\tensorflow-yolo3\\pb_file\\model.pb'
        with tf.gfile.GFile(pb_file_path,'wb') as f:
            f.write(out_put_grah_def.SerializeToString())
            print("pb save done")
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))



if __name__ == '__main__':
    """
    parser = argparse.ArgumentParser(argument_default = argparse.SUPPRESS)
    parser.add_argument(
        '--image_file', type = str, help = 'image file path'
    )
    FLAGS = parser.parse_args()
    if config.pre_train_yolo3 == True:
        detect(FLAGS.image_file, config.model_dir, config.yolo3_weights_path)
    else:
        detect(FLAGS.image_file, config.model_dir)
    """
    flag = False
    if flag:
        annotation_path = "F:\\deeplearning_dataset\\new_ribbon\\split_imge\\split_img_annotation.txt"
        image_path = "F:\\deeplearning_dataset\\new_ribbon\\split_imge"
        model_path = "F:\\github_working\\version_2_190114\\alsochen-tensorflow-yolo3-threeoutput\\tensorflow-yolo3\\checkpoint\\original_net\\model.ckpt-3999"
        batch_detect(annotation_path, image_path, model_path)
    else:
        image_path = "F:\\deeplearning_dataset\\new_ribbon\\split_imge\\1109_(98)_1.jpg"
        model_path = "F:\\github_working\\version_2_190114\\alsochen-tensorflow-yolo3-threeoutput\\tensorflow-yolo3\\checkpoint\\net_changged\\model.ckpt-3870"
        detect(image_path, model_path)