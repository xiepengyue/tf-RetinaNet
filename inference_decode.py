#coding=utf8
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- encoding: utf8 -*-
# author: xiepengyue
import sys
import os
import tensorflow as tf
from encoder import BoxEncoder
from inputs import dataset_generator

#from sklearn.metrics import precision_score
#from sklearn.metrics import recall_score, accuracy_score
import numpy as np
import argparse
import time
import cv2
from PIL import Image, ImageDraw, ImageFont
from configuration import conf

num = 1

def label_to_name_map():
    class_name = ('person',)
    label_2_name_map = { k:v for k, v in enumerate(class_name, start=0)}
    return label_2_name_map

def draw_bbox(result_dir, label_2_name_map, img_path, file_name, det_bboxes, cls_pred, scores, input_size, output_size):
    def color_map(label_2_name_map):
        color_step = int(250/len(label_2_name_map))
        color_dict = {0: (255, 0, 0, 100)}
        return color_dict

    img_H, img_W = output_size[:]
    img_PIL = Image.open(img_path).convert('RGBA')
    bboxs = []
    water_print =Image.new('RGBA', img_PIL.size, (0,0,0,0))
    draw=ImageDraw.Draw(water_print)
    color_dict = color_map(label_2_name_map)

    for ind, b in enumerate(det_bboxes):
        # if tf_box_order:
        #     b[0], b[1], b[2], b[3] = b[1], b[0], b[3], b[2]
        x1 = np.clip(int(b[0]/input_size[1] * img_W), 0, img_W)
        y1 = np.clip(int(b[1]/input_size[0] * img_H), 0, img_H)
        x2 = np.clip(int(b[2]/input_size[1] * img_W), 0, img_W)
        y2 = np.clip(int(b[3]/input_size[0] * img_H), 0, img_H)

        x_min, y_min, x_max, y_max = x1, y1, x2, y2
        pred_cls = cls_pred[ind]
        text_label = '{}:{:.2f}'.format(label_2_name_map[pred_cls], scores[ind])
        rec_color = color_dict[pred_cls]
    ##################################################################################
        fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 23)
        draw.rectangle((x_min, y_min-28, x_max, y_min),fill = rec_color)
        x_center = (x_max+x_min)/2
        draw.text((x_center-69, y_min-30), text_label, font=fnt, fill=(0,0,0,255))
        bboxs.append([x_min, y_min, x_max, y_max, rec_color]) #x1, y1, x2, y2,  color is a tuple such as: (0,255,255,110)
    ##################################################################################################3
    img_PIL = Image.alpha_composite(img_PIL, water_print)
    result_img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)

    for bbox in bboxs:
        x1, y1, x2, y2, color = bbox[:]
        rec_color = (color[2],color[1],color[0])
        cv2.rectangle(result_img, (x1, y1), (x2, y2), rec_color, 2)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    result_img_path = os.path.join(result_dir, file_name)
    global num
    print(num, result_img_path)
    cv2.imwrite(result_img_path, result_img)
    num += 1
    return

def evalution():
    # gpu_options = tf.GPUOptions(allow_growth=True)
    # config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.25

    graph =tf.Graph()
    sess=tf.Session(graph=graph,config=config)
    with graph.as_default():

        model_ckpt_path = './checkpoints/retinanet2_mojing2/1cls_224x384_6ssd_a7_1b_nop7_p045_n04_alph09_Res50_rbw/model_12.ckpt'
        saver=tf.train.import_meta_graph(model_ckpt_path+'.meta')
        saver.restore(sess,model_ckpt_path)

        input       = graph.get_tensor_by_name('images:0')
        is_training = graph.get_tensor_by_name('traing_mode:0')
        loc_preds   = tf.get_collection('output_tensor')[0]
        cls_preds   = tf.get_collection('output_tensor')[1]

        input_size = (224, 384)
        box_encoder = BoxEncoder()
        _bboxes, _cls_pred, _score = box_encoder.decode(loc_preds[0], cls_preds[0], input_size)
        label_2_name_map = label_to_name_map()
        print("\n>>>>>>>>>>>>>>Test<<<<<<<<<<<<<<<<<<<\n")


        for file_name in os.listdir(image_test_dir):

#######################
        # # image_test_dir ='/workspace/tensorflow/object_det/data/body_detection_data/mirror/spring/v0_JPEGImages'
        # # spring_test = "/workspace/tensorflow/object_det/data/body_detection_data/mirror/spring/test.txt"
        # image_test_dir ='/workspace/tensorflow/object_det/data/det_img_test/image'
        # spring_test = "/workspace/tensorflow/object_det/data/det_img_test/test.txt"
        # with open(spring_test, 'r') as f:
        #     lines = f.readlines()
        #     imgname_list = []
        #     for i in range(0, len(lines)):
        #         imgname = lines[i].rstrip('\n').split(' ')[-1]
        #         imgname_list.append(imgname)
        #     f.close()
        # result_dir = './inference2/mix_12_1cls_224x384_6ssd_a7_1b_nop7_p045_n04_alph09_Res50_rbw_nms04_s07'
        # #np.random.shuffle(imgname_list)
        # for file_name in imgname_list:
##################

            img_path = os.path.join(image_test_dir, file_name)
            if not os.path.exists(img_path):
                print("img load error!!")
                continue
            image = cv2.imread(img_path)
            im_h, im_w, _ = image.shape
            output_size = (im_h, im_w)

            image = cv2.resize(image, (input_size[1], input_size[0]))#.astype('float32')
            b,g,r=cv2.split(image)
            image = cv2.merge([r,g,b])
            #image = image - np.array([123.68, 116.78, 103.94])
            image = image.astype(np.float32)*(1./255)
            if conf.net == 'ShuffleNetV2':
                image = (2.0 * image) - 1.0
            if channels_first:
                image = image.transpose((2, 0, 1))
            image_4D = np.expand_dims(image, axis=0)


            start_t = time.time()
            [bboxes, cls_pred, score] = sess.run(fetches=[_bboxes, _cls_pred, _score],
                                            feed_dict={input:image_4D, is_training:False})
            print("used time: [{:.2f}]".format((time.time()-start_t) * 1000))
            print(" ~~~~~~~~~inference over ~~~~~~~~~~~~~~~~~")

            draw_bbox(result_dir, label_2_name_map, img_path, file_name, bboxes, cls_pred, score, input_size, output_size)
            print("draw over")
    sess.close()

if __name__=='__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    image_test_dir ='/workspace/tensorflow/object_det/data/body_detection_data/songhui/JPEGImages/'
    result_dir = './inference/songhui_1cls_448x672_a2_crop03_p05_n033_nms035_s075_myloss'

    tf_box_order = True
    channels_first = False

    evalution()
