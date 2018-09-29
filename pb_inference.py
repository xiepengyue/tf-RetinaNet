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


num = 1

def label_to_name_map():
    # class_name = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    #                 'dog','horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

    #class_name = ( 'student','parent')
    class_name = ('per',)
    label_2_name_map = { k:v for k, v in enumerate(class_name, start=0)}
    return label_2_name_map

def draw_bbox(result_dir, label_2_name_map, img_path, file_name, det_bboxes, cls_pred, scores, input_size, output_size):

    def color_map(label_2_name_map):
        color_step = int(250/len(label_2_name_map))
        # for i in range(0, 255, color_step):
        #     pass
        #color_dict  = { k:( 255 - color_step*k , 150, color_step*k, 100) for k in label_2_name_map.keys()}
        color_dict = {0: (255, 0, 0, 100)}
        return color_dict

    img_H, img_W = output_size[:]

    #frame_H, frame_W, frame_C = frame_img.shape[:]
    img_PIL = Image.open(img_path).convert('RGBA')
    bboxs = []
    water_print =Image.new('RGBA', img_PIL.size, (0,0,0,0))
    draw=ImageDraw.Draw(water_print)
    color_dict = color_map(label_2_name_map)

    #print(img_W, img_H)
    #print(det_bboxes)

    for ind, b in enumerate(det_bboxes):

        # if tf_box_order:
        #     b[0], b[1], b[2], b[3] = b[1], b[0], b[3], b[2]

        x1 = np.clip(int(b[0]/input_size[1] * img_W), 0, img_W)
        y1 = np.clip(int(b[1]/input_size[0] * img_H), 0, img_H)
        x2 = np.clip(int(b[2]/input_size[1] * img_W), 0, img_W)
        y2 = np.clip(int(b[3]/input_size[0] * img_H), 0, img_H)

        x_min, y_min, x_max, y_max = x1, y1, x2, y2

        #print(x_min, y_min, x_max, y_max)
        pred_cls = cls_pred[ind]
        #print("#########3   ", pred_cls)
        #print(label_2_name_map[pred_cls])
        text_label = '{}:{:.2f}'.format(label_2_name_map[pred_cls], scores[ind])
        rec_color = color_dict[pred_cls]
    ##################################################################################
        fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 23)
        draw.rectangle((x_min, y_min-28, x_max, y_min),fill = rec_color)
        x_center = (x_max+x_min)/2
        #draw.text((x_center-69, y_min-30), unicode(text_label,'utf-8'), font=fnt, fill=(0,0,0,255))
        draw.text((x_center-69, y_min-30), text_label, font=fnt, fill=(0,0,0,255))
        #draw.text((x_center-69, y_min-30), text_label,  fill=(0,0,0,255))
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
    config.gpu_options.per_process_gpu_memory_fraction = 0.2


    graph =tf.Graph()
    sess=tf.Session(graph=graph,config=config)
    with graph.as_default():

        model_pb_path = './checkpoints/retinanet2_pb/1cls_448x672_5ssd_a4_1branch_nop3p7_data2_p047_n04_alph098_smooth10_tttttttttttttt/model.pb'
        output_graph_def = tf.GraphDef()
        with open(model_pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        input       = graph.get_tensor_by_name('images:0')
        is_training = graph.get_tensor_by_name('traing_mode:0')
        _bboxes = graph.get_tensor_by_name('concat_12:0')
        _cls_pred = graph.get_tensor_by_name('GatherV2_6:0')
        _score = graph.get_tensor_by_name('GatherV2_8:0')

        input_size = (224, 320)

        label_2_name_map = label_to_name_map()
        print("\n>>>>>>>>>>>>>>Test<<<<<<<<<<<<<<<<<<<\n")


        # for file_name in os.listdir(image_test_dir):

#######################
        # image_test_dir ='/workspace/tensorflow/object_det/data/body_detection_data/mirror/spring/v0_JPEGImages'
        # spring_test = "/workspace/tensorflow/object_det/data/body_detection_data/mirror/spring/test.txt"
        image_test_dir ='/workspace/tensorflow/object_det/data/det_img_test/image'
        spring_test = "/workspace/tensorflow/object_det/data/det_img_test/test.txt"
        with open(spring_test, 'r') as f:
            lines = f.readlines()
            imgname_list = []
            for i in range(0, len(lines)):
                imgname = lines[i].rstrip('\n').split(' ')[-1]
                imgname_list.append(imgname)
            f.close()
        result_dir = './inference/mix_24_1cls_224x320_5ssd_a4_1branch_nop7_data2_p047_n04_nms04_s07_ttttt'
        #np.random.shuffle(imgname_list)
        for file_name in imgname_list:
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

            if channels_first:
                image = image.transpose((2, 0, 1))
            image_4D = np.expand_dims(image, axis=0)
            #image_4D = np.stack([image], axis=0)

    # while True:
    #     try:
    #         image_4D, loc_trues, cls_trues = sess.run(dataset)
            start_t = time.time()
            [bboxes, cls_pred, score] = sess.run(fetches=[_bboxes, _cls_pred, _score],
                                            feed_dict={input:image_4D, is_training:False})
            print("used time: [{:.2f}]".format((time.time()-start_t) * 1000))
            print(" ~~~~~~~~~inference over ~~~~~~~~~~~~~~~~~")
            #print(bboxes)
            #print(cls_pred)
            #print(score)

            draw_bbox(result_dir, label_2_name_map, img_path, file_name, bboxes, cls_pred, score, input_size, output_size)
            print("draw over")
        # except tf.errors.OutOfRangeError:
        #     print(">>>>  ave_loss: [{:.3f}]  <<<<<<<".format(1))
        #     return
    sess.close()

if __name__=='__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    #image_test_dir = r'/home/pengyue/xpy/make_video/ph_demo/image'
    image_test_dir ='/workspace/tensorflow/object_det/data/body_detection_data/songhui/JPEGImages/'
    #image_test_dir = '/workspace/tensorflow/object_det/data/body_detection_data/class/'

    result_dir = './inference/songhui_1cls_448x672_a2_crop03_p05_n033_nms035_s075_myloss'
    tf_box_order = True
    channels_first = False
    #image_test_dir = r'/home/pengyue/pytorch/ssd_pytorch/data/VOCdevkit/VOC2007/JPEGImages'
    #result_dir = './inference/image_voc2007'
    evalution()
