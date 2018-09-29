#coding=utf8
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- encoding: utf8 -*-
# author: xiepengyue
import os
import tensorflow as tf


#from sklearn.metrics import precision_score
#from sklearn.metrics import recall_score, accuracy_score
import numpy as np
import argparse
import time
import cv2
from PIL import Image, ImageDraw, ImageFont
from data_generator import MyPascalVocGenerator
from eval import evaluate
import sys
sys.path.append("..")
from encoder import BoxEncoder
from configuration import conf

num = 1

def label_to_name_map():
    # class_name = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    #                 'dog','horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

    #class_name = ( 'student','parent')
    class_name = ('person',)
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
    img_PIL = Image.open(img_path).convert('RGBA')
    bboxs = []
    water_print =Image.new('RGBA', img_PIL.size, (0,0,0,0))
    draw=ImageDraw.Draw(water_print)
    color_dict = color_map(label_2_name_map)
    for ind, b in enumerate(det_bboxes):
        #
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
        #draw.text((x_center-69, y_min-30), unicode(text_label,'utf-8'), font=fnt, fill=(0,0,0,255))
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
    result_img_path = os.path.join(result_dir, str(file_name)+'.jpg')
    global num
    #print(num, result_img_path)
    cv2.imwrite(result_img_path, result_img)
    num += 1
    return

def preprocess(img_path, input_size):
    if not os.path.exists(img_path):
        raise ValueError('Invalid img load error: {}'.format(img_path))
    image = cv2.imread(img_path)
    im_h, im_w, _ = image.shape
    output_size = (im_h, im_w)

    image = cv2.resize(image, (input_size[1], input_size[0]))#.astype('float32')
    b,g,r=cv2.split(image)
    image = cv2.merge([r,g,b])
    #image = image - np.array([123.68, 116.78, 103.94])
    image = image.astype(np.float32)*(1./255)

    if conf.channels_first:
        image = image.transpose((2, 0, 1))
    image_4D = np.expand_dims(image, axis=0)
    #image_4D = np.stack([image], axis=0)

    return image_4D, output_size

def inference(generator, model_ckpt_path):
    # gpu_options = tf.GPUOptions(allow_growth=True)
    # config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.25


    graph =tf.Graph()
    sess=tf.Session(graph=graph,config=config)
    with graph.as_default():
        #saver=tf.train.import_meta_graph('./checkpoints/test1/model_31350.ckpt.meta')
        #saver.restore(sess,'./checkpoints/test1/model_31350.ckpt')        #2cls_test_fl_alt_e3m

        saver=tf.train.import_meta_graph(model_ckpt_path + '.meta')
        saver.restore(sess, model_ckpt_path)

        input       = graph.get_tensor_by_name('images:0')
        is_training = graph.get_tensor_by_name('traing_mode:0')
        loc_preds   = tf.get_collection('output_tensor')[0]
        cls_preds   = tf.get_collection('output_tensor')[1]

        input_size = conf.input_size
        box_encoder = BoxEncoder()
        _bboxes, _cls_pred, _score = box_encoder.decode(loc_preds[0], cls_preds[0], input_size)

        label_2_name_map = label_to_name_map()
        print("\n>>>>>>>>>>>>>>Test<<<<<<<<<<<<<<<<<<<\n")

        num_images = generator.size()
        all_detections = [[None for i in range(generator.num_classes())] for j in range(num_images)]
        img_path_list = generator.get_imgpath_list()
        for i, img_path in enumerate(img_path_list):
            image_4D, output_size = preprocess(img_path, input_size)

            start_t = time.time()
            [bboxes, cls_pred, score] = sess.run(fetches=[_bboxes, _cls_pred, _score],
                                            feed_dict={input:image_4D, is_training:False})
            used_time = (time.time()-start_t) * 1000
            #print("used time: [{:.2f}]".format((time.time()-start_t) * 1000))
            #draw_bbox(result_dir, label_2_name_map, img_path, i, bboxes, cls_pred, score, input_size, output_size)


            # if conf.tf_box_order:
            #     bboxes[:, [0, 1, 2, 3]] = bboxes[:, [1, 0, 3, 2]]
            input_size1 = (input_size[1],input_size[0])
            output_size1 = (output_size[1],output_size[0])

            input_scale  = list(input_size1) * 2
            output_scale = list(output_size1) * 2
            bboxes = bboxes/input_scale * output_scale
            bboxes = np.clip(bboxes[:,:], 0, output_scale).astype(int)

            image_detections = np.concatenate((bboxes, np.expand_dims(score, axis=1), np.expand_dims(cls_pred, axis=1)), axis=1)
            for label in range(generator.num_classes()):
                all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]
            sys.stdout.write('process: [{}/{}]  used_time: {:.2f}ms\r'.format(i + 1, num_images, used_time))
            sys.stdout.flush()
        print('\n\n >>>> Is Computing , please waiting... >>>>\n')
        average_precisions = evaluate(generator, all_detections, iou_threshold=0.5)
        print_evaluation(average_precisions)

    sess.close()

def print_evaluation(average_precisions):
    # print evaluation
    present_classes = 0
    precision = 0
    for label, (average_precision, num_annotations) in average_precisions.items():
        print('{:.0f} instances of class'.format(num_annotations),
              generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
        if num_annotations > 0:
            present_classes += 1
            precision       += average_precision
    print('mAP: {:.4f}'.format(precision / present_classes))
    log_txt.write('test_set_num:{}   mAP: {:.4f}\n'.format(generator.size(), precision / present_classes))
    log_txt.close()

if __name__=='__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model_ckpt_path = '/workspace/tensorflow/object_det/Retinanet/retinanet-tensorflow/checkpoints/retinanet2_mojing/1cls_224x320_5ssd_a5_1b_nop7_data2_p055_n05_alph098_finetune/model_1.ckpt'
    txt_path = './log_result/mAP_log.txt'
    log_txt = open(txt_path, 'a')
    #num = len(log_txt.readlines())
    #log_txt.write('{}>>>>>'.format(int(num/2)+1))
    log_txt.write('<<<<<<<' + '_'.join(model_ckpt_path.split('/')[-2:]).split('.')[0] + '_nms_t:[{:2f}]_score_t:[{:2f}]:\n'.format(conf.nms_thred, conf.cls_thred))

    #image_test_dir = r'/home/pengyue/xpy/make_video/ph_demo/image'
    #image_test_dir ='/workspace/tensorflow/object_det/data/body_detection_data/songhui/JPEGImages/'
    #image_test_dir = '/workspace/tensorflow/object_det/data/body_detection_data/class/'

    #result_dir = './result_det/toG1_anchor5_crop03_5_nms04_s055'
    tf_box_order = True
    channels_first = False
    generator = MyPascalVocGenerator(mode='test')

    inference(generator, model_ckpt_path)
