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


def compute_mAP(generator, batch, input_size, all_detections, loc_preds, cls_preds):
        output_size = (1080, 1920)
        for i, (loc_pred, cls_pred) in enumerate(zip(loc_preds, cls_preds)):
            bboxes, cls_pred, score = BoxEncoder().decode(loc_pred, cls_pred, input_size, tf_box_order=conf.tf_box_order)
            if conf.tf_box_order:
                bboxes[:, [0, 1, 2, 3]] = bboxes[:, [1, 0, 3, 2]]
                input_size1 = (input_size[1],input_size[0])
                output_size1 = (output_size[1],output_size[0])

            input_scale  = list(input_size1) * 2
            output_scale = list(output_size1) * 2
            bboxes = bboxes/input_scale * output_scale
            bboxes = np.clip(bboxes[:,:], 0, output_scale).astype(int)

            image_detections = np.concatenate((bboxes, np.expand_dims(score, axis=1), np.expand_dims(cls_pred, axis=1)), axis=1)
            for label in range(generator.num_classes()):
                all_detections[i + batch*conf.batch_size][label] = image_detections[image_detections[:, -1] == label, :-1]
            #sys.stdout.write('process: [{}/{}]  used_time: {:.2f}ms\r'.format(i + 1, num_images, used_time))
            #sys.stdout.flush()
        return all_detections


if __name__=='__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    model_ckpt_path = '/workspace/tensorflow/object_det/Retinanet/retinanet-tensorflow/checkpoints/retinanet2_mojing/1cls_448x672_6ssd_a2_1branch_nop3/model_8.ckpt'
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
