# -*- encoding: utf8 -*-
# author: ronniecao
import sys
import os
import numpy
#import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import graph_util

import numpy as np
import cv2
import argparse

import time
import functools
import tensorflow.contrib.eager as tfe

from loss import loss_fn
#from loss_v2 import loss_fn
#from inputs import dataset_generator
from inputs_multi import dataset_generator
#from retinanet import RetinaNet         #retinanet1
from retinanet2.retinanet import RetinaNet    #retinanet2

from configuration import conf
from encoder import BoxEncoder

from mAP.data_generator import MyPascalVocGenerator
from mAP.eval import evaluate
#from mAP.train_mAP import evaluate



class ConvNet():

    def __init__(self, n_channel=3, num_class=conf.num_class, image_size=conf.input_size):
        self.n_channel = n_channel
        self.num_class = num_class
        self.image_size = image_size

        #channels_first
        #self.images = tf.placeholder(dtype=tf.float32, shape=[None, self.n_channel, self.image_size, self.image_size],name='images')
        self.images = tf.placeholder(dtype=tf.float32, shape=[None, self.image_size[0], self.image_size[1], self.n_channel],name='images')
        self.is_training = tf.placeholder(dtype=tf.bool, name='traing_mode')
        self.global_step = tf.Variable(0, dtype=tf.int64, name='global_step')

        model = RetinaNet()
        #self.loc_preds, self.cls_preds = model(self.images, training=self.is_training)     #retinanet1
        self.loc_preds, self.cls_preds = model(self.images, is_training=self.is_training)   #retinanet2
        self.d_bboxes, self.d_cls_pred, self.d_score = BoxEncoder().decode(self.loc_preds[0], self.cls_preds[0], self.image_size, tf_box_order=conf.tf_box_order)

    def main(self):
        # gpu_options = tf.GPUOptions(allow_growth=True)
        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.22
        self.sess = tf.Session(config=config)

        if not args.restore:
            self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(var_list=tf.global_variables(), write_version=tf.train.SaverDef.V2, max_to_keep=5)

        # for var in tf.global_variables():
        #     print(var)
        with self.sess.as_default():
            if args.restore:
                model_path = './checkpoints/retinanet2_mojing/1cls_448x672_5ssd_a4_1branch_nop3p7_data2_p047_n04_alph098_smooth10_tttttttttttttt/model_24.ckpt'
                self.saver.restore(self.sess, model_path)
                print("\n Restore  all weighes successful !!!")
            print("\n Building session !!! \n ")
            print(self.d_bboxes, self.d_cls_pred, self.d_score)

            constant_graph = graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, ['images', 'traing_mode', 'concat_12', 'GatherV2_6', 'GatherV2_8'])
            with tf.gfile.FastGFile(os.path.join(model_pb_dir, 'model.pb'), mode='wb') as f:
                f.write(constant_graph.SerializeToString())
            print("save as bp file over")

        self.sess.close()



if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Tensorflow RetinaNet Training')
    #parser.add_argument('--cuda-device', default=3, type=int, help='gpu device index')
    parser.add_argument('--grad_clip_norm', '-g', type=float)
    parser.add_argument('--restore', '-r', action='store_true', help='run continue ')
    parser.add_argument('--backbone_weight_restore', '-rbw', action='store_true', help='backbone_weight_restore')
    parser.add_argument('--recall_dir', dest='recall_dir', help='recall_dir',
                        default='./demo/recall_dir', type=str)

    args = parser.parse_args()
    ###################################################################################################################
    model_save_dir = "./checkpoints/retinanet2_mojing/1cls_448x672_5ssd_a4_1branch_nop3p7_data2_p047_n04_alph098_smooth10_tttttttttttttt"
    model_pb_dir = "./checkpoints/retinanet2_pb/1cls_448x672_5ssd_a4_1branch_nop3p7_data2_p047_n04_alph098_smooth10_tttttttttttttt"

    if not os.path.exists(model_pb_dir):
        os.mkdir(model_pb_dir)
    GPU_index = '1'
    print('\n==> ==> ==> Using device {}'.format(GPU_index))
    print("\n############################# save in << %s >>  and on << %s >>GPU ##############################\n"%(model_save_dir, GPU_index ))
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_index

    generator = MyPascalVocGenerator(mode='test')
    convnet = ConvNet(n_channel=3, num_class=conf.num_class, image_size=conf.input_size)
    convnet.main()
