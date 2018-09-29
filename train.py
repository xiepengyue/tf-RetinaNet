# -*- encoding: utf8 -*-
# author: ronniecao
import sys
import os
import numpy
#import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorboardX import SummaryWriter

#from utils.utils import progress_bar
#from sklearn.metrics import precision_score
#from sklearn.metrics import recall_score, accuracy_score
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

from tensorflow.python import pywrap_tensorflow
from configuration import conf
from encoder import BoxEncoder

from mAP.data_generator import MyPascalVocGenerator
from mAP.eval import evaluate, get_annotations
#from mAP.train_mAP import evaluate



class ConvNet():

    def __init__(self, n_channel=3, num_class=conf.num_class, image_size=conf.input_size):
        self.n_channel = n_channel
        self.num_class = num_class
        self.image_size = image_size

        self.images = tf.placeholder(dtype=tf.float32, shape=[None, self.image_size[0], self.image_size[1], self.n_channel],name='images')
        num_bbox = 4
        if conf.use_secondbig_loss_constrain:
            num_bbox = 8
        self.loc_trues = tf.placeholder(dtype=tf.float32, shape=[None, None, num_bbox], name='loc_target')  ##############
        self.cls_trues = tf.placeholder(dtype=tf.float32, shape=[None, None], name='cls_target')  ##############
        self.is_training = tf.placeholder(dtype=tf.bool, name='traing_mode')

        self.global_step = tf.Variable(0, dtype=tf.int64, name='global_step')

        #self.logits = self.model(self.images, is_training=self.is_training)
        model = RetinaNet(conf.net)
        #self.loc_preds, self.cls_preds = model(self.images, training=self.is_training)     #retinanet1
        self.loc_preds, self.cls_preds = model(self.images, is_training=self.is_training)   #retinanet2

        self.anchor_boxes = BoxEncoder().get_anchor_boxes(self.image_size)
        self.d_bboxes, self.d_cls_pred, self.d_score = BoxEncoder().decode(self.loc_preds[0], self.cls_preds[0], self.image_size, tf_box_order=conf.tf_box_order)


        self.loc_loss, self.cls_loss, self.iou_loss = loss_fn(self.loc_preds, self.loc_trues, self.cls_preds, self.cls_trues, self.anchor_boxes, num_classes=self.num_class)

        #self.loc_loss, self.cls_loss = loss_fn(self.loc_preds, self.loc_trues, self.cls_preds, self.cls_trues, num_classes=self.num_class)

        self.regularization_loss = tf.losses.get_regularization_loss()

        tf.add_to_collection('losses', self.loc_loss)
        tf.add_to_collection('losses', self.cls_loss)
        tf.add_to_collection('losses', self.iou_loss)
        tf.add_to_collection('losses', self.regularization_loss)
        self.total_loss = tf.add_n(tf.get_collection('losses'))

        tf.summary.scalar('loc_loss', self.loc_loss)
        tf.summary.scalar('cls_loss', self.cls_loss)
        tf.summary.scalar('total_loss', self.total_loss)

        # lr = tf.cond(tf.less(self.global_step, 5000),
        #              lambda: tf.constant(0.0001),
        #              lambda: tf.cond(tf.less(self.global_step, 8000),
        #                              lambda: tf.constant(0.00005),
        #                              lambda: tf.cond(tf.less(self.global_step, 12000),
        #                                              lambda: tf.constant(0.000025),
        #                                              lambda: tf.constant(0.00001))))

        #lr = tf.train.exponential_decay(0.0001, self.global_step, 1000, 0.96, staircase=True)
        #self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)#.minimize(self.avg_loss, global_step=self.global_step)
        #self.optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr)#.minimize(self.avg_loss, global_step=self.global_step)
        #self.optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9, use_nesterov=True)

        self.lr = tf.Variable(float(1e-5), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.lr.assign(self.lr * 0.1)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)


        tf.summary.scalar('learning_rate', self.lr)


        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     #self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)
        #
        #     grads = self.optimizer.compute_gradients(self.total_loss)
        #     for i, (g, v) in enumerate(grads):
        #         if g is not None:
        #             grads[i] = (tf.clip_by_norm(g, 5), v)  # clip gradients
        #     self.train_op = self.optimizer.apply_gradients(grads, global_step=self.global_step)

        vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[180:]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if args.grad_clip_norm is not None:
                grads_and_vars = self.optimizer.compute_gradients(self.total_loss)
                grads = [x[0] for x in grads_and_vars]
                vars = [x[1] for x in grads_and_vars]
                grads, _ = tf.clip_by_global_norm(grads, args.grad_clip_norm)
                self.train_op = self.optimizer.apply_gradients(zip(grads, vars), global_step=self.global_step)
            else:
                self.train_op = self.optimizer.minimize(self.total_loss,  global_step=self.global_step)
                #self.train_op = self.optimizer.minimize(self.total_loss, var_list=vars1, global_step=self.global_step)

        tf.add_to_collection('input_tensor',self.images)
        tf.add_to_collection('input_tensor',self.loc_trues)
        tf.add_to_collection('input_tensor',self.cls_trues)
        tf.add_to_collection('input_tensor',self.is_training)
        tf.add_to_collection('output_tensor', self.loc_preds)
        tf.add_to_collection('output_tensor', self.cls_preds)
        tf.add_to_collection('decode_tensor', self.d_bboxes)
        tf.add_to_collection('decode_tensor', self.d_cls_pred)
        tf.add_to_collection('decode_tensor', self.d_score)

    def train_num_epoch(self, train_writer, merged, dataset, epoch):
        """Trains model on `dataset` using `optimizer`."""

        ave_loss, ave_time = 0., 0.
        batch = 1
        iterator = dataset.make_one_shot_iterator()
        dataset = iterator.get_next()
        while True:
            try:
                #images: [Batch, C, H, W]
                # loc_trues:  shape #[Batch, anchor, 4]
                # cls_trues:  shape #[Batch, anchor]
                images, loc_trues, cls_trues = self.sess.run(dataset)

                start = time.time()
                [summary, _, loc_loss, cls_loss, total_loss, iou_loss, iteration] = self.sess.run(
                fetches=[merged, self.train_op, self.loc_loss, self.cls_loss, self.total_loss, self.iou_loss, self.global_step],
                feed_dict={self.images:images,self.loc_trues:loc_trues, self.cls_trues:cls_trues, self.is_training:True})
                #print('cls_loss:  ', cls_loss, '   loc_loss:  ', loc_loss)
                ave_time += (time.time() - start)
                ave_loss += total_loss
                train_writer.add_summary(summary, iteration)
                if batch % conf.log_interval == 0:
                    time_in_ms = (ave_time * 1000) / (conf.log_interval)
                    ave_loss = (ave_loss) / (conf.log_interval)
                    print("[TRAINING] Batch: {}({:.0f}/{}) \t".format(batch, epoch, conf.num_epochs),
                          "loc_loss: {:.6f} | cls_loss: {:.6f} |ave_loss: {:.6f} | iou_loss: {:.6f} | ave_time: {:.2f}ms".format(loc_loss, cls_loss, ave_loss, iou_loss, time_in_ms))
                    ave_time = 0.
                    ave_loss = 0.
                batch +=1
                # if batch>10:
                #     return iteration
            except tf.errors.OutOfRangeError:
                return iteration

    def eavlution(self, test_writer, merged, dataset, epoch, iteration):
        """test model on `dataset` not using `optimizer`."""
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ test ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        total_time = 0.
        batch = 1
        batch_loss, avg_loss = 0., 0.
        iterator = dataset.make_one_shot_iterator()
        dataset = iterator.get_next()


        num_images = generator.size()
        all_detections = [[None for i in range(generator.num_classes())] for j in range(num_images)]
        while True:
            try:
                #images: [Batch, C, H, W]
                # loc_trues:  shape #[Batch, anchor, 4]
                # cls_trues:  shape #[Batch, anchor]

                images, loc_trues, cls_trues = self.sess.run(dataset)

                start = time.time()
                [summary, loc_loss, cls_loss, total_loss, iou_loss] = self.sess.run(
                fetches=[merged, self.loc_loss, self.cls_loss, self.total_loss, self.iou_loss],
                feed_dict={self.images:images, self.loc_trues:loc_trues, self.cls_trues:cls_trues, self.is_training:False})

                #all_detections = self.compute_mAP(generator, batch-1, self.image_size, all_detections, d_bboxes, d_cls_pred, d_score)

                batch_loss += total_loss
                total_time += (time.time() - start)
                test_writer.add_summary(summary, iteration)
                if batch % (conf.log_interval) == 0:
                    avg_loss += batch_loss
                    ave_loss = batch_loss / conf.log_interval
                    ave_time = ( total_time * 1000) / (conf.log_interval)
                    print("[EVALUATION] Batch: {}({:.0f}/{})\t".format(batch, epoch, conf.num_epochs),
                          "loc_loss: {:.6f} | cls_loss: {:.6f} | iou_loss: {:.6f} | ave_loss: {:.3f} | ave_time: {:.2f}".format(
                                                                            loc_loss, cls_loss, iou_loss, ave_loss, ave_time))
                    batch_loss = 0.
                    total_time = 0.
                batch +=1
                # if batch>30:
                #     return avg_loss
            except tf.errors.OutOfRangeError:
                avg_loss /= batch
                print(">>>>  ave_loss: [{:.3f}]  <<<<<<<".format(avg_loss))

                print('\n\n >>>> Is Computing , please waiting... >>>>\n')
                #average_precisions = evaluate(generator, all_detections, iou_threshold=0.5)
                #mAP = self.print_evaluation(average_precisions)
                return avg_loss#, mAP

    def mAP_eavlution(self):
        """test model on `dataset` not using `optimizer`."""
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ test ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        total_time = 0.
        num_images = generator.size()
        all_detections = [[None for i in range(generator.num_classes())] for j in range(num_images)]
        img_path_list = generator.get_imgpath_list()
        for ind, img_path in enumerate(img_path_list):
            image_4D, output_size = preprocess(img_path, self.image_size)

            start = time.time()
            [d_bboxes, d_cls_pred, d_score] = self.sess.run(
            fetches=[self.d_bboxes, self.d_cls_pred, self.d_score],
            feed_dict={self.images:image_4D,  self.is_training:False})

            used_time = (time.time() - start)
            all_detections = self.compute_mAP(generator, ind, self.image_size, output_size, all_detections, d_bboxes, d_cls_pred, d_score)

            sys.stdout.write('process: [{}/{}]  used_time: {:.2f}ms\r'.format(ind + 1, num_images, used_time*1000))
            sys.stdout.flush()
        print('\n\n >>>> Is Computing , please waiting... >>>>\n')
        average_precisions = evaluate(generator, all_detections, iou_threshold=0.5)
        mAP = self.print_evaluation(average_precisions)
        return mAP

    def mAP_eavlution1(self):
        """test model on `dataset` not using `optimizer`."""
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ test ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        total_time = 0.
        total_losses = 0.
        num_images = generator.size()

        all_annotations, img_size = get_annotations(generator)

        all_detections = [[None for i in range(generator.num_classes())] for j in range(num_images)]
        img_path_list = generator.get_imgpath_list()
        for ind, img_path in enumerate(img_path_list):
            image_4D, output_size = preprocess(img_path, self.image_size)
            boxes = all_annotations[ind][0].tolist()
            labels = tf.ones(len(boxes), tf.int32)
            boxes /= np.array(img_size[ind]*2)

            print(type(boxes))
            print(labels)
            #boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
            #boxes = tf.cast(boxes, dtype=tf.float32)
            boxes = boxes.astype(np.float32)
            loc_trues, cls_trues = BoxEncoder().encode(boxes, labels, conf.input_size)
            print(loc_trues)
            print(type(loc_trues))
            print(cls_trues)
            print(type(cls_trues))
            print(tf.reshape(loc_trues, [1, -1, 4]))
            print(np.expand_dims(np.array(cls_trues), axis=0))


            start = time.time()
            [loc_loss, cls_loss, total_loss, iou_loss, d_bboxes, d_cls_pred, d_score] = self.sess.run(
            fetches=[self.loc_loss, self.cls_loss, self.total_loss, self.iou_loss, self.d_bboxes, self.d_cls_pred, self.d_score],
            feed_dict={self.images:image_4D, self.loc_trues:np.array(loc_trues).reshape([1, -1, 4]), self.cls_trues:np.array(cls_trues).reshape([1, -1]), self.is_training:False})

            used_time = (time.time() - start)
            all_detections = self.compute_mAP(generator, ind, self.image_size, output_size, all_detections, d_bboxes, d_cls_pred, d_score)

            total_losses += total_loss
            if ind+1 % (100) == 0:
                print("[EVALUATION] Batch: [{}/{:.0f}] ({:.0f}/{})\t".format(ind, num_images, epoch, conf.num_epochs),
                      "loc_loss: {:.6f} | cls_loss: {:.6f} | iou_loss: {:.6f} | total_loss: {:.3f} | used_time: {:.2f}".format(
                                                                        loc_loss, cls_loss, iou_loss, total_loss, used_time))


            #sys.stdout.write('process: [{}/{}]  used_time: {:.2f}ms\r'.format(ind + 1, num_images, used_time*1000))
            #sys.stdout.flush()
        print("Total average loss: {:.4f}".format(total_losses/num_images))
        print('\n\n >>>> Is Computing , please waiting... >>>>\n')
        average_precisions = evaluate(generator, all_detections, iou_threshold=0.5)
        mAP = self.print_evaluation(average_precisions)
        return mAP, total_losses/num_images


    def compute_mAP(self, generator, ind, input_size, output_size, all_detections, bboxes, cls_pred, score):
            #output_size = (1080, 1920)

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
            all_detections[ind][label] = image_detections[image_detections[:, -1] == label, :-1]

        return all_detections


    def print_evaluation(self, average_precisions):
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
        return precision / present_classes

    def main(self):
        # gpu_options = tf.GPUOptions(allow_growth=True)
        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.46
        self.sess = tf.Session(config=config)

        if not args.restore:
            self.sess.run(tf.global_variables_initializer())

        if args.backbone_weight_restore:
            backbone_path = "./retinanet2/shufflenetv2_model/shufflenet_v2_1x/model.ckpt-1661328"
            backbone_path = "./checkpoints/retinanet2_mojing/1cls_448x672_5ssd_a4_1branch_nop3p7_data2_p045_n04/model_15.ckpt"
            reader = pywrap_tensorflow.NewCheckpointReader(backbone_path) #tf.train.NewCheckpointReader
            var_to_shape_map = reader.get_variable_to_shape_map()
            if conf.net == 'ShuffleNetV2':
                restore_vars = [var for var in tf.global_variables() if var.name[:-2] in var_to_shape_map]
            elif conf.net == 'Res50':
                restore_vars = [var for var in tf.global_variables() if var.name[:-2] in var_to_shape_map and 'resnet_v2_50' in var.name]
            else:
                ValueError('Invalid net type received: {}'.format(conf.net))
            #print(restore_vars)
            print("\nbackbone_weight_restore prama number: {}".format(len(restore_vars)))
            self.saver_bw = tf.train.Saver(var_list=restore_vars, write_version=tf.train.SaverDef.V2, max_to_keep=5)

        self.saver = tf.train.Saver(var_list=tf.global_variables(), write_version=tf.train.SaverDef.V2, max_to_keep=5)
        train_writer = tf.summary.FileWriter(conf.summary_dir + '/train', self.sess.graph)
        test_writer = tf.summary.FileWriter(conf.summary_dir + '/test')
        merged = tf.summary.merge_all()

        train_ds = dataset_generator('train',
                                  conf.input_size,
                                  num_epochs=1,
                                  batch_size=conf.batch_size,
                                  buffer_size=100,
                                  return_iterator=False,
                                  channels_first=False)  # TODO edit this when in real training
        val_ds = dataset_generator('val',
                                  conf.input_size,
                                  num_epochs=1,
                                  batch_size=conf.batch_size,
                                  buffer_size=100,
                                  return_iterator=False,
                                  channels_first=False)  # TODO edit this when in real training

        with self.sess.as_default():
            if args.restore:
                model_path = './checkpoints/retinanet2_mojing2/1cls_224x384_6ssd_a6_1b_nop7_p04_n045_alph098_data3_ShuffleNetV2/model_37.ckpt'
                self.saver.restore(self.sess, model_path)
                print("\n Restore  all weighes successful !!!")
            if args.backbone_weight_restore:
                self.saver_bw.restore(self.sess, backbone_path)
                print("\n Restore  backbone weighes successful !!!")

            print("\n Building session !!! \n ")

            best_loss = 10000.
            best_mAP = 0.
            flag = 0
            for epoch in range(100):
                print('==> ==> ==> Start training from epoch {:.0f}...\n'.format(epoch+1))
                # Load the dataset
                iteration = self.train_num_epoch(train_writer, merged, train_ds, epoch+1)
                eval_loss = self.eavlution(test_writer, merged, val_ds, epoch+1, iteration)
                mAP = self.mAP_eavlution()

                if best_mAP < mAP:
                    flag = 0
                    best_mAP = mAP
                    print("\n================>>>>>>>>>>>>>>>>>>>> saving <<<<<<<<<<<<<<<<===========================")
                    self.saver.save(self.sess, os.path.join(conf.checkpoint_dir, "model_%d.ckpt"%(epoch+1)))
                    print("\n================>>>>>>>>>>>>>>>>>>>> saving <<<<<<<<<<<<<<<<===========================")
                else:
                    flag += 1
                    if flag >= 4:
                        flag = 0
                        self.sess.run(self.learning_rate_decay_op)
                        print("***** learning rate decline. ******")
        self.sess.close()

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


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Tensorflow RetinaNet Training')
    #parser.add_argument('--cuda-device', default=3, type=int, help='gpu device index')
    parser.add_argument('--grad_clip_norm', '-g', type=float, help='whether use grad_clip or not ')
    parser.add_argument('--restore', '-r', action='store_true', help='whether restore all parameters weights or not')
    parser.add_argument('--backbone_weight_restore', '-rbw', action='store_true', help='whether restore just backbone_weight or not')
    parser.add_argument('--recall_dir', dest='recall_dir', help='recall_dir',
                        default='./demo/recall_dir', type=str)

    args = parser.parse_args()
    ###################################################################################################################
    model_save_dir = "./checkpoints/retinanet2_mojing2/1cls_224x384_6ssd_a7_1b_nop7_p045_n04_Fltf_Res50_rbw"
    summary_dir = "./summary/retinanet2_mojing2/1cls_224x384_6ssd_a7_1b_nop7_p045_n04_Fltf_Res50_rbw"#1cls_224x320_6ssd_a6_1b_nop7_data2_p05_n04_alph098_smooth10
    if not os.path.exists(conf.checkpoint_dir):
        os.mkdir(conf.checkpoint_dir)
    if not os.path.exists(conf.summary_dir):
        os.mkdir(conf.summary_dir)
    GPU_index = '0'
    print('\n==> ==> ==> Using device {}'.format(GPU_index))
    print("\n############################# save in << %s >>  and on << %s >>GPU ##############################\n"%(conf.checkpoint_dir, GPU_index ))
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_index

    generator = MyPascalVocGenerator(mode='test')
    convnet = ConvNet(n_channel=3, num_class=conf.num_class, image_size=conf.input_size)
    convnet.main()
