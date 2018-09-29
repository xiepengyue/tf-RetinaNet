"""Encode object boxes and labels."""
import math
import tensorflow as tf
from utils.box import meshgrid, box_iou, box_nms, change_box_order
from configuration import conf
import os
import numpy as np


def _make_list_input_size(input_size):
    input_size = [input_size] * 2 if isinstance(input_size, int) else input_size
    height, width = input_size[:]
    input_size = (width, height)
    return tf.cast(input_size, tf.float32)


class BoxEncoder:
    def __init__(self):
        # TODO
        # NOTE anchor areas should change according to the ACTUAL object's size
        # Otherwise the height and width of anchor would be out of tune
        # E.g., when the input is 448 x 448, object size ranges in []
        # anchor_areas might be [14^2, 28^2, 56^2, 112^2, 224^2]
        #self.anchor_areas = [14 * 14., 28 * 28., 56 * 56., 112 * 112., 224 * 224.]  # p3 -> p7
        # self.anchor_areas = [28 * 28., 56 * 56., 84.* 84, 112 * 112., 160 * 160.]  # p3 -> p7 anchor4
        # self.aspect_ratios = [1 / 2., 1 / 1., 2 / 1.]
        # self.scale_ratios = [1., pow(2, 1 / 3.), pow(2, 2 / 3.)]

        #self.anchor_areas = [28 * 28., 56 * 56., 84 * 84., 112 * 112., 140 * 140.]  # p3 -> p7   448x672 anchor2
        #self.anchor_areas = [28 * 28., 40 * 40., 60 * 60., 84 * 84., 112 * 112.]  # p3 -> p7   224x320 anchor3
        self.anchor_areas = [20 * 2., 36 * 36., 64 * 64., 112 * 112., 160 * 160.]  # p3 -> p7   224x320 anchor4444
        self.aspect_ratios = [1 / 2., 1 / 1., 2 / 1.]
        self.scale_ratios = [1., pow(2, 1 / 3.), pow(2, 2 / 3.)]

        # self.anchor_areas = [20 * 20., 32 * 32., 48 * 48., 80 * 80., 112 * 112.]  #448,448  anchor5
        # self.aspect_ratios = [1 / 2.5, 1 / 1.5, 1.3 / 1]
        # self.scale_ratios = [1., pow(2, 1 / 3.), pow(2, 2 / 3.)]
        self.anchor_wh = self._get_anchor_wh(mode=conf.anchor_mode)
        self.num_anchors = conf.num_anchors
        #self.anchor_boxes = self._get_anchor_boxes(conf.input_size)


    def _get_anchor_wh(self, mode='RetinaNet'):
        """Compute anchor width and height for each feature map.

        Returns:
            anchor_wh: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 2].
        """
        if mode == 'RetinaNet':
            print(" >>>>>> with RetinaNet anchor")
            anchor_wh = []
            self.anchor_areas = [self.anchor_areas[i] for i in conf.feature_index]
            for s in self.anchor_areas:
                for ar in self.aspect_ratios:  # w/h = ar
                    h = math.sqrt(s / ar)
                    w = ar * h
                    for sr in self.scale_ratios:  # scale
                        anchor_h = h * sr
                        anchor_w = w * sr
                        anchor_wh.append([anchor_w, anchor_h])
            #num_fms = len(self.anchor_areas)
            num_fms = len(self.anchor_areas)
            return tf.reshape(anchor_wh, [num_fms, -1, 2])  # shape [5, 9(3x3), 2]

        if mode == 'ssd':
            print(">>>>>>>> with ssd anchor")
            #anchor_areas = [14, 28, 56, 84, 112, 140]    anchor1
            #anchor_areas = [28, 56, 84, 112, 140, 168]    #anchor2
            #anchor_areas = [48, 64, 70, 80, 96, 112]    #anchor3
            #anchor_areas = [28, 40, 64, 80, 96, 112]    #anchor4 for 448x672
            #self.aspect_ratios = [1., 1 / 1.5, 1 / 2.5, 1.5] #anchor4,5
            #anchor_areas = [24, 36, 48, 60, 80, 96]    #anchor5,6
            #self.aspect_ratios = [1., 1 / 1.2, 1 / 1.5, 1 / 2, 1.5] #anchor6
            anchor_areas = [24, 36, 52, 76, 108, 148]    #anchor7
            self.aspect_ratios = [1., 1 / 1.5, 1 / 2, 1 / 2.5, 1.5] #anchor 7
            #self.aspect_ratios = [1., 2., 3., 1 / 2., 1 / 3.]
            self.anchor_areas = [(anchor_areas[i-1], anchor_areas[i]) for i in range(1, len(anchor_areas))]
            self.anchor_areas = [self.anchor_areas[i] for i in conf.feature_index]
            anchor_wh = []
            for i, s in enumerate(self.anchor_areas):
                for ar in self.aspect_ratios:  # w/h = ar
                    anchor_h = s[0] / math.sqrt(ar)
                    anchor_w = ar * anchor_h
                    anchor_wh.append([anchor_w, anchor_h])
                anchor_s =  math.sqrt(s[0] * s[1])
                anchor_wh.append([anchor_s, anchor_s])
            num_fms = len(self.anchor_areas)
            return tf.reshape(anchor_wh, [num_fms, -1, 2])  # shape [5, 6, 2]


    def _get_anchor_boxes(self, input_size):
        """Compute anchor boxes for each feature map.
        Args:
            input_size: (list) model input size of (w, h)

        Returns:
            boxes: (list) anchor boxes for each feature map. Each of size [#anchors, 4],
                          where #anchors = fmw * fmh * #anchors_per_cell
        """
        num_fms = len(self.anchor_areas)
        fm_sizes = [(tf.ceil(input_size[0] / pow(2., i + 3)), tf.ceil(input_size[1] / pow(2., i + 3)))
                    for i in conf.feature_index]  # TODO modify by p3 -> p7 feature map sizes
        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            grid_size = tf.div(input_size, fm_size)
            fm_w, fm_h = [tf.cast(i, tf.int32) for i in [fm_size[0], fm_size[1]]]

            xy = tf.cast(meshgrid(fm_w, fm_h), tf.float32) + 0.5  # [fm_h*fm_w, 2]
            xy = tf.tile(tf.reshape((xy * grid_size), [fm_h, fm_w, 1, 2]), [1, 1, self.num_anchors, 1])
            wh = tf.tile(tf.reshape(self.anchor_wh[i], [1, 1, self.num_anchors, 2]), [fm_h, fm_w, 1, 1])
            box = tf.concat([xy, wh], 3)  # [x, y, w, h]
            boxes.append(tf.reshape(box, [-1, 4]))
        return tf.concat(boxes, 0)

    def get_anchor_boxes(self, input_size):
        input_size = _make_list_input_size(input_size)
        return self._get_anchor_boxes(input_size)

    def num_anchor_boxes(self, input_size):
        input_size = _make_list_input_size(input_size)
        return tf.shape(self._get_anchor_boxes(input_size), out_type=tf.int32)[0]

    def encode(self, boxes, labels, input_size, pos_iou_threshold=0.5, neg_iou_threshold=0.4):
        """Encode target bounding boxes and class labels.

        We obey the Faster RCNN box coder:
            tx = (x - anchor_x) / anchor_w
            ty = (y - anchor_y) / anchor_h
            tw = log(w / anchor_w)
            th = log(h / anchor_h)

        Args:
            boxes: (tensor) bounding boxes of (xmin, ymin, xmax, ymax), sized [#obj, 4].
            labels: (tensor) object class labels, sized [#obj, ].
            input_size: (int/tuple) model input size of (w, h), should be the same.
        Returns:
            loc_trues: (tensor) encoded bounding boxes, sized [#anchors, 4].
            cls_trues: (tensor) encoded class labels, sized [#anchors, ].
        """

        input_size = _make_list_input_size(input_size)
        boxes = tf.reshape(boxes, [-1, 4])
        anchor_boxes = self._get_anchor_boxes(input_size)

        boxes = change_box_order(boxes, 'xyxy2xywh')
        boxes *= tf.tile(input_size, [2])  # scaled back to original size    ####exchange these two lines????

        ious = box_iou(anchor_boxes, boxes, order='xywh')     #[#anchor, num_bboxes]
        max_ids = tf.argmax(ious, axis=1)                     #[#anchor,]
        max_ious = tf.reduce_max(ious, axis=1)                #[#anchor,]

        gboxes = tf.gather(boxes, max_ids)  # broadcast automatically, [#anchors, 4]
        loc_xy = (gboxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:]
        loc_wh = tf.log(gboxes[:, 2:] / anchor_boxes[:, 2:])
        loc_trues = tf.concat([loc_xy, loc_wh], 1)            #[#anchors, 4]

        cls_trues = tf.gather(labels, max_ids)  # TODO: check if needs add 1 here

        cls_trues = tf.where(max_ious < pos_iou_threshold, tf.zeros_like(cls_trues), cls_trues)
        ignore = (max_ious > neg_iou_threshold) & (max_ious < pos_iou_threshold)  # ignore ious between (0.4, 0.5), and marked as -1
        cls_trues = tf.where(ignore, tf.ones_like(cls_trues) * -1, cls_trues)
        cls_trues = tf.cast(cls_trues, tf.float32)

        ###################################################################################
        """second bigger iou """
        if conf.use_secondbig_loss_constrain:
            mask_ious = tf.one_hot(max_ids, tf.shape(ious, out_type=tf.int32)[1])
            ious -= mask_ious
            second_max_ids = tf.argmax(ious, axis=1)                     #[#anchor,]
            sec_gboxes = tf.gather(boxes, second_max_ids)  # broadcast automatically, [#anchors, 4]
            se_loc_xy = (sec_gboxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:]
            se_loc_wh = tf.log(sec_gboxes[:, 2:] / anchor_boxes[:, 2:])
            sec_loc_trues = tf.concat([se_loc_xy, se_loc_wh], 1)
            loc_trues = tf.concat([loc_trues, sec_loc_trues], 1)

        ###################################################################################
        return loc_trues, cls_trues
        #return anchor_boxes, cls_trues

    def decode(self, loc_preds, cls_preds,
               input_size=conf.input_size,
               output_size = None,
               cls_thred=conf.cls_thred,
               max_output_size=conf.max_output_size,
               nms_thred=conf.nms_thred,
               return_score=True,
               tf_box_order=conf.tf_box_order):
        """Decode outputs back to bouding box locations and class labels.

        We obey the Faster RCNN box coder:
            tx = (x - anchor_x) / anchor_w
            ty = (y - anchor_y) / anchor_h
            tw = log(w / anchor_w)
            th = log(h / anchor_h)
        Args:
            loc_preds: (tensor) predicted locations, sized [#anchors, 4].
            cls_preds: (tensor) predicted class labels, sized [#anchors, #classes].
            input_size: (int/tuple) model input size of (w, h), should be the same.
            cls_thred: class score threshold
            max_output_size: max output nums after nms
            nms_thred: non-maximum suppression threshold
            return_score: (bool) indicate whether to return score value.
            tf_box_order: (bool) True: [ymin, xmin, ymax, xmax]
                                False: [xmin, ymin, xmax, ymax]
        Returns:
            boxes: (tensor) decode box locations, sized [#obj, 4].
                            order determined by param: tf_box_order
            labels: (tensor) class labels for each box, sized [#obj, ].
            NOTE: #obj == min(#detected_objs, #max_output_size)
        """

        input_size = _make_list_input_size(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size)

        loc_xy = loc_preds[:, :2]
        loc_wh = loc_preds[:, 2:]

        xy = loc_xy * anchor_boxes[:, 2:] + anchor_boxes[:, :2]
        wh = tf.exp(loc_wh) * anchor_boxes[:, 2:]
        boxes = tf.concat([xy - wh / 2, xy + wh / 2], 1)  # [#anchors, 4]

        labels = tf.argmax(cls_preds, 1)  # [#anchors, ]
        #score = tf.reduce_max(tf.sigmoid(cls_preds), 1)
        score = tf.sigmoid(tf.reduce_max(cls_preds, 1))  ######### xpy

        #ids = tf.where(tf.greater_equal(score, cls_thred+0.6) & tf.less_equal(score,  cls_thred+0.8))
        ids = tf.cast(score > cls_thred, tf.int32)
        ids = tf.where(tf.not_equal(ids, 0))

        #if not ids.numpy().any():  # Fail to detect, choose the max score
        if ids.shape[0] == 0:  # Fail to detect, choose the max score
            ids = tf.expand_dims(tf.argmax(score), axis=-1)
            print("!!! Box decode: Fail to detect, choose the max score !!!!!!!!!!!!!!!!!!")
        else:
            ids = tf.squeeze(ids, -1)
            #print("Here!!!!!!!!!!!!!")
        if tf_box_order:
            # [ymin, xmin, ymax, xmax]
            boxes = tf.transpose(tf.gather(tf.transpose(boxes), [1, 0, 3, 2]))
            keep = tf.image.non_max_suppression(tf.gather(boxes, ids),
                                                tf.gather(score, ids),
                                                max_output_size=max_output_size,
                                                iou_threshold=nms_thred)
        else:
            # [xmin, ymin, xmax, ymax]
            keep = box_nms(tf.gather(boxes, ids), tf.gather(score, ids), threshold=nms_thred)

        def _index(t, index):
            """Gather tensor successively
            E.g., _index(boxes, [idx_1, idx_2]) = tf.gather(tf.gather(boxes, idx_1), idx_2)
            """
            if not isinstance(index, (tuple, list)):
                index = list(index)
            for i in index:
                t = tf.gather(t, i)
            #t = tf.gather(t, index[0])
            return t

        #return boxes,labels,score
        bboxes = _index(boxes, [ids, keep])
        if tf_box_order:
            bbox = tf.split(axis=1, num_or_size_splits=4, value=bboxes)
            bboxes = tf.concat([bbox[1],bbox[0],bbox[3],bbox[2]], axis=1)
            #bboxes[:, [0, 1, 2, 3]] = bboxes[:, [1, 0, 3, 2]]

        if return_score:
            return bboxes, _index(labels, [ids, keep]), _index(score, [ids, keep])
        return _index(boxes, [ids, keep]), _index(labels, [ids, keep])

    def decode_batch(self,
                     batch_loc_preds,
                     batch_cls_preds,
                     input_size=conf.input_size,
                     output_size = None,
                     tf_box_order=False):
        """Choose the most confident one from multiple (if any) predictions per image.
        Make sure each image only has one output (loc + cls)

        Args:
            batch_loc_preds: (tensor) predicted locations, sized [batch, #anchors, 4].
            batch_cls_preds: (tensor) predicted class labels, sized [batch, #anchors, #classes].
            input_size: (int/tuple) model input size of (w, h), should be the same.
            tf_box_order: (bool) True: [ymin, xmin, ymax, xmax]
                                False: [xmin, ymin, xmax, ymax]
        Returns:
            batch_loc: (tensor)  decode batch box locations, sized [batch, 4]. [y_min, x_min, y_max, x_max]
            batch_cls: (tensor) class label for each box, sized [batch, ]
            batch_scores: (tensor) score for each box, sized [batch, ]

        """
        batch_loc, batch_cls, batch_scores = [], [], []
        for i, (loc_preds, cls_preds) in enumerate(zip(batch_loc_preds, batch_cls_preds)):
            loc, cls, scores = self.decode(loc_preds, cls_preds,
                                           input_size,
                                           output_size = output_size,
                                           tf_box_order=tf_box_order)
            # print("\n########## shape ")
            # print(loc.shape)
            # print(cls.shape)
            # print(scores.shape)
            # print("\n########## shape1 ")
            return loc, cls, scores

            if scores.shape[0] == 0:
                return [None] * 3
            max_score_id = tf.argmax(scores)
            batch_loc.append(tf.gather(loc, max_score_id).numpy() / input_size[0])
            for item in ['cls', 'scores']:
                eval('batch_' + item).append(tf.gather(eval(item), max_score_id).numpy())
        return [tf.convert_to_tensor(item, dtype=tf.float32) for item in [batch_loc, batch_cls, batch_scores]]


def test():
    input_size = (448, 448)
    dataset = dataset_generator('train', input_size, 1, 1, 100,return_iterator=False)
    box_encoder = BoxEncoder()
    model = RetinaNet()

    with tf.device("gpu:0"):
        for i, (image, loc_trues, cls_trues) in enumerate(tfe.Iterator(dataset)):
            print('imagw_shape:{}'.format(image.shape))
            print('loc_trues shape: {}'.format(loc_trues.shape))
            print('cls_trues shape: {}'.format(cls_trues.shape))
            # loc_preds, cls_preds = model(image)
            # print(type(loc_preds[0]))
            # print(type(cls_preds))
            # print('loc_preds shape: {}'.format(loc_preds.shape))
            # print('cls_preds shape: {}'.format(cls_preds.shape))
            # print(type(loc_preds.cpu()))
            # print(type(cls_preds.cpu()))
            #
            # with tf.device("cpu:0"):
            #     # Decode one by one in a batch
            #     loc_preds, cls_preds, score = box_encoder.decode_batch(loc_preds.cpu(), cls_preds.cpu(), input_size)
            #     print('loc_preds {} shape: {}'.format(loc_preds, loc_preds.shape))
            #     print('cls_preds {} shape: {}'.format(cls_preds, cls_preds.shape))
            #     print('score {} shape: {}'.format(score, score.shape))
            #     break


    # gpu_options = tf.GPUOptions(allow_growth=True)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # with sess.as_default():
    #     print(sess.run(box_encoder.get_anchor_boxes(input_size)))


if __name__ == "__main__":
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    import tensorflow.contrib.eager as tfe
    from inputs_multi import dataset_generator
    from retinanet1 import RetinaNet

    tfe.enable_eager_execution()
    test()
