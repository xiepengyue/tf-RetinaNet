from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys
sys.path.append("..")
from configuration import conf

from retinanet2.fpn import RetinaNet_FPN50, RetinaNet_FPN34
from retinanet2.shufflenetv2 import Shufflenet_FPN
import tensorflow as tf
import os


slim = tf.contrib.slim

class RetinaNet():
    """ RetinaNet defined in Focal loss paper
     See: https://arxiv.org/pdf/1708.02002.pdf
    """
    def __init__(self, model='Res50'):
        self.model = model
        if self.model == 'Res50':
            self._retina_fpn = RetinaNet_FPN50
        if self.model == 'Res34':
            self._retina_fpn = RetinaNet_FPN34
        if self.model == 'ShuffleNetV2':
            self._retina_fpn = Shufflenet_FPN

    def __call__(self, inputs, num_classes=conf.num_class_f, num_anchors=conf.num_anchors, scope=None, reuse=None, is_training=True):
        """
        Args:
            num_classes: # of classification classes
            num_anchors: # of anchors in each feature map
        """
        self._num_classes = num_classes# + 1
        self._num_anchors = num_anchors
        self._scope = scope
        self._reuse = reuse
        return self._forward(inputs, is_training=is_training)

    def _add_fcn_head(self, inputs, output_planes, head_offset):
        """
        inputs: a [batch, height, width, channels] float tensor
        output_planes: # of outputs dim
        layer_offset: idx of feature maps
        """
        with tf.variable_scope(self._scope, "Retina_FCN_Head_"+str(head_offset), [inputs], reuse=self._reuse):
            net = slim.repeat(inputs, 4, slim.conv2d, 256, kernel_size=[3, 3], activation_fn=tf.nn.relu)
            net = slim.conv2d(net, output_planes, kernel_size=[3, 3], activation_fn=None)
        return net

    # def _forward(self, inputs, is_training=True):
    #     batch_size = tf.shape(inputs)[0]
    #     feature_maps = self._retina_fpn(inputs, is_training=is_training)
    #     loc_predictions = []
    #     class_predictions = []
    #     for idx, feature_map in enumerate(feature_maps):
    #         loc_prediction = self._add_fcn_head(feature_map,
    #                                             self._num_anchors * 4,
    #                                             "Box")
    #         class_prediction = self._add_fcn_head(feature_map,
    #                                               self._num_anchors*self._num_classes,
    #                                               "Class")
    #         loc_prediction = tf.reshape(loc_prediction, [batch_size, -1, 4])
    #         class_prediction = tf.reshape(class_prediction, [batch_size, -1, self._num_classes])
    #         loc_predictions.append(loc_prediction)
    #         class_predictions.append(class_prediction)
    #     return tf.concat(loc_predictions, axis=1), tf.concat(class_predictions, axis=1)
    def _forward(self, inputs, is_training=True):
        batch_size = tf.shape(inputs)[0]
        if self.model == 'ShuffleNetV2':
            inputs = (2.0 * inputs) - 1.0
        feature_maps = self._retina_fpn(inputs, is_training=is_training)
        loc_predictions = []
        class_predictions = []
        for idx, feature_map in enumerate(feature_maps):
            if conf.use_one_branch:
                prediction = self._add_fcn_head(feature_map,
                                                    self._num_anchors * (4 + self._num_classes),
                                                    "Box_and_Class")
                prediction = tf.reshape(prediction, [batch_size, -1, 4 + self._num_classes])
                loc_prediction, class_prediction = tf.split(axis=2, num_or_size_splits=[4, self._num_classes], value=prediction)

            else:
                loc_prediction = self._add_fcn_head(feature_map,
                                                    self._num_anchors * 4,
                                                    "Box")
                class_prediction = self._add_fcn_head(feature_map,
                                                      self._num_anchors*self._num_classes,
                                                      "Class")
                loc_prediction = tf.reshape(loc_prediction, [batch_size, -1, 4])
                class_prediction = tf.reshape(class_prediction, [batch_size, -1, self._num_classes])

            loc_predictions.append(loc_prediction)
            class_predictions.append(class_prediction)
        return tf.concat(loc_predictions, axis=1), tf.concat(class_predictions, axis=1)


def test():
    net = RetinaNet()
    inputs = tf.Variable(tf.random_normal([8, 448, 448, 3], dtype=tf.float32), name="inputs")
    loc_predictions, class_predictions = net(inputs, 2, 9)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        print(sess.run([loc_predictions, class_predictions]))
        print(sess.run(tf.shape(loc_predictions)))
        print(sess.run(tf.shape(class_predictions)))
    sess.close()

if __name__ == "__main__":
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    test()
