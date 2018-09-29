"""Encode object boxes and labels."""
import math
import tensorflow as tf
import os
import numpy as np

def change_box_order(boxes, order):
    """Change box order between (xmin, ymin, xmax, ymax) and (xcenter, ycenter, width, height).

    Args:
      boxes: (tf.tensor) bounding boxes, sized [N, 4].
      order: (str) either 'xyxy2xywh' or 'xywh2xyxy'.

    Returns:
      (tf.tensor) converted bounding boxes, sized [N, 4].
    """
    assert order in ['xyxy2xywh', 'xywh2xyxy']
    a = boxes[:, :2]
    b = boxes[:, 2:]
    if order == 'xyxy2xywh':
        return tf.concat([(a + b) / 2, b - a], 1)
    return tf.concat([a - b / 2, a + b / 2], 1)

def box_iou(box1, box2, order='xyxy'):
    """Compute the intersection over union of two set of boxes.
    The default box order is (xmin, ymin, xmax, ymax).

    Args:
      box1: (tf.tensor) bounding boxes, sized [A, 4].
      box2: (tf.tensor) bounding boxes, sized [B, 4].
      order: (str) box order, either 'xyxy' or 'xywh'.

    Return:
      (tensor) iou, sized [A, B].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if order == 'xywh':
        box1, box2 = [change_box_order(i, 'xywh2xyxy') for i in [box1, box2]]

    # A: #box1, B: #box2

    lt = tf.maximum(box1[:, None, :2], box2[:, :2])  # [A, B, 2], coordinates left-top
    rb = tf.minimum(box1[:, None, 2:], box2[:, 2:])  # [A, B, 2], coordinates right-bottom
    #print(lt)

    wh = tf.clip_by_value(rb - lt,  # [A, B, 2], only clip the minimum
                          clip_value_min=0, clip_value_max=tf.float32.max)
    inter = wh[:, :, 0] * wh[:, :, 1]  # [A, B]
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [A,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [B,]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def area(boxlist, scope=None):
  """Computes area of boxes.

  Args:
    boxlist: BoxList holding N boxes following order [ymin, xmin, ymax, xmax]
    scope: name scope.

  Returns:
    a tensor with shape [N] representing box areas.
  """
  with tf.name_scope(scope, 'Area'):
    y_min, x_min, y_max, x_max = tf.split(
        value=boxlist, num_or_size_splits=4, axis=1)
    return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])

def intersection(boxlist1, boxlist2, scope=None):
  """Compute pairwise intersection areas between boxes.

  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes
    scope: name scope.

  Returns:
    a tensor with shape [N, M] representing pairwise intersections
  """
  with tf.name_scope(scope, 'Intersection'):
    y_min1, x_min1, y_max1, x_max1 = tf.split(
        value=boxlist1, num_or_size_splits=4, axis=1)
    y_min2, x_min2, y_max2, x_max2 = tf.split(
        value=boxlist2, num_or_size_splits=4, axis=1)
    all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
    all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
    all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths

def b_iou(boxlist1, boxlist2, scope=None):
  """Computes pairwise intersection-over-union between box collections.

  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes
    scope: name scope.

  Returns:
    a tensor with shape [N, M] representing pairwise iou scores.
  """
  with tf.name_scope(scope, 'IOU'):
    intersections = intersection(boxlist1, boxlist2)
    areas1 = area(boxlist1)
    areas2 = area(boxlist2)
    unions = (
        tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
    return tf.where(
        tf.equal(intersections, 0.0),
        tf.zeros_like(intersections), tf.truediv(intersections, unions))


def test():
    box1 = [[100., 150., 200., 300.],
            [100., 200., 200., 300.]]

    box2 = [[100., 100., 200., 300.],
            [100., 200., 200., 300.]]
    box1 =tf.convert_to_tensor(box1, name='bbox1')
    box2 =tf.convert_to_tensor(box2, name='bbox2')
    iou  = box_iou(box1, box2, order='xyxy')
    print(iou)
    iou = b_iou(box1, box2)
    print(iou)




if __name__ == "__main__":
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    import tensorflow.contrib.eager as tfe

    tfe.enable_eager_execution()
    test()
