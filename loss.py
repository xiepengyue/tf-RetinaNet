import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
from retinanet2.retinanet import RetinaNet
from configuration import conf
#from utils.box import meshgrid, box_iou, box_nms, change_box_order
from tensorflow.python.ops import array_ops

def focal_loss_tf(prediction_tensor, target_tensor, num_classes, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.

    target_tensor = tf.reshape(target_tensor, [-1, 1])
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    loss = tf.reduce_sum(per_entry_cross_ent)

    normalizer = tf.cast(tf.shape(prediction_tensor, out_type=tf.int32)[0], tf.float32)
    normalizer = tf.maximum(1.0, normalizer)

    return loss / normalizer



def focal_loss_my(x_pred, y_true, num_classes):
    """ Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """

    """ Compute the focal loss given the target tensor and the predicted tensor.

    As defined in https://arxiv.org/abs/1708.02002

    Args
        y_true: Tensor of target data from the generator with shape (B, N, num_classes).
        x_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

    Returns
        The focal loss of x_pred w.r.t. y_true.
    """
    alpha=0.9
    # alpha=0.98
    gamma= 2.0 #1.0 #

    labels = tf.reshape(y_true, [-1, 1])
    classification = tf.sigmoid(x_pred)

    # filter out "ignore" anchors
    # compute the focal loss
    alpha_factor = tf.ones_like(labels) * alpha
    alpha_factor = tf.where(tf.equal(labels, 1), alpha_factor, 1 - alpha_factor)
    prob_weight = tf.where(tf.equal(labels, 1), classification, 1 - classification)
    focal_weight = alpha_factor * (1 - prob_weight) ** gamma
    #focal_weight = (1 - prob_weight) ** gamma

    #cls_loss = focal_weight * tf.keras.losses.binary_crossentropy(labels, classification)
    cls_loss = -tf.log(prob_weight) * focal_weight

    # compute the normalizer: the number of positive anchors
    normalizer = tf.cast(tf.shape(y_true, out_type=tf.int32)[0], tf.float32)
    normalizer = tf.maximum(1.0, normalizer)

    return tf.reduce_sum(cls_loss) / normalizer * 200


def focal_loss(x, y, num_classes):
    """Focal loss.

    Args:
        x: (tensor) sized [N, D]
        y: (tensor) sized [N,]
        num_classes: numbers of classes
    Return:
      (tensor) focal loss.
    """
    alpha = 0.25
    gamma = 2

    y = tf.cast(y, tf.int32)
    t = tf.one_hot(y, depth=num_classes+1)  # [N, #total_cls]      #num_classes + 1
    t = t[:, 1:]  # exclude background

    p = tf.sigmoid(x)
    pt = p * t + (1 - p) * (1 - t)  # pt = p if t > 0 else 1-p
    w = alpha * t + (1 - alpha) * (1 - t)  # w = alpha if t > 0 else 1-alpha
    w = w * tf.pow((1 - pt), gamma)

    loss = tf.losses.sigmoid_cross_entropy(t, x, w)

    return loss

    positive_index = tf.where(y > 0)
    num_case = tf.cast(tf.shape(positive_index, out_type=tf.int32)[0], tf.float32)
    num_case = tf.maximum(num_case, 1.0)
    return tf.reduce_sum(loss)/num_case


def focal_loss_alt(x, y, num_classes):
    """Focal loss alternative.

    Args:
        x: (tensor) sized [N, D]
        y: (tensor) sized [N,]
        num_classes: numbers of classes

    Return:
      (tensor) focal loss.
    """
    alpha = 0.25

    y = tf.cast(y, tf.int32)
    t = tf.one_hot(y, depth=num_classes + 1)  # [N, #total_cls]
    t = t[:, 1:]

    xt = x * (2 * t - 1)  # xt = x if t > 0 else -x
    pt = tf.log_sigmoid(2 * xt + 1)

    w = alpha * t + (1 - alpha) * (1 - t)
    loss = -w * pt / 2
    loss = tf.reduce_sum(loss)

    positive_index = tf.where(y > 0)
    num_case = tf.cast(tf.shape(positive_index, out_type=tf.int32)[0], tf.float32)
    num_case = tf.maximum(num_case, 1.0)
    loss = tf.reduce_sum(loss)/num_case
    return loss#/num_case


def smooth_l1(y_pred, y_true,  sigma=3.0):
    """ Compute the smooth L1 loss of y_pred w.r.t. y_true.

    Args
        y_true: Tensor from the generator of shape ( N, 4). The last value for each box is the state of the anchor (ignore, negative, positive).
        y_pred: Tensor from the network of shape (N, 4).

    Returns
        The smooth L1 loss of y_pred w.r.t. y_true.
    """
    sigma_squared = sigma ** 2
    regression        = y_pred
    regression_target = y_true

    # compute smooth L1 loss
    # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
    #        |x| - 0.5 / sigma / sigma    otherwise
    regression_diff = regression - regression_target
    regression_diff = tf.abs(regression_diff)

    regression_loss = tf.where(tf.less(regression_diff, 1.0 / sigma_squared),
        0.5 * sigma_squared * tf.pow(regression_diff, 2),
        regression_diff - 0.5 / sigma_squared
    )

    # compute the normalizer: the number of positive anchors
    num_case = tf.cast(tf.shape(y_true, out_type=tf.int32)[0], tf.float32)
    num_case = tf.maximum(num_case, 1.0)

    return tf.reduce_sum(regression_loss)/num_case


def iou_loss(loc_preds, loc_trues, anchor_boxes, ious_pred):
    """ Compute the smooth L1 loss of y_pred w.r.t. y_true.

    Args
        loc_preds: Tensor from the network of shape (N, 4).
        iou_pred : (N,1)

    Returns
        The MES of error between iou_pred and iou_true which computed by loc_preds.
    """
    def compute_iou(box1, box2, order='xyxy'):
        # A: #box1, B: #box2
        #assert tf.shape(box1, out_type=tf.int32)[0] == tf.shape(box2, out_type=tf.int32)[0]
        lt = tf.maximum(box1[:, :2], box2[:, :2])  # [N, 2], coordinates left-top
        rb = tf.minimum(box1[:, 2:], box2[:, 2:])  # [N, 2], coordinates right-bottom
        wh = tf.clip_by_value(rb - lt, clip_value_min=0, clip_value_max=tf.float32.max) # [N 2], only clip the minimum

        inter = wh[:, 0] * wh[:, 1]  # [N, 1]
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [N,]
        ious = inter / (area1 + area2 - inter)
        return ious

    loc_xy = loc_preds[:, :2]
    loc_wh = loc_preds[:, 2:]
    xy = loc_xy * anchor_boxes[:, 2:] + anchor_boxes[:, :2]
    wh = tf.exp(loc_wh) * anchor_boxes[:, 2:]
    pread_boxes = tf.concat([xy - wh / 2, xy + wh / 2], 1)  # [#anchors, 4]

    loc_xy_true = loc_trues[:, :2]
    loc_wh_true = loc_trues[:, 2:]
    xy_true = loc_xy_true * anchor_boxes[:, 2:] + anchor_boxes[:, :2]
    wh_true = tf.exp(loc_wh_true) * anchor_boxes[:, 2:]
    true_boxes = tf.concat([xy_true - wh_true / 2, xy_true + wh_true / 2], 1)  # [#anchors, 4]


    ious_true = compute_iou(pread_boxes, true_boxes, order='xyxy')     #[#anchor, num_bboxes]
    ious_pred = tf.reshape(tf.sigmoid(ious_pred), [-1])
    loss = tf.nn.l2_loss((ious_true - ious_pred), name='iou_loss')

    num_case = tf.cast(tf.shape(loc_preds, out_type=tf.int32)[0], tf.float32)
    num_case = tf.maximum(num_case, 1.0)

    return loss/num_case


def secondbig_loss_constrain(loc_preds, sec_loc_trues, anchor_boxes):
    """ Compute the smooth L1 loss of y_pred w.r.t. y_true.

    Args
        loc_preds: Tensor from the network of shape (N, 4).
        iou_pred : (N,1)

    Returns
        The MES of error between iou_pred and iou_true which computed by loc_preds.
    """
    def smooth_ln(ious, sigma=0.75):
        sigma_factor = sigma * tf.ones_like(ious)
        loss = tf.where(tf.less_equal(ious, sigma),
                        -tf.log(tf.ones_like(ious) - ious),
                        (ious-sigma_factor) / (tf.ones_like(ious) - sigma_factor) - tf.log(tf.ones_like(ious) - sigma_factor))
        return loss

    def compute_IOG(box1, box2, order='xyxy', epsilon=1e-8):
        # A: #box1, B: #box2
        #assert tf.shape(box1, out_type=tf.int32)[0] == tf.shape(box2, out_type=tf.int32)[0]
        lt = tf.maximum(box1[:, :2], box2[:, :2])  # [N, 2], coordinates left-top
        rb = tf.minimum(box1[:, 2:], box2[:, 2:])  # [N, 2], coordinates right-bottom
        wh = tf.clip_by_value(rb - lt, clip_value_min=0, clip_value_max=tf.float32.max) # [N 2], only clip the minimum

        inter = wh[:, 0] * wh[:, 1]  # [N, 1]
        #area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [N,]
        #ious = inter / (area1 + area2 - inter)
        ious = inter / (area2+ epsilon)
        return ious

    loc_xy = loc_preds[:, :2]
    loc_wh = loc_preds[:, 2:]
    xy = loc_xy * anchor_boxes[:, 2:] + anchor_boxes[:, :2]
    wh = tf.exp(loc_wh) * anchor_boxes[:, 2:]
    pread_boxes = tf.concat([xy - wh / 2, xy + wh / 2], 1)  # [#anchors, 4]

    loc_xy_true = sec_loc_trues[:, :2]
    loc_wh_true = sec_loc_trues[:, 2:]
    xy_true = loc_xy_true * anchor_boxes[:, 2:] + anchor_boxes[:, :2]
    wh_true = tf.exp(loc_wh_true) * anchor_boxes[:, 2:]
    true_boxes = tf.concat([xy_true - wh_true / 2, xy_true + wh_true / 2], 1)  # [#anchors, 4]

    ious_sec_true = compute_IOG(pread_boxes, true_boxes, order='xyxy')     #[#anchor, num_bboxes]
    loss = smooth_ln(ious_sec_true)

    num_case = tf.cast(tf.shape(loc_preds, out_type=tf.int32)[0], tf.float32)
    num_case = tf.maximum(num_case, 1.0)

    return tf.reduce_sum(loss)/num_case


def loss_fn(loc_preds, loc_trues, cls_preds, cls_trues, anchor_boxes, num_classes=20):
    """Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

    Args:
        loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
        loc_trues: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
        cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
        cls_trues: (tensor) encoded target labels, sized [batch_size, #anchors].

    loss:
        (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
    """
    if conf.use_iou_loss:
        cls_pred = tf.split(axis=2, num_or_size_splits=num_classes+1, value=cls_preds)
        cls_preds = cls_pred[0]
        ious_pred = cls_pred[1]


    """
    # 1. cls_loss = FocalLoss(loc_preds, loc_trues)
    # ==================================================================
    """
    #assert tf.shape(tf.reshape(cls_preds, [-1, num_classes]), out_type=tf.int32)[0] == tf.shape(tf.reshape(cls_trues, [-1]), out_type=tf.int32)[0]
    mask_index = tf.where(cls_trues > -1)
    masked_cls_preds = tf.reshape(tf.gather_nd(cls_preds, mask_index), [-1, num_classes])  # [#valid_anchors, #cls]
    masked_cls_trues = tf.reshape(tf.gather_nd(cls_trues, mask_index), [-1])  # [#valid_anchors]


    #cls_loss = focal_loss_alt(masked_cls_preds, masked_cls_trues, num_classes)
    #cls_loss = focal_loss(masked_cls_preds, masked_cls_trues, num_classes)
    cls_loss = focal_loss_my(masked_cls_preds, masked_cls_trues, num_classes)
    #cls_loss = focal_loss_tf(masked_cls_preds, masked_cls_trues, num_classes)
    # ==================================================================


    """
    # 2. loc_loss: tf.losses.huber_loss
    # ==================================================================
    """
    # TODO: cannot use boolean_mask/slice between GPU and CPU
    mask_index = tf.where(cls_trues > 0)

    masked_loc_preds = tf.gather_nd(loc_preds, mask_index)  # [#valid_pos, 4]
    masked_loc_trues = tf.gather_nd(loc_trues, mask_index)  # [#valid_pos, 4]

    if conf.use_secondbig_loss_constrain:
        masked_loc_true = tf.split(axis=1, num_or_size_splits=[4,4], value=masked_loc_trues)
        masked_loc_trues = masked_loc_true[0]
        second_loc_trues = masked_loc_true[1]
        batch_size = tf.shape(cls_trues, out_type=tf.int32)[0]
        anchor_boxes = tf.tile(tf.reshape(anchor_boxes, [1, -1, 4]), [batch_size, 1, 1])
        masked_anchor_boxes = tf.gather_nd(anchor_boxes, mask_index)  # [#valid_pos, 4]
        secondbig_loss = secondbig_loss_constrain(masked_loc_preds, second_loc_trues, masked_anchor_boxes)

    #loc_loss = tf.losses.huber_loss(masked_loc_preds, masked_loc_trues)
    loc_loss = smooth_l1(masked_loc_preds, masked_loc_trues)
    # #
    # print(tf.shape(mask_index, out_type=tf.int32)[0])
    #
    # mask_index2 = tf.where(cls_trues < 0)
    # print(tf.shape(mask_index2, out_type=tf.int32)[0])
    #
    # mask_index1 = tf.where(tf.equal(cls_trues, 0))
    # print(tf.shape(mask_index1, out_type=tf.int32)[0])
    # # ==================================================================


    """
    # 3.  iou_loss: tf.nn.l2_loss
    # =================================================================
    """
    if conf.use_iou_loss:
        batch_size = tf.shape(cls_trues, out_type=tf.int32)[0]
        anchor_boxes = tf.tile(tf.reshape(anchor_boxes, [1, -1, 4]), [batch_size, 1, 1])
        masked_anchor_boxes = tf.gather_nd(anchor_boxes, mask_index)  # [#valid_pos, 4]
        masked_ious_pred    = tf.reshape(tf.gather_nd(ious_pred, mask_index), [-1, 1])  # [#valid_pos, 4]
        ious_loss = iou_loss(masked_loc_preds, masked_loc_trues, masked_anchor_boxes, masked_ious_pred)
        # =================================================================
        return loc_loss, cls_loss, ious_loss

    if conf.use_secondbig_loss_constrain:
        return loc_loss, cls_loss, secondbig_loss*10
    else:
        return loc_loss, cls_loss, tf.constant(0.)



def test1():
    # with tf.device("/gpu:0"):
    # [batch_size, #anchors]s
    # loc_preds = tf.random_uniform([3, 10, 4])
    # loc_trues = tf.random_uniform([3, 10, 4])
    # cls_preds = tf.random_uniform([3, 10, 12])
    # cls_trues = tf.random_uniform([3, 10])

    from inputs_multi import dataset_generator
    from retinanet2.retinanet import RetinaNet
    from encoder import BoxEncoder


    image_size= (448,672)

    dataset = dataset_generator('val', image_size, 1, 1, 100)
    model = RetinaNet()
    anchor_boxes = BoxEncoder().get_anchor_boxes(image_size)


    with tf.device("/gpu:0"):
        for i, (image, loc_trues, cls_trues) in enumerate(tfe.Iterator(dataset)):
            loc_preds, cls_preds = model(image, is_training=True)
            loc_loss, cls_loss, ious_loss = loss_fn(loc_preds, loc_trues, cls_preds, cls_trues, anchor_boxes, num_classes=1)
            # loc_loss, cls_loss = loss_fn(loc_preds, loc_trues, cls_preds, cls_trues, num_classes=1)
            print("Step 0: Location loss: {:.5f}  |  Class loss: {:.5f}".format(loc_loss.numpy(), cls_loss.numpy()))
            #break


def test2():
    loc_preds = [[0.1, 0.1, 0.2, 0.2],
                [0.1, 0.2, 0.1, 0.1]]

    loc_trues = [[0.1, 0.1, 0.2, 0.2],
                [0.1, 0.2, 0.1, 0.1]]

    anchor_boxes = [[200., 100., 200., 200.],
                    [200., 200., 200., 200.]]

    ious_pred = [[80.],[9.]]

    loc_preds =tf.convert_to_tensor(loc_preds, name='bbox1')
    loc_trues =tf.convert_to_tensor(loc_trues, name='bbox2')
    anchor_boxes =tf.convert_to_tensor(anchor_boxes, name='anchor')
    ious_pred =tf.convert_to_tensor(ious_pred, name='iou')

    loss = iou_loss(loc_preds, loc_trues, anchor_boxes, ious_pred)
    print(loss)

def test(mode='test'):

    from inputs_multi import parse_anno_xml
    from utils.preprocess import preprocess_for_train
    from retinanet2.retinanet import RetinaNet
    from encoder import BoxEncoder
    import numpy as np

    image_size= (224,384)
    model = RetinaNet('ShuffleNetV2')
    anchor_boxes = BoxEncoder().get_anchor_boxes(image_size)
    print(anchor_boxes[:30])
    #
    # result_dir = r'../inference/input_test2'
    # if not os.path.exists(result_dir):
    #     os.mkdir(result_dir)

    # impath = r'/workspace/tensorflow/object_det/data/body_detection_data/mirror/nantong/nantong_images/3652615053362176_0.jpg'
    # xml_path = r'/workspace/tensorflow/object_det/data/body_detection_data/mirror/nantong/nantong_annotations_xml/3652615053362176_0.xml'
    impath = r'/workspace/tensorflow/object_det/data/body_detection_data/mirror/spring/v0_JPEGImages/164_1040.jpg'
    xml_path = r'/workspace/tensorflow/object_det/data/body_detection_data/mirror/spring/v0_Annotations_xml/164_1040.xml'

    bboxes, labels = parse_anno_xml(xml_path)
    #print(type(bboxes))
    box = bboxes.copy()
    box *= np.array([image_size[1], image_size[0]] * 2)
    print(box[:, 2:] - box[:, :2])
    #img_size =tf.convert_to_tensor(img_size, name='image_size')
    im_raw = tf.read_file(impath)
    image = tf.image.decode_jpeg(im_raw, channels=3)

    print("#########################")
    bboxes, labels = tf.convert_to_tensor(bboxes), tf.convert_to_tensor(labels)
    image, bboxes, labels = preprocess_for_train(image, bboxes, labels, out_shape=image_size)




    loc_trues, cls_trues = BoxEncoder().encode(bboxes, labels, image_size,
                                                        pos_iou_threshold=0.5,
                                                        neg_iou_threshold=0.33)
    if conf.use_secondbig_loss_constrain:
        #loc_trues = tf.stack([loc_trues, loc_trues],axis=0)
        loc_trues = tf.stack([loc_trues],axis=0)
    else:
        loc_trues = tf.stack([loc_trues],axis=0)
    cls_trues = tf.stack([cls_trues],axis=0)
    #print(loc_trues, cls_trues)
    image = tf.stack([image],axis=0)
    loc_preds, cls_preds = model(image, is_training=True)
    #print(loc_preds, cls_preds)
    #print(image)
    loc_loss, cls_loss, ious_loss = loss_fn(loc_preds, loc_trues, cls_preds, cls_trues, anchor_boxes, num_classes=1)
    #loss = iou_loss(loc_preds, loc_trues, anchor_boxes, ious_pred)
    print(loc_loss, cls_loss, ious_loss)

    #draw_bobx(result_dir, image.numpy()*255, bboxes.numpy(), labels.numpy())

    #print(image.shape, '-' * 30, "{}th's label: {} [{}]".format(1, np.unique(labels.numpy()), labels.shape))
        #break

if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    tfe.enable_eager_execution()
    test()
