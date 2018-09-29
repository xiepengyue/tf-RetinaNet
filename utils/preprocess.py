import tensorflow as tf
import tensorflow.contrib.eager as tfe
#import tf_extended as tfex
from tensorflow.python.ops import control_flow_ops

import sys
sys.path.append("..")
from utils import image_ops
#import image_ops
import numpy as np
import os
from configuration import conf

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def distort_intensity(image, scope=None):
    with tf.name_scope(scope, 'distort_intensity', [image]):
        prob = tf.random_uniform([], 0, 1.0)
        do_cond = tf.less(prob, .5)
        image = control_flow_ops.cond(do_cond,
                                      lambda: tf.image.random_brightness(image, max_delta=32. / 255.),
                                      lambda: image)
        image = control_flow_ops.cond(do_cond,
                                      lambda: tf.image.random_contrast(image, lower=0.7, upper=1.3),
                                      lambda: image)
        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)

def mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.
  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)
  Note that the rank of `image` must be known.
  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.
  Returns:
    the centered image.
  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=2, values=channels)


def random_crop_patch(image, bboxes, labels, prob=conf.random_crop_patch_prob, seed=None):
    """Random flip left-right of an image and its bounding boxes."""

    def crop_bboxes(image, bboxes, labels, offset_height, offset_width, target_height, target_width):
        """crop bounding boxes coordinates.
        Args:
            bboxes: [xmin, ymin, xmax, ymax]
        """
        #print('>>>>',tf.shape(bboxes))
        #print('>>>>',bboxes)
        bboxes = tf.reshape(bboxes, [-1, 4])
        nocrop_boxes = bboxes

        img_H = tf.cast(tf.shape(image, out_type=tf.int32)[0], tf.float32)
        img_W = tf.cast(tf.shape(image, out_type=tf.int32)[1], tf.float32)
        offset_width = tf.cast(offset_width, tf.float32)
        offset_height = tf.cast(offset_height, tf.float32)
        target_width = tf.cast(target_width, tf.float32)
        target_height = tf.cast(target_height, tf.float32)

        img    = [img_W, img_H]
        offset = [offset_width, offset_height]
        crop   = [target_width - 1., target_height - 1.]

        box_col = tf.split(axis=1, num_or_size_splits=4, value=bboxes)
        for i in range(4):
            box_col[i] = box_col[i] * img[i%2] - offset[i%2]
            box_col[i]  = tf.clip_by_value(box_col[i], 0., crop[i%2]) / crop[i%2]
        bboxes = tf.concat(axis=1, values=box_col)

        keep = tf.where(tf.less_equal(bboxes[:,0], bboxes[:,2]-0.05) & tf.less_equal(bboxes[:,1],  bboxes[:,3]-0.05))
        crop_cond = tf.greater(tf.shape(keep)[0], 100)

        bboxes, labels = tf.cond(crop_cond,
                    lambda: (tf.gather_nd(bboxes, keep), tf.gather_nd(labels, keep)),
                    lambda: (nocrop_boxes, labels))

        #bboxes = tf.gather_nd(bboxes, keep)
        #labels = tf.gather_nd(labels, keep)

        return crop_cond, bboxes, labels

    def crop_images(image, seed=None):
        """crop image coordinates.
        Args:
            bboxes: [xmin, ymin, xmax, ymax]
        """
        img_H = tf.cast(tf.shape(image, out_type=tf.int32)[0], tf.float32)
        img_W = tf.cast(tf.shape(image, out_type=tf.int32)[1], tf.float32)

        uniform_random = tf.random_uniform([], conf.random_crop_image_prob, 1.0, seed=seed)

        target_height = tf.cast(img_H * uniform_random , dtype=tf.int32)
        target_width = tf.cast(img_W * uniform_random, dtype=tf.int32)

        offset_height = tf.random_uniform([], minval=0, maxval=(tf.cast(img_H, dtype=tf.int32) - target_height), dtype=tf.int32)
        offset_width = tf.random_uniform([], minval=0, maxval=(tf.cast(img_W, dtype=tf.int32) - target_width), dtype=tf.int32)

        return offset_height, offset_width, target_height, target_width

    def process(image, bboxes, labels):
        """ process
        """
        offset_height, offset_width, target_height, target_width = crop_images(image)
        crop_cond, crop_bbox, crop_label = crop_bboxes(image, bboxes, labels, offset_height, offset_width, target_height, target_width)
        crop_bbox, crop_label = tf.cond(crop_cond,
                                        lambda: (crop_bbox, crop_label),
                                        lambda: (bboxes, labels))
        crop_image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)
        return crop_image, crop_bbox, crop_label

    def no_process(image, bboxes, labels):
        """ without process
        """
        return image, bboxes, labels

    # Random crop. Tensorflow implementation.
    with tf.name_scope('random_crop_patch'):
        image = tf.convert_to_tensor(image, name='image')

        uniform_random = tf.random_uniform([], 0, 1.0, seed=seed)

        crop_cond = tf.less(uniform_random, prob)

        # crop image.
        image, bboxes, labels = tf.cond(crop_cond, #tf.convert_to_tensor(crop_cond),
                                         lambda: process(image, bboxes, labels),
                                         lambda: no_process(image, bboxes, labels))

        return image, bboxes, labels

def distorted_bounding_box_crop(image,
                                labels,
                                bboxes,
                                min_object_covered=0.3,
                                aspect_ratio_range=(0.9, 1.1),
                                area_range=(0.1, 1.0),
                                max_attempts=200,
                                clip_bboxes=True,
                                scope=None):
    """Generates cropped_image using a one of the bboxes randomly distorted.
    See `tf.image.sample_distorted_bounding_box` for more documentation.
    Args:
        image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
            where each coordinate is [0, 1) and the coordinates are arranged
            as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
            image.
        min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
            area of the image must contain at least this fraction of any bounding box
            supplied.
        aspect_ratio_range: An optional list of `floats`. The cropped area of the
            image must have an aspect ratio = width / height within this range.
        area_range: An optional list of `floats`. The cropped area of the image
            must contain a fraction of the supplied image within in this range.
        max_attempts: An optional `int`. Number of attempts at generating a cropped
            region of the image of the specified constraints. After `max_attempts`
            failures, return the entire image.
        scope: Optional scope for name_scope.
    Returns:
        A tuple, a 3-D Tensor cropped_image and the distorted bbox
    """
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bboxes]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].
        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
                tf.shape(image),
                bounding_boxes=tf.expand_dims(bboxes, 0),
                min_object_covered=min_object_covered,
                aspect_ratio_range=aspect_ratio_range,
                area_range=area_range,
                max_attempts=max_attempts,
                use_image_if_no_bounding_boxes=True)
        distort_bbox = distort_bbox[0, 0]

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        # Restore the shape since the dynamic slice loses 3rd dimension.
        cropped_image.set_shape([None, None, 3])

        # Update bounding boxes: resize and filter out.
        bboxes = tfex.bboxes_resize(distort_bbox, bboxes)
        labels, bboxes = tfex.bboxes_filter_overlap(labels, bboxes,
                                                   threshold=BBOX_CROP_OVERLAP,
                                                   assign_negative=False)
        return cropped_image, labels, bboxes, distort_bbox


def preprocess_for_train(image, bboxes, labels, out_shape, channels_first=False, scope='preprocessing_train'):
    with tf.name_scope(scope, 'preprocessing_train', [image, labels, bboxes]):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        # Convert to float scaled [0, 1].
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            #image = tf.cast(image, tf.float32)

        # Randomly crop image
        image, bboxes, labels = random_crop_patch(image, bboxes, labels, seed=None)


        #image, labels, bboxes, distort_bbox = distorted_bounding_box_crop(image, bboxes, labels)
#
        # Resize image to output size.
        image = tf.image.resize_images(image, out_shape)

        # mean_image_subtraction
        #image = mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])

        # Randomly flip the image horizontally.
        image, bboxes = image_ops.random_flip_lr(image, bboxes)

        # Randomly distort the black-and-white intensity.
        image = distort_intensity(image)

        # H x W x C --> C x H x W
        if channels_first:
            image = tf.transpose(image, perm=(2, 0, 1))
        return image, bboxes, labels


def preprocess_for_val(image, bboxes, labels, out_shape, channels_first=False, scope='preprocessing_eval'):
    with tf.name_scope(scope):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        # Convert to float scaled [0, 1].
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # Resize image to output size.
        image = tf.image.resize_images(image, out_shape)

        # H x W x C --> C x H x W
        if channels_first:
            image = tf.transpose(image, perm=(2, 0, 1))
        return image, bboxes, labels


def preprocess(image, bboxes, labels, mode='train', out_shape=(300, 300), channels_first=False):
    """Pre-process an given image.
       NOTE: Default use NxCxHxW, 'channels_first' is typically faster on GPUs
    Args:
      image: A `Tensor` representing an image of arbitrary size.
      bboxes: (list) bounding boxes [xmin, ymin, xmax, ymax]
      labels: (int) corresponding label in [1, 2, ..., #num_classes]
      out_shape: The height and width of the image after preprocessing.
      mode: `train` if we're preprocessing the image for training and `val` for evaluation.

    Returns:
      A preprocessed image.
    """
    func = eval('preprocess_for_' + mode)
    return func(image, bboxes, labels, out_shape=out_shape, channels_first=channels_first)


def test(mode='train'):

    from inputs_multi import parse_anno_xml, draw_bobx
    result_dir = r'../inference/input_test2'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    impath = r'/workspace/tensorflow/object_det/data/body_detection_data/mirror/nantong/nantong_images/3652615053362176_0.jpg'
    xml_path = r'/workspace/tensorflow/object_det/data/body_detection_data/mirror/nantong/nantong_annotations_xml/3652615053362176_0.xml'
    bboxes, labels = parse_anno_xml(xml_path)
    #img_size =tf.convert_to_tensor(img_size, name='image_size')
    im_raw = tf.read_file(impath)
    image = tf.image.decode_jpeg(im_raw, channels=3)

    print("#########################")
    bboxes, labels = tf.convert_to_tensor(bboxes), tf.convert_to_tensor(labels)
    image, bboxes, labels = preprocess_for_train(image, bboxes, labels, out_shape=(448,672))


    image = tf.stack([image],axis=0)
    bboxes = tf.stack([bboxes],axis=0)
    labels = tf.stack([labels],axis=0)

    draw_bobx(result_dir, image.numpy()*255, bboxes.numpy(), labels.numpy())
    print(image.shape, '-' * 30, "{}th's label: {} [{}]".format(1, np.unique(labels.numpy()), labels.shape))
        #break

    # dataset = dataset_generator(mode, (448, 448), 1, 8, 100, return_iterator=True)
    # gpu_options = tf.GPUOptions(allow_growth=True)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # # graph =tf.Graph()
    # # sess=tf.Session(graph=graph,config=config)
    # with sess.as_default():
    # #with tf.Session() as sess:
    #     i = 0
    #     while True:
    #         image, loc_trues, cls_trues = sess.run(dataset)
    #         #print(image.shape, '-' * 30, "{}th's label: {} [{}]".format(i, np.unique(cls_trues.numpy()), cls_trues.shape))
    #         print(image.shape, '-' * 30, "{}th's label: {}  {} [{}]".format(i, image.shape ,loc_trues.shape, cls_trues.shape))
    #         i += 1


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    tfe.enable_eager_execution()
    test()
