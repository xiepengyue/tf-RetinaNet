import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys
sys.path.append("..")
from configuration import conf

BATCH_NORM_MOMENTUM = 0.997
BATCH_NORM_EPSILON = 1e-3


def shufflenet(inputs, num_planes, is_training, num_channels=256, depth_multiplier='1.0'):
    """
    This is an implementation of ShuffleNet v2:
    https://arxiv.org/abs/1807.11164
    Arguments:
        images: a float tensor with shape [batch_size, image_height, image_width, 3],
            a batch of RGB images with pixel values in the range [0, 1].
        is_training: a boolean.
        num_classes: an integer.
        depth_multiplier: a string, possible values are '0.5', '1.0', '1.5', and '2.0'.
    Returns:
        a float tensor with shape [batch_size, num_classes].
    """
    possibilities = {'0.5': 48, '1.0': 116, '1.5': 176, '2.0': 224}
    initial_depth = possibilities[depth_multiplier]

    def batch_norm(x):
        x = tf.layers.batch_normalization(
            x, axis=3, center=True, scale=True,
            training=is_training,
            momentum=BATCH_NORM_MOMENTUM,
            epsilon=BATCH_NORM_EPSILON,
            fused=True, name='batch_norm'
        )
        return x

    # with tf.name_scope('standardize_input'):
    #     inputs = (2.0 * inputs) - 1.0

    feature_maps = []
    with tf.variable_scope('ShuffleNetV2'):
        params = {
            'padding': 'SAME', 'activation_fn': tf.nn.relu,
            'normalizer_fn': batch_norm, 'data_format': 'NHWC',
            'weights_initializer': tf.contrib.layers.xavier_initializer()
        }
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, depthwise_conv], **params):

            c1 = slim.conv2d(inputs, 24, (3, 3), stride=2, scope='Conv1')
            mp1 = slim.max_pool2d(c1, (3, 3), stride=2, padding='SAME', scope='MaxPool')

            c3 = block(mp1, num_units=num_planes[0], out_channels=initial_depth, scope='Stage2')
            c4 = block(c3, num_units=num_planes[1], scope='Stage3')
            c411 = block(c4, num_units=num_planes[2], scope='Stage4')

            final_channels = 1024 if depth_multiplier != '2.0' else 2048
            c5 = slim.conv2d(c411, final_channels, (1, 1), stride=1, scope='Conv5')

            l4 = slim.conv2d(c4, num_channels, [1, 1], stride=1, scope='lat4')
            p5 = slim.conv2d(c5, num_channels, [1, 1], stride=1, scope='conv5')
            # Top down
            t4 = slim.conv2d_transpose(p5, num_channels, [4, 4], stride=[2, 2])
            p4 = conv2d_same(t4+l4, num_channels, 3, stride=1)

            if 'p3' in conf.feature_maps:
                l3 = slim.conv2d(c3, num_channels, [1, 1], stride=1, scope='lat3')
                t3 = slim.conv2d_transpose(p4, num_channels, [4, 4], stride=[2, 2])
                p3 = conv2d_same(t3+l3, num_channels, 3, stride=1)
                feature_maps.append(p3)

            assert 'p4' in conf.feature_maps
            assert 'p5' in conf.feature_maps
            feature_maps.append(p4)
            feature_maps.append(p5)

            if 'p6' in conf.feature_maps:
                p6 = conv2d_same(c5, num_channels, 3, stride=2, scope='conv6')
                feature_maps.append(p6)
                if 'p7' in conf.feature_maps:
                    p7 = conv2d_same(tf.nn.relu(p6), num_channels, 3, stride=2, scope='conv7')
                    feature_maps.append(p7)
            else:
                assert 'p7' not in conf.feature_maps

    return feature_maps

def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
  """Strided 2-D convolution with 'SAME' padding.

  When stride > 1, then we do explicit zero-padding, followed by conv2d with
  'VALID' padding.

  Note that

     net = conv2d_same(inputs, num_outputs, 3, stride=stride)

  is equivalent to

     net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
     net = subsample(net, factor=stride)

  whereas

     net = slim.conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')

  is different when the input's height or width is even, which is why we add the
  current function. For more details, see ResnetUtilsTest.testConv2DSameEven().

  Args:
    inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
    num_outputs: An integer, the number of output filters.
    kernel_size: An int with the kernel_size of the filters.
    stride: An integer, the output stride.
    rate: An integer, rate for atrous convolution.
    scope: Scope.

  Returns:
    output: A 4-D tensor of size [batch, height_out, width_out, channels] with
      the convolution output.
  """
  if stride == 1:
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate,
                       padding='SAME', scope=scope)
  else:
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = tf.pad(inputs,
                    [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                       rate=rate, padding='VALID', scope=scope)


def block(x, num_units, out_channels=None, scope='stage'):
    with tf.variable_scope(scope):

        with tf.variable_scope('unit_1'):
            x, y = basic_unit_with_downsampling(x, out_channels)

        for j in range(2, num_units + 1):
            with tf.variable_scope('unit_%d' % j):
                x, y = concat_shuffle_split(x, y)
                x = basic_unit(x)
        x = tf.concat([x, y], axis=3)

    return x


def concat_shuffle_split(x, y):
    with tf.name_scope('concat_shuffle_split'):
        shape = tf.shape(x)
        batch_size = shape[0]
        height, width = shape[1], shape[2]
        depth = x.shape[3].value

        z = tf.stack([x, y], axis=3)  # shape [batch_size, height, width, 2, depth]
        z = tf.transpose(z, [0, 1, 2, 4, 3])
        z = tf.reshape(z, [batch_size, height, width, 2*depth])
        x, y = tf.split(z, num_or_size_splits=2, axis=3)
        return x, y


def basic_unit(x):
    in_channels = x.shape[3].value
    x = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_before')
    x = depthwise_conv(x, kernel=3, stride=1, activation_fn=None, scope='depthwise')
    x = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_after')
    return x


def basic_unit_with_downsampling(x, out_channels=None):
    in_channels = x.shape[3].value
    out_channels = 2 * in_channels if out_channels is None else out_channels

    y = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_before')
    y = depthwise_conv(y, kernel=3, stride=2, activation_fn=None, scope='depthwise')
    y = slim.conv2d(y, out_channels // 2, (1, 1), stride=1, scope='conv1x1_after')

    with tf.variable_scope('second_branch'):
        x = depthwise_conv(x, kernel=3, stride=2, activation_fn=None, scope='depthwise')
        x = slim.conv2d(x, out_channels // 2, (1, 1), stride=1, scope='conv1x1_after')
        return x, y


@tf.contrib.framework.add_arg_scope
def depthwise_conv(
        x, kernel=3, stride=1, padding='SAME',
        activation_fn=None, normalizer_fn=None,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        data_format='NHWC', scope='depthwise_conv'):

    with tf.variable_scope(scope):
        assert data_format == 'NHWC'
        in_channels = x.shape[3].value
        W = tf.get_variable(
            'depthwise_weights',
            [kernel, kernel, in_channels, 1], dtype=tf.float32,
            initializer=weights_initializer
        )
        x = tf.nn.depthwise_conv2d(x, W, [1, stride, stride, 1], padding, data_format='NHWC')
        x = normalizer_fn(x) if normalizer_fn is not None else x  # batch normalization
        x = activation_fn(x) if activation_fn is not None else x  # nonlinearity
        return x


def Shufflenet_FPN(inputs,
          is_training=True,
          reuse=None,
          scope=None):
    num_planes = [4, 8, 4]
    return shufflenet(inputs, num_planes=num_planes, is_training=is_training)
