import numpy as np
import xml.etree.ElementTree as ET

from functools import partial

import tensorflow as tf
import tensorflow.contrib.eager as tfe
#from tensorflow.python.data import Dataset
#from tensorflow.data import Dataset

from encoder import BoxEncoder
from configuration import conf
import os
from utils.preprocess import preprocess
import cv2

num = 0
def get_name_list(txt_path_list, mode):
    """Get the list of filenames indicates the index of image and annotation
        Args:
            txt_path: (str) path to the text file, typically is 'path/to/ImageSets/Main/trainval.txt'
        Returns:
            (list) of filenames: E.g., [000012, 000068, 000070, 000073, 000074, 000078, ...]
        """
    # with tf.gfile.GFile(txt_path) as f:
    #     lines = f.readlines()
    # return [l.strip() for l in lines]

    if mode == 'train':
        abs_path = conf.train_path
    elif mode == 'val':
        abs_path = conf.test_path
    else:
        raise ValueError('Invalid data type received: {}'.format(mode))


    image_names_pair = []
    for ind, txt_path in enumerate(txt_path_list):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        for i in range(0, len(lines)):
            image_xml, image_path = lines[i].rstrip('\n').split(' ')[:]
            #assert image_xml.endswith('.xml')
            #assert image_xml.endswith('.jpg')
            image_names_pair.append([abs_path[ind][0] + image_xml, abs_path[ind][1] + image_path])
        f.close()
    return image_names_pair

def parse_anno_xml(xml_path):
    """Parse the annotation file (.xml)

    Args:
        xml_path: path to xml file
    Returns:
        bboxes: (list) contains normalized coordinates of [xmin, ymin, xmax, ymax]
        labels: (list) contains **int** index of corresponding class
    """
    root = ET.parse(xml_path).getroot()

    # image shape, [h, w, c]
    shape = [int(root.find('size').find(i).text) for i in ['height', 'width', 'depth']]
    height, width, _ = shape


    # annotations
    label_texts, labels, bboxes = [], [], []
    # if height != 1080 or width != 1920:
    #     print("!!!!!!!!!!!!!!!!!")

    for obj in root.findall('object'):
        label_text = obj.find('name').text.lower().strip()
        #label_texts.append(label_text.encode('utf8'))

        labels.append(conf.name_to_label_map[label_text])
        #name_to_label_map = {'person': 1}
        #labels.append(name_to_label_map[label_text])

        box = [int(obj.find('bndbox').find(p).text) for p in ['xmin', 'ymin', 'xmax', 'ymax']]
        box /= np.array([width, height] * 2)  # rescale boundingbox to [0, 1]
        bboxes.append(box.tolist())

    return bboxes, labels


def split_filename(mode):
    """Convert mode to split filename, that is,
    if mode == 'train', filename = 'trainval.txt'
    if mode == 'val',   filename = 'val.txt'
    """
    if mode == 'train':
        return ['{}/{}'.format(conf.dataset_root, file) for file in conf.train_set]
    elif mode == 'val':
        return ['{}/{}'.format(conf.dataset_root, file) for file in conf.test_set]
    else:
        raise ValueError('Invalid data type received: {}'.format(mode))


def construct_naive_dataset(name_list):
    """Construct corresponding naive dataset

    Args:
        name_list :[xml_name, img_name]
    '''
    args:
        bboxes : # [N,4] , N is not equal
        labels : # [N,] , N is not equal

    '''
    """
    impath_list = ['{}/{}'.format(conf.dataset_root, p[1]) for p in name_list]

    bboxes_list = []
    labels_list = []

    for name in name_list:
        bboxes, labels = parse_anno_xml('{}/{}'.format(conf.dataset_root, name[0]))
        bboxes_list.append(bboxes)
        labels_list.append(labels)

    #bboxes_list = [parse_anno_xml('{}/{}'.format(conf.dataset_root, p[0]))[0] for p in name_list]
    #labels_list = [parse_anno_xml('{}/{}'.format(conf.dataset_root, p[0]))[1] for p in name_list]

############################################################################
    max_n_item = max([len(item) for item in labels_list])
    for index, (item1,item2) in enumerate(zip(bboxes_list,labels_list)):
        while len(item1) < max_n_item:
            item1.append([0]*4)
            item2.append(-2)
        continue
############################################################################

    #bboxes_list = [[parse_anno_xml('{}/{}.xml'.format(conf.annot_dir, p))[0][0]] for p in name_list]
    #labels_list = [[parse_anno_xml('{}/{}.xml'.format(conf.annot_dir, p))[1][0]] for p in name_list]

    return tf.constant(impath_list), tf.constant(bboxes_list), tf.constant(labels_list)


def dataset_generator(mode,
                      input_size=conf.input_size,
                      num_epochs=conf.num_epochs,
                      batch_size=conf.batch_size,
                      buffer_size=conf.buffer_size,
                      return_iterator=False,
                      channels_first=False):
    """Create dataset including [image_dataset, bboxes_dataset, labels_dataset]
        Args:
            mode: (str) 'train' or 'val'
            input_size: (int) input size (h, w)
            num_epochs: (int) nums of looping over the dataset
            batch_size: (int) batch size for input
            buffer_size: (int) representing the number of elements from this dataset
                               from which the new dataset will sample, say, it
                               maintains a fixed-size buffer and chooses the next
                               element uniformly at random from that buffer
            return_iterator: (bool) if false, return dataset instead
    """
    assert mode in ['train', 'val'], "Unknown mode {} besides 'train' and 'val'".format(mode)

    # Helper function to decode image data and processing it
    # ==============================================================================
    def _decode_image(impath, bboxes, labels):
        im_raw = tf.read_file(impath)
        # convert to a grayscale image and downscale x2
        image = tf.image.decode_jpeg(im_raw, channels=3)#, ratio=2)
        # image.set_shape([None, None, 1])

        ########################################
        mask_index = tf.where(labels > -2)
        bboxes = tf.gather_nd(bboxes, mask_index)
        labels = tf.gather_nd(labels, mask_index)
        ########################################
        return image, bboxes, labels

    _preprocess = partial(preprocess, mode=mode, out_shape=input_size, channels_first=channels_first)

    def _encode_boxes(image, bboxes, labels):
        loc_target, cls_target = BoxEncoder().encode(bboxes, labels, input_size,
                                                        pos_iou_threshold=conf.pos_iou_threshold,
                                                        neg_iou_threshold=conf.neg_iou_threshold)
        return image, loc_target, cls_target

    # ==============================================================================

    name_list = get_name_list(split_filename(mode), mode)
    if mode == 'train':
        np.random.shuffle(name_list)

    impath_list, bboxes_list, labels_list = construct_naive_dataset(name_list)
    dataset = tf.data.Dataset.from_tensor_slices((impath_list, bboxes_list, labels_list))

    dataset = dataset.map(_decode_image)
    dataset = dataset.map(_preprocess)
    dataset = dataset.map(_encode_boxes)

    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.repeat(num_epochs)
    batched_dataset = dataset.batch(batch_size)
    if return_iterator:
        iterator = batched_dataset.make_one_shot_iterator()
        batched_dataset = iterator.get_next()
        return batched_dataset
    else:
        # image, labels, bboxes
        return batched_dataset

def label_to_name_map():
    #class_name = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    #                 'dog','horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

    #class_name = ('student','parent')
    class_name = ('person',)
    label_2_name_map = { k:v for k, v in enumerate(class_name, start=1)}
    return label_2_name_map

def draw_bobx(result_dir, image, boxes, labels):
    #label_2_name_map = label_to_name_map()
    label_2_name_map = {-1:'ignor', 0:'bg', 1:'person'}

    for j in range(1):
        label = labels[j]
        bboxes = boxes[j]

        # a = bboxes[:, :2]
        # b = bboxes[:, 2:]
        #
        # bboxes = tf.concat([a - b / 2, a + b / 2], 1)

        b,g,r=cv2.split(image[j])
        im_ = cv2.merge([r,g,b])


        for i in range(bboxes.shape[0]):
        #for i in range(9,18):
            b = bboxes[i]
            x1 = int(b[0]*672)
            y1 = int(b[1]*448)
            x2 = int(b[2]*672)
            y2 = int(b[3]*448)

            # x1 = int(b[0])
            # y1 = int(b[1])
            # x2 = int(b[2])
            # y2 = int(b[3])

            txt_ = label_2_name_map[label[i]]
            cv2.rectangle(im_, (x1,y1),(x2,y2), (0, 0, 255), 2)
            #cv2.putText(im_, txt_, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,225,0), 2, False)
            # if i > 100:
            #     break
        global num
        image_name = '{}.jpg'.format(num)
        num += 1
        print(image_name)
        cv2.imwrite(os.path.join(result_dir, image_name), im_)

def test_bobx(result_dir, image, boxes, labels):
    #label_2_name_map = label_to_name_map()
    label_2_name_map = {-1:'ignor', 0:'bg', 1:'person'}

    for j in range(1):
        label = labels[j]
        bboxes = boxes[j]
        if bboxes.shape[0] == 0:
            print("warning !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")


def test(mode='train'):
    tfe.enable_eager_execution()
    result_dir = r'./inference/input_test'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    dataset = dataset_generator(mode, (224, 320), 1, 8, 100, channels_first=False)
    print("#########################")
    for i, (image, loc_trues, cls_trues) in enumerate(tfe.Iterator(dataset)):
        #print(image.numpy()*255, loc_trues, cls_trues)
        #draw_bobx(result_dir, image.numpy()*255, loc_trues.numpy(), cls_trues.numpy())
        test_bobx(result_dir, image.numpy()*255, loc_trues.numpy(), cls_trues.numpy())
        print(image.shape, '-' * 30, "{}th's label: {} [{}]".format(i, np.unique(cls_trues.numpy()), cls_trues.shape))
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
    test()
