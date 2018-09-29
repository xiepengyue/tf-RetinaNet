import os
from easydict import EasyDict as edict

conf = edict()

# conf.project_root = os.path.join(os.environ['HOME'], 'pytorch/ssd_pytorch')
# conf.project_path = './'

# Dataset configuration
# ==============================================================================================
# conf.dataset_root = os.path.join(conf.project_root, 'data/VOCdevkit')
# conf.image_dir = os.path.join(conf.dataset_root, 'VOC2007/JPEGImages')
# conf.annot_dir = os.path.join(conf.dataset_root, 'VOC2007/Annotations')
# conf.split_dir = os.path.join(conf.dataset_root, 'VOC2007/ImageSets/Main')
#
# conf.class_name = (
#             'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
#             'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


# conf.project_path = './'
# conf.class_name = ('student','parent')
# conf.dataset_root = r'/home/datadisk3/pengyue/xpy/tensorflow/tf_object_det/data/mojing'
# conf.image_dir = os.path.join(conf.dataset_root, 'JPEGImages')
# conf.annot_dir = os.path.join(conf.dataset_root, 'Annotations')
# conf.split_dir = os.path.join(conf.dataset_root, 'ImageSets/Main')


conf.project_path = './'
conf.class_name = ('person',)
conf.dataset_root = r'/workspace/tensorflow/object_det/data/body_detection_data'
# conf.image_dir = os.path.join(conf.dataset_root, 'JPEGImages')
# conf.annot_dir = os.path.join(conf.dataset_root, 'Annotations')
# conf.split_dir = os.path.join(conf.dataset_root, 'ImageSets/Main')

conf.train_set = ['mirror/nantong/train.txt',
                'mirror/spring/split_train.txt',
                'toG/part0/train.txt',
                'toG/part1/train.txt',
                'arm_detection_2/train.txt',
                 'mojing/train.txt',
                 'new_det_data/train.txt',
                 ]

conf.test_set = ['mirror/nantong/test.txt',
                'mirror/spring/split_test.txt',   #split_test
                'toG/part0/test.txt',
                'toG/part1/test.txt',
                'mojing/test.txt',
                ]

conf.train_path =[['mirror/nantong/nantong_annotations_xml/', 'mirror/nantong/nantong_images/'],
                ['mirror/spring/v0_Annotations_xml/','mirror/spring/v0_JPEGImages/'],
                ['toG/part0/gen_annotation_xml/','toG/part0/gen_image/body/'],
                ['toG/part1/gen_annotation_xml/','toG/part1/gen_image/'],
                ['arm_detection_2/xml/', 'arm_detection_2/image/'],
                ['mojing/xml/', 'mojing/JPEGImages/'],
                ['new_det_data/xml/', 'new_det_data/image/'],
                ]

conf.test_path = [['mirror/nantong/nantong_annotations_xml/', 'mirror/nantong/nantong_images/'],
                ['mirror/spring/v0_Annotations_xml/','mirror/spring/v0_JPEGImages/'],
                ['toG/part0/gen_annotation_xml/','toG/part0/gen_image/body/'],
                ['toG/part1/gen_annotation_xml/','toG/part1/gen_image/'],
                ['mojing/xml/', 'mojing/JPEGImages/'],
                ]

# NOTE: range starts from 1
conf.num_class = len(conf.class_name)
conf.name_to_label_map = {v: k for k, v in enumerate(conf.class_name, start=1)}
#conf.image_size = (1080, 1920)  # (h, w)


# ==============================================================================================
# nms score threshold
conf.nms_thred = 0.4
# score threshold
conf.cls_thred = 0.7
# max detections
conf.max_output_size = 200
# random_crop_patch_prob threshold
conf.random_crop_patch_prob = 0.2
conf.random_crop_image_prob = 0.8 #(crop image size to original image)

#========================================================================================
"""  use iou_loss or not!!!! """
conf.use_iou_loss = False
if conf.use_iou_loss:
    conf.num_class_f = conf.num_class + 1
else:
    conf.num_class_f = conf.num_class

"""  use secondbig_loss_constrain or not!!!! """
conf.use_secondbig_loss_constrain = False
#========================================================================================
"""  use 1111 branch or not """
conf.use_one_branch = True

# network structure
# ==============================================================================================
conf.net = 'Res50'  # ShuffleNetV2,  Res50
print("\n>>>>>>>>>>>>>>>>> net structure is: {}  <<<<<<<<<<<<<<<<<<<<<<<\n".format(conf.net))
# Model configuration

#========================================================================================
""" anchor_mode in [ 'ssd', 'RetinaNet'] """
conf.anchor_mode = 'ssd'

if conf.anchor_mode == 'ssd':
    conf.num_anchors = 6
elif conf.anchor_mode == 'RetinaNet':
    conf.num_anchors = 9
else:
    print("no this anchor_mode!!!")

#feature_maps alternative in ['p3', 'p4', 'p5', 'p6', 'p7']  and ['p4' , 'p5'] must be retain
conf.feature_maps = ['p3','p4', 'p5', 'p6']
conf.feature_index = [int(i[-1])-3 for i in conf.feature_maps]
print(">>>>>>>> with feature_maps:", conf.feature_maps)


# Training configuration
# ==============================================================================================
conf.checkpoint_dir = os.path.join(conf.project_path, 'checkpoints/retinanet2_mojing')
conf.summary_dir = os.path.join(conf.project_path, 'summary/retinanet2_mojing')
conf.log_interval = 10

#positive threshold
conf.pos_iou_threshold = 0.45
#negative threshold
conf.neg_iou_threshold = 0.4

#apply tf nms
conf.tf_box_order = True
conf.channels_first = False
#========================================================================================

#conf.input_size = (448,672)
#conf.input_size = (224,320)
conf.input_size = (224,384)

conf.batch_size = 8
conf.num_epochs = 100
conf.buffer_size = 2000
conf.shuffle = True
#conf.learning_rate = 1e-5

# Deployment configuration
# ==============================================================================================
#conf.deployment_save_dir = 'deploy'

# TESTSET CORRECTION
# ---------------------------------------------------------------------------------
