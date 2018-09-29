import numpy as np
import xml.etree.ElementTree as ET
import os
from easydict import EasyDict as edict
import cv2
import tensorflow as tf
import sys
sys.path.append("..")
from configuration import conf


from sklearn.cluster import KMeans

num = 0


class MyPascalVocGenerator():
    """ Generate data for a Pascal VOC dataset.

    See http://host.robots.ox.ac.uk/pascal/VOC/ for more information.
    """

    def __init__(self, mode='test' ):
        """ Initialize a Pascal VOC data generator.

        Args
            base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).
            csv_class_file: Path to the CSV classes file.
        """

        self.classes              = conf.class_name
        #self.image_names: [[image_path, image_xml], [], [], ...]
        self.mode                 = mode
        self.image_path_list      = self.get_imgpath_list()


    def size(self):
        """ Size of the dataset.
        """
        return len(self.image_path_list)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return len(self.classes)

    def name_to_label(self, name):
        """ Map name to label.
        """
        return {v: k for k, v in enumerate(self.classes, start=0)}[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return {k: v for k, v in enumerate(self.classes, start=0)}[label]

    def get_imgpath_list(self):
        """ get name pair include [img.xml, img.jpg] .
        """
        name_list = self.name_list(self.mode)
        impath_list = ['{}/{}'.format(conf.dataset_root, p[1]) for p in name_list]
        return impath_list

    def get_xmlpath_list(self):
        """ get name pair include [img.xml, img.jpg] .
        """
        name_list = self.name_list(self.mode)
        xmlpath_list = ['{}/{}'.format(conf.dataset_root, p[0]) for p in name_list]
        return xmlpath_list

    def name_list(self, mode):
        """ get name pair include [img.xml, img.jpg] .
        """
        return self.get_name_list(self.split_filename(mode), mode)

    def get_name_list(self, txt_path_list, mode):
        """Get the list of filenames indicates the index of image and annotation
            Args:
                txt_path: (str) path to the text file, typically is 'path/to/ImageSets/Main/trainval.txt'
            Returns:
                (list) of filenames: E.g., [000012, 000068, 000070, 000073, 000074, 000078, ...]
            """
        if mode == 'train':
            abs_path = conf.train_path
        elif mode == 'test':
            abs_path = conf.test_path
        else:
            raise ValueError('Invalid data type received: {}'.format(mode))

        image_names_pair = []
        for ind, txt_path in enumerate(txt_path_list):
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            for i in range(0, len(lines)):
                image_xml, image_path = lines[i].rstrip('\n').split(' ')[:]
                image_names_pair.append([abs_path[ind][0] + image_xml, abs_path[ind][1] + image_path])
            f.close()
        return image_names_pair

    def load_annotations(self, xml_path):
        """Parse the annotation file (.xml)

        Args:
            xml_path: path to xml file
        Returns:
            bboxes: (list) contains normalized coordinates of [xmin, ymin, xmax, ymax]
            labels: (list) contains **int** index of corresponding class
        """
        try:
            root = ET.parse(xml_path).getroot()
            image_size = [int(root.find('size').find(i).text) for i in ['width', 'height']]
            annotations = []

            for obj in root.findall('object'):
                label_text = obj.find('name').text.lower().strip()
                box = [int(obj.find('bndbox').find(p).text) for p in ['xmin', 'ymin', 'xmax', 'ymax']]
                annotations.append(box+[int(self.name_to_label(label_text))])
            return annotations, image_size

        except ET.ParseError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)
        except ValueError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)

    def split_filename(self, mode):
        """Convert mode to split filename, that is,
        if mode == 'train', filename = 'trainval.txt'
        if mode == 'test',   filename = 'test.txt'
        """
        if mode == 'train':
            return ['{}/{}'.format(conf.dataset_root, file) for file in conf.train_set]
        elif mode == 'test':
            return ['{}/{}'.format(conf.dataset_root, file) for file in conf.test_set]
        else:
            raise ValueError('Invalid data type received: {}'.format(mode))





if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"


    generator = MyPascalVocGenerator(mode='test')
    print(generator.name_to_label('person'))
    print(generator.size())
    print(generator.num_classes())
    print(len(generator.get_imgpath_list()))
    xml_list = generator.get_xmlpath_list()
    print(len(xml_list))
    print(generator.load_annotations(xml_list[5]))










    #
