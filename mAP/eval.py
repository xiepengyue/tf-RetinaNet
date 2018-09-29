"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#from __future__ import print_function

#from .visualization import draw_detections, draw_annotations

import numpy as np
import os
import sys
import cv2

def compute_overlap(box1, box2):
    """Compute the intersection over union of two set of boxes.
    The default box order is (xmin, ymin, xmax, ymax).

    Args:
      box1: (tf.tensor) bounding boxes, sized [4,].    #xyxy
      box2: (tf.tensor) bounding boxes, sized [B, 4].  # xyxy
      order: (str) box order, either 'xyxy' or 'xywh'.

    Return:
      (tensor) iou, sized [A, B].
    """
    box = np.array([box1] + box2[:])

    # box2 = np.array(box2)
    # box2[:, 2:] = box2[:, :2] + box2[:, 2:]   # xywh2xyxy
    # box = np.array([box1] + box2[:].tolist())

    x1 = box[:, 0]
    y1 = box[:, 1]
    x2 = box[:, 2]
    y2 = box[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    xx1 = np.maximum(x1[0], x1[1:])
    yy1 = np.maximum(y1[0], y1[1:])
    xx2 = np.minimum(x2[0], x2[1:])
    yy2 = np.minimum(y2[0], y2[1:])
    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)

    inter = w * h
    iou = inter / (areas[0] + areas[1:] - inter)
    return iou

def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    '''
    evaluate(
        generator,
        model,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        save_path=None):
    '''
    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        raw_image    = generator.load_image(i)
        image        = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        # run network
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if save_path is not None:
            draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name)

            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return all_detections


def get_annotations(generator):
    """ Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]

    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    img_size = [j for j in range(generator.size())]

    num_image = generator.size()
    for i, xml_path in enumerate(generator.get_xmlpath_list()):
        # load the annotations
        annotations, image_size = generator.load_annotations(xml_path)
        annotations = np.array(annotations)

        img_size[i] = image_size

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        sys.stdout.write('{}/{}\r'.format(i + 1, num_image))
        sys.stdout.flush()
    print('\n')
    return all_annotations, img_size



def evaluate(generator, all_detections, iou_threshold=0.5):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    #all_detections     = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations, img_size    = get_annotations(generator)
    average_precisions = {}
    num_image = generator.size()

    # process detections and annotations
    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(num_image):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(d[:4].tolist(), annotations.tolist())
                assigned_annotation = np.argmax(overlaps, axis=0)
                max_overlap         = overlaps[assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

            sys.stdout.write('[{}/{}] / [{}/{}]\r'.format(i + 1, num_image, label+1, generator.num_classes()))
            sys.stdout.flush()

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    return average_precisions

def iou_test():
    box1 = [100,200,300,400]
    box2 = [[100,200,300,400],
            [70,200,290,400],
            [120,210,300,410],
            [100,150,300,400]]
    overlaps = compute_overlap(box1, box2)
    print(overlaps)
    print(np.argmax(overlaps, axis=0))



if __name__ == '__main__':
    iou_test()









    #
