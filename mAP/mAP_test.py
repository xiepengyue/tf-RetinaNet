from __future__ import print_function
import numpy as np
import os
import cv2

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
    print("><<<<<<<<>>>>>>>>>>>>>")
    print(mrec)
    print(mpre)
    print("><<<<<<<<>>>>>>>>>>>>>")

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        print(mpre)
    print("><<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(mrec)
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
    print(i)

    # and sum (\Delta recall) * prec
    print((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

if __name__ == '__main__':
    average_precisions = {}
    label = 0
    num_annotations = 8
    false_positives = np.array([0,1,0,0,1,0,1])
    true_positives = np.array([1,0,1,1,0,1,0])
    # compute false positives and true positives
    false_positives = np.cumsum(false_positives)
    true_positives  = np.cumsum(true_positives)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>111")
    print(false_positives)
    print(true_positives)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>0000")
    # compute recall and precision
    recall    = true_positives / num_annotations
    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

    print(recall)
    print(precision)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>111")

    # compute average precision
    average_precision  = _compute_ap(recall, precision)
    average_precisions[label] = average_precision, num_annotations

    print(average_precisions)









    #
