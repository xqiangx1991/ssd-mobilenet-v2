# -*- coding: utf-8 -*-

# @Time   : 2019/9/6:13:38
# @Author : xuchen.xq

import tensorflow as tf
from src.dataset.augment_utils import bbox_party_inside_window
from src.dataset.augment_utils import bbox_overlapping_above_threshold
from src.dataset.augment_utils import bboxes_regression
from src.utils.standard_fields import InputDataKeys
import numpy as np

def ssd_random_crop(tensor_dict,
                    min_object_covereds = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
                    aspect_ratio_ranges = ((0.5, 2.0),) * 7,
                    area_ranges = ((0.1, 1.0),) * 7,
                    overlap_thresholds =  (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
                    clip_boxes = (True,) * 7,
                    random_coefs = (0.15,) * 7,
                    ):
    # 随机选择一个index
    selected_index = np.random.random_integers(low=1, high=len(min_object_covereds))
    selected_index = selected_index - 1

    min_object_covered = min_object_covereds[selected_index]
    aspect_ratio_range = aspect_ratio_ranges[selected_index]
    area_range = area_ranges[selected_index]
    overlap_threshold = overlap_thresholds[selected_index]
    clip_box = clip_boxes[selected_index]
    random_coef = random_coefs[selected_index]

    output_tensor_dict = random_crop_image(tensor_dict,
                                           min_object_covered,
                                           aspect_ratio_range,
                                           area_range,
                                           overlap_threshold,
                                           clip_box,
                                           random_coef)

    return output_tensor_dict

def random_horizontal_flip_image(tensor_dict):

    def do_random_flip_fn():
        return horizontal_flip_image(tensor_dict)

    random_value = np.random.uniform()
    do_flip_image = tf.greater(random_value, 0.5)

    tensor_dict_cp = dict()
    for key in  tensor_dict.keys():
        tensor_dict_cp[key] = tensor_dict[key]

    result = tf.cond(do_flip_image,
                     do_random_flip_fn,
                     lambda : tensor_dict_cp)

    return result

def random_crop_image(tensor_dict,
                      min_object_covered = 1.0,
                      aspect_ratio_range = (0.75, 1.33),
                      area_range = (0.1, 1.0),
                      overlap_threshold = 0.3,
                      clip_bboxes = True,
                      random_coef = 0.0):

    def do_random_crop_image_fn():
        return crop_image(tensor_dict,
                          min_object_covered,
                          aspect_ratio_range,
                          area_range,
                          overlap_threshold,
                          clip_bboxes)

    # 随机选取是否要进行crop
    random_value = np.random.uniform()
    do_crop_image = tf.greater(random_value, random_coef)

    tensor_dict_cp = dict()
    for key in tensor_dict.keys():
        tensor_dict_cp[key] = tensor_dict[key]

    result = tf.cond(do_crop_image,
                     do_random_crop_image_fn,
                     lambda:tensor_dict_cp)
    return result


def crop_image(tensor_dict,
               min_object_covered,
               aspect_ratio_range,
               area_range,
               overlap_threshold,
               clip_bboxes):
    '''

    :param image: rank == 3, [height, width, channels]
    :param labels:[N, 1]
    :param bboxes:[N, 4]
    :param aspect_ratio_range:
    :return:
    '''

    if InputDataKeys.image not in tensor_dict:
        raise ValueError("image not in tensor_dict")

    if InputDataKeys.groundtruth_bboxes not in tensor_dict:
        raise ValueError("bbox not in tensor_dict")

    image = tensor_dict[InputDataKeys.image]
    bboxes = tensor_dict[InputDataKeys.groundtruth_bboxes]

    image_shape = tf.shape(image)

    # clip bbox value to [0,1]
    expand_bboxes = tf.expand_dims(tf.clip_by_value(t = bboxes,
                                                    clip_value_min = 0.0,
                                                    clip_value_max = 1.0),
                                   axis = 0)


    bbox_begin, bbox_size, distorted_bbox = tf.image.sample_distorted_bounding_box(image_size = image_shape,
                                                                           bounding_boxes = expand_bboxes,
                                                                           min_object_covered = min_object_covered,
                                                                           aspect_ratio_range = aspect_ratio_range,
                                                                           area_range = area_range,
                                                                           max_attempts = 100,
                                                                           use_image_if_no_bounding_boxes = True)

    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # 1 剔除bbox全部超出了范围
    bounding_indices = bbox_party_inside_window(bboxes, distorted_bbox)
    # 得到的index肯定是[N]，但是为了得到mask，需要变成[N,1]
    bounding_indices = tf.expand_dims(bounding_indices,1)
    mask1 = tf.scatter_nd(bounding_indices, tf.ones_like(bounding_indices,tf.bool), [len(bboxes),1])
    # 2 少于一定的ioa的bbox去除
    intersect_indices = bbox_overlapping_above_threshold(bboxes, distorted_bbox, overlap_threshold)
    # 同样，得到的index应该是[N]，需要变成[N,1]
    intersect_indices = tf.expand_dims(intersect_indices,1)
    mask2 = tf.scatter_nd(intersect_indices, tf.ones_like(intersect_indices, tf.bool), [len(bboxes),1])

    mask = tf.logical_and(mask1, mask2)
    mask = tf.squeeze(mask,-1)

    selected_indices = tf.where(mask)
    selected_indices = tf.squeeze(selected_indices)

    selected_bboxes = tf.gather(bboxes, selected_indices)
    # 3 对bbox进行regression
    selected_bboxes = bboxes_regression(selected_bboxes, distorted_bbox)
    if clip_bboxes:
        selected_bboxes = tf.clip_by_value(t = selected_bboxes,
                                           clip_value_min = 0.0,
                                           clip_value_max = 1.0)

    output_dict = dict()
    output_dict[InputDataKeys.image] = distorted_image
    output_dict[InputDataKeys.groundtruth_bboxes] = selected_bboxes

    # 把其余的加入输出
    for key in tensor_dict.keys():
        if key in [InputDataKeys.groundtruth_classes, InputDataKeys.pids]:
            item = tensor_dict[key]
            selected_item = tf.gather(item, selected_indices)
            selected_item = tf.reshape(selected_item, shape=[-1,1])
            output_dict[key] = selected_item
        elif key not in [InputDataKeys.image, InputDataKeys.groundtruth_bboxes]:
            output_dict[key] = tensor_dict[key]

    return output_dict



def horizontal_flip_image(tensor_dict):

    '''

    :param image:[height, width, channel]
    :param bboxes:[N,4]
    :return:
    '''

    if InputDataKeys.image not in tensor_dict:
        raise ValueError("image not in tensor_dict")

    image = tensor_dict[InputDataKeys.image]

    flipped_image = tf.image.flip_left_right(image)

    if InputDataKeys.groundtruth_bboxes in tensor_dict:
        ymin,xmin,ymax,xmax = tf.unstack(tensor_dict[InputDataKeys.groundtruth_bboxes],axis = 1)
        flipped_xmin = tf.math.subtract(1.0, xmax)
        flipped_xmax = tf.math.subtract(1.0, xmin)
        bboxes = tf.stack([ymin, flipped_xmin, ymax, flipped_xmax], axis=1)
        tensor_dict[InputDataKeys.groundtruth_bboxes] = bboxes

    tensor_dict[InputDataKeys.image] = flipped_image

    return tensor_dict
