# -*- coding: utf-8 -*-

# @Time   : 2019/9/9:11:17
# @Author : xuchen.xq

import tensorflow as tf

def bbox_party_inside_window(bboxes, window):
    '''
    bboxes是否全部都在window外面
    :param bboxes:[N, 4]
    :param window:[4]
    :return: index:[N]
    '''

    window = tf.reshape(window, [-1])
    bboxes = tf.reshape(bboxes, [-1, 4, 1])
    bbox_ymin, bbox_xmin, bbox_ymax, bbox_xmax = tf.unstack(bboxes, axis=1)
    window_ymin, window_xmin, window_ymax, window_xmax = tf.unstack(window)

    mask = tf.concat([
        tf.greater_equal(bbox_ymin, window_ymax),
        tf.greater_equal(bbox_xmin, window_xmax),
        tf.less_equal(bbox_ymax, window_ymin),
        tf.less_equal(bbox_xmax, window_xmin)
    ], axis=1)

    valid_indices = tf.reshape(tf.where(tf.logical_not(tf.reduce_any(mask, 1))), [-1])
    valid_indices = tf.cast(valid_indices, tf.int32)

    return valid_indices

def bbox_overlapping_above_threshold(bboxes, window, threshold):
    '''

    :param bboxes:[N, 4]
    :param window:[4]
    :param threshold:scale
    :return: index:[N]
    '''

    # 转换成batch的
    bboxes = tf.reshape(bboxes, [1,-1,4])
    window = tf.reshape(window, [1,1,4])

    ioa_value = ioa(bboxes, window) # [1,N] 其中的1是batch
    # 去掉batch
    ioa_value = tf.reshape(ioa_value, [-1,1]) #[N,1]
    keep_mask = tf.greater_equal(ioa_value, threshold)
    keep_indices = tf.where(keep_mask)
    keep_indices = tf.reshape(keep_indices,[-1])
    keep_indices = tf.cast(keep_indices, tf.int32)
    return keep_indices


def bboxes_regression(bboxes, window):
    '''
    [yin, xmin, ymax, xmax]
    :param bboxes:[N, 4]
    :param window:[4]
    :return:[N, 4]
    '''

    bboxes = tf.reshape(bboxes,[-1,4])
    window = tf.reshape(window,[-1])

    window_height = window[2] - window[0]
    window_width = window[3] - window[1]

    # offset
    window_offset = [window[0], window[1], window[0], window[1]]
    bboxes_offset = bboxes - window_offset

    # scale
    ymin_output = bboxes_offset[:,0] / window_height
    xmin_output = bboxes_offset[:,1] / window_width
    ymax_output = bboxes_offset[:,2] / window_height
    xmax_output = bboxes_offset[:,3] / window_width

    bboxes_output = tf.stack([ymin_output, xmin_output, ymax_output, xmax_output], axis = 1)

    return bboxes_output


def ioa(bboxes1, bboxes2):
    '''
    TODO:考虑到后续要把这部分的功能移到更通用的模块，这里要按照batch的思路进行计算

    :param bboxes1:[batch_size, N, 4]
    :param bboxes2:[batch_size, M, 4]
    :return:[batch_size, N, M]
    '''

    interact_area = intersection(bboxes1, bboxes2)

    # calculate bboxes2 area
    ymin_1 = bboxes1[:, :, 0]
    xmin_1 = bboxes1[:, :, 1]
    ymax_1 = bboxes1[:, :, 2]
    xmax_1 = bboxes1[:, :, 3]

    height2 = tf.maximum(ymax_1 - ymin_1, 0)
    width2 = tf.maximum(xmax_1 - xmin_1, 0)
    area2 = height2 * width2


    return tf.truediv(interact_area, area2)


def intersection(bboxes1, bboxes2):
    '''
    TODO:要加batch
    :param bboxes1:[batch_size, N, 4]
    :param bboxes2:[batch_size, M, 4]
    :return:[batch_size,N,M]
    '''
    ymin_1 = bboxes1[:,:,0]
    xmin_1 = bboxes1[:,:,1]
    ymax_1 = bboxes1[:,:,2]
    xmax_1 = bboxes1[:,:,3]

    ymin_2 = bboxes2[:, :, 0]
    xmin_2 = bboxes2[:, :, 1]
    ymax_2 = bboxes2[:, :, 2]
    xmax_2 = bboxes2[:, :, 3]

    ymin_max = tf.maximum(ymin_1, ymin_2)
    xmin_max = tf.maximum(xmin_1, xmin_2)
    ymax_min = tf.minimum(ymax_1, ymax_2)
    xmax_min = tf.minimum(xmax_1, xmax_2)

    intersect_height = tf.maximum(ymax_min - ymin_max, 0)
    intersect_width = tf.maximum(xmax_min - xmin_max, 0)

    area = intersect_height * intersect_width

    return area

