# -*- coding: utf-8 -*-

# @Time   : 2019/12/2:10:23
# @Author : xuqiang

import tensorflow as tf

def iou(bbox1, bbox2):
    """
    :param bbox1:[batch_size, count,4]
    :param bbox2:[batch_size, anchor_count,4]
    :return jaccard:[batch_size,count,anchor_count,1]
    """
    anchors_ymin = bbox2[..., 0]
    anchors_xmin = bbox2[..., 1]
    anchors_ymax = bbox2[..., 2]
    anchors_xmax = bbox2[..., 3]

    bbox_ymin = tf.expand_dims(bbox1[..., 0], axis=-1)
    bbox_xmin = tf.expand_dims(bbox1[..., 1], axis=-1)
    bbox_ymax = tf.expand_dims(bbox1[..., 2], axis=-1)
    bbox_xmax = tf.expand_dims(bbox1[..., 3], axis=-1)

    # bbox_ymin = bbox1[..., 0]
    # bbox_xmin = bbox1[..., 1]
    # bbox_ymax = bbox1[..., 2]
    # bbox_xmax = bbox1[..., 3]

    anchor_vols = (anchors_xmax - anchors_xmin) * (anchors_ymax - anchors_ymin)
    bbox_vol = (bbox_xmax - bbox_xmin) * (bbox_ymax - bbox_ymin)

    xmin = tf.maximum(bbox_xmin, anchors_xmin)
    ymin = tf.maximum(bbox_ymin, anchors_ymin)
    xmax = tf.minimum(bbox_xmax, anchors_xmax)
    ymax = tf.minimum(bbox_ymax, anchors_ymax)

    h = tf.maximum(ymax - ymin, 0)
    w = tf.maximum(xmax - xmin, 0)

    inter_vol = h * w
    union_vol = anchor_vols + bbox_vol - inter_vol
    jaccard = tf.where(tf.equal(inter_vol, 0), tf.zeros_like(inter_vol), tf.truediv(inter_vol, union_vol))
    return jaccard

def set_value_use_index(x, indicator, value):

    indicator = tf.cast(indicator, x.dtype)
    return tf.add(tf.multiply(x, 1 - indicator), value * indicator)