# -*- coding: utf-8 -*-

# @Time   : 2019/9/9:11:17
# @Author : xuqiang

import tensorflow as tf

class BoxCoder:
    def __init__(self,
                 anchors,
                 scale_factors=[10,10,5,5]):
        super(BoxCoder, self).__init__()

        self._anchors = anchors
        self._scale_factors = scale_factors

    def encode(self, bboxes):
        EPSILON = 1e-8

        anchors_ymin = self._anchors[..., 0]
        anchors_xmin = self._anchors[..., 1]
        anchors_ymax = self._anchors[..., 2]
        anchors_xmax = self._anchors[..., 3]

        anchors_yc = (anchors_ymin + anchors_ymax) / 2
        anchors_xc = (anchors_xmin + anchors_xmax) / 2
        anchors_h = anchors_ymax - anchors_ymin + EPSILON
        anchors_w = anchors_xmax - anchors_xmin + EPSILON

        bbox_ymin = bboxes[..., 0]
        bbox_xmin = bboxes[..., 1]
        bbox_ymax = bboxes[..., 2]
        bbox_xmax = bboxes[..., 3]

        bbox_yc = (bbox_ymin + bbox_ymax) / 2
        bbox_xc = (bbox_xmin + bbox_xmax) / 2
        bbox_h = bbox_ymax - bbox_ymin + EPSILON
        bbox_w = bbox_xmax - bbox_xmin + EPSILON

        ty = (bbox_yc - anchors_yc) / anchors_h * self._scale_factors[0]
        tx = (bbox_xc - anchors_xc) / anchors_w * self._scale_factors[1]
        th = tf.math.log(bbox_h / anchors_h) * self._scale_factors[2]
        tw = tf.math.log(bbox_w / anchors_w) * self._scale_factors[3]

        output = tf.stack([ty, tx, th, tw], axis=2)

        return output

    def decode(self, encode_bboxes):
        anchors_ymin = self._anchors[..., 0]
        anchors_xmin = self._anchors[..., 1]
        anchors_ymax = self._anchors[..., 2]
        anchors_xmax = self._anchors[..., 3]

        anchors_yc = (anchors_ymin + anchors_ymax) / 2
        anchors_xc = (anchors_xmin + anchors_xmax) / 2
        anchors_h = anchors_ymax - anchors_ymin
        anchors_w = anchors_xmax - anchors_xmin

        ty = encode_bboxes[..., 0] / self._scale_factors[0]
        tx = encode_bboxes[..., 1] / self._scale_factors[1]
        th = encode_bboxes[..., 2] / self._scale_factors[2]
        tw = encode_bboxes[..., 3] / self._scale_factors[3]

        bbox_yc = ty * anchors_h + anchors_yc
        bbox_xc = tx * anchors_w + anchors_xc
        bbox_h = tf.math.exp(th) * anchors_h
        bbox_w = tf.math.exp(tw) * anchors_w

        ymin = bbox_yc - bbox_h / 2
        xmin = bbox_xc - bbox_w / 2
        ymax = bbox_yc + bbox_h / 2
        xmax = bbox_xc + bbox_w / 2

        output = tf.stack([ymin, xmin, ymax, xmax], axis=2)

        return output
