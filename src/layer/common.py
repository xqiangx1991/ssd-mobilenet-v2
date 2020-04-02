# -*- coding: utf-8 -*-

# @Time   : 2020/2/12:20:33
# @Author : xuqiang

import tensorflow as tf

def l2_regularizer(weight = 0.00004 * 0.5):
    return tf.keras.regularizers.l2(weight)

def truncated_normal_initializer(mean = 0.0, stddev=0.03):
    return tf.keras.initializers.TruncatedNormal(mean = mean, stddev=stddev)

def constant_initializer():
    return tf.keras.initializers.constant()

def non_max_suppression_with_scores(bboxes,
                        scores,
                        max_output_size,
                        iou_threshold=0.6,
                        score_threshold=float("-inf")):
    selected_indices, selected_scores = \
        tf.image.non_max_suppression_with_scores(bboxes, scores,
                                                 max_output_size, iou_threshold,
                                                 score_threshold)

    return selected_indices, selected_scores

class Conv2DBlock(tf.keras.layers.Layer):

    def __init__(self,
                 filters,
                 kernel_size,
                 strides,
                 padding,
                 use_bias,
                 kernel_initializer = truncated_normal_initializer(stddev=0.09),
                 kernel_regularizer = l2_regularizer(),
                 ac = True,
                 bn = True,
                 momentum = 0.997,
                 **kwargs):
        super().__init__(**kwargs)

        self.batch_normalization = None
        self.activation = None

        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            activation=None,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer
        )

        if bn:
            self.batch_normalization = tf.keras.layers.BatchNormalization(
                momentum=momentum
            )

        if ac:
            self.activation = tf.keras.layers.ReLU(max_value=6)

    def call(self, inputs, training = False):
        outputs = self.conv(inputs)

        if self.batch_normalization is not None:
            outputs = self.batch_normalization(outputs, training = training)

        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs


class DepthwiseConv2D(tf.keras.layers.Layer):

    def __init__(self,
                 kernel_size,
                 strides,
                 padding,
                 use_bias,
                 depth_multiplier = 1,
                 depthwise_initializer = truncated_normal_initializer(stddev=0.09),
                 momentum = 0.997,
                 ac = True,
                 bn = True,
                 **kwargs):
        super().__init__(**kwargs)

        self.batch_normalization = None
        self.activation = None

        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            depth_multiplier=depth_multiplier,
            activation=None,
            depthwise_initializer=depthwise_initializer
        )

        if bn:
            self.batch_normalization = tf.keras.layers.BatchNormalization(
                momentum=momentum
            )

        if ac:
            self.activation = tf.keras.layers.ReLU(max_value=6)

    def call(self, inputs, training = False):
        outputs = self.depthwise_conv(inputs)

        if self.batch_normalization is not None:
            outputs = self.batch_normalization(outputs)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs
