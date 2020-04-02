# -*- coding: utf-8 -*-

# @Time   : 2019/11/11:15:49
# @Author : xuqiang

import tensorflow as tf
from src.layer.common import Conv2DBlock
from src.layer.common import DepthwiseConv2D


# https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v2.py

def _make_division(v, divisor,min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value,int(v + divisor/2)//divisor*divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class MobileNetV2(tf.keras.models.Model):
    def __init__(self,
                 alpha = 1.0,
                 classes = 1000,
                 **kwargs):
        super().__init__(**kwargs)

        with tf.name_scope("MobileNet") as scope:

            first_block_filters = _make_division(32 * alpha, 8)

            self.conv2d0 = Conv2DBlock(
                filters=first_block_filters,
                kernel_size=3,
                strides=2,
                padding='SAME',
                use_bias=False
            )

            self.inverted_res_group0 = \
                InvetedResGroup(filters=16, in_channel=first_block_filters, strides=1,
                                repeat=1, alpha=alpha, expansion=1)
            self.inverted_res_group1 = \
                InvetedResGroup(filters=24, in_channel=16, strides=2,
                                repeat=2, alpha=alpha)
            self.inverted_res_group2 = \
                InvetedResGroup(filters=32, in_channel=24, strides=2,
                                repeat=3, alpha=alpha)
            self.inverted_res_group3 = \
                InvetedResGroup(filters=64, in_channel=32, strides=2,
                                repeat=4, alpha=alpha)
            self.inverted_res_group4 = \
                InvetedResGroup(filters=96, in_channel=64, strides=1,
                                repeat=3, alpha=alpha)

            # 拆分出一部分操作
            self.conv2d1 = Conv2DBlock(
                filters=6 * 96,
                kernel_size=1,
                strides=1,
                padding='SAME',
                use_bias=False
            )

            self.inverted_res_group5 = \
                InvetedResGroup(filters=160, in_channel=160, strides=2,
                                repeat=1, alpha=alpha, expansion=1)

            self.inverted_res_group6 = \
                InvetedResGroup(filters=160, in_channel=160, strides=1,
                                repeat=2, alpha=alpha)
            self.inverted_res_group7 = \
                InvetedResGroup(filters=320, in_channel=160, strides=1,
                                repeat=1, alpha=alpha)

            last_block_filters = 1280
            if alpha > 1.0:
                last_block_filters = _make_division(1280 * alpha, 8)

            self.conv2d2 = Conv2DBlock(
                filters=last_block_filters,
                kernel_size=1,
                strides=1,
                use_bias=False,
                padding='SAME'
            )


    def call(self, inputs, training=None, mask=None):
        outputs = inputs

        outputs = self.conv2d0(outputs, training=training)
        outputs = self.inverted_res_group0(outputs, training=training)
        outputs = self.inverted_res_group1(outputs, training=training)
        outputs = self.inverted_res_group2(outputs, training=training)
        outputs = self.inverted_res_group3(outputs, training=training)
        outputs = self.inverted_res_group4(outputs, training=training)

        # 中间有一部分结果需要输出
        branch1 = self.conv2d1(outputs, training=training)
        outputs = self.inverted_res_group5(branch1, training=training)
        outputs = self.inverted_res_group6(outputs, training=training)
        outputs = self.inverted_res_group7(outputs, training=training)
        # 输出
        branch2 = self.conv2d2(outputs, training = training)

        return branch1, branch2



class InvetedResGroup(tf.keras.layers.Layer):

    def __init__(self,
                 filters,
                 in_channel,
                 strides,
                 repeat,
                 alpha,
                 expansion = 6,
                 **kwargs):
        super().__init__(**kwargs)
        self.ops = list()

        base_inverted = InvertedResBlock(
            filters=filters,
            in_channel=in_channel,
            expansion = expansion,
            strides=strides,
            alpha=alpha
        )
        self.ops.append(base_inverted)

        for i in range(repeat - 1):
            inverted = InvertedResBlock(
                filters=filters,
                in_channel = filters,
                expansion = expansion,
                strides=1,
                alpha = alpha
            )
            self.ops.append(inverted)

    def call(self, inputs, training = False):
        outputs = inputs
        for op in self.ops:
            outputs = op(outputs)
        return outputs



class InvertedResBlock(tf.keras.layers.Layer):

    def __init__(self,
                 filters,
                 in_channel,
                 expansion,
                 strides,
                 alpha,
                 **kwargs):
        super().__init__(**kwargs)
        with tf.name_scope("InvertedResBlock"):
            pointwise_filters = int(filters * alpha)
            pointwise_filters = _make_division(pointwise_filters, 8)

            self.ops = list()
            if expansion > 1:
                convblock0 = Conv2DBlock(
                    filters=expansion * in_channel,
                    kernel_size=1,
                    strides=1,
                    padding='SAME',
                    use_bias=False
                )
                self.ops.append(convblock0)

            depthwise0 = DepthwiseConv2D(
                kernel_size=3,
                strides=strides,
                padding='SAME',
                use_bias=False)

            self.ops.append(depthwise0)


            convblock1 = Conv2DBlock(
                filters=pointwise_filters,
                kernel_size=1,
                strides=1,
                padding='SAME',
                use_bias=False,
                ac = False
            )
            self.ops.append(convblock1)

            self.add_op = None
            if (in_channel == pointwise_filters) and (strides == 1):
                self.add_op = tf.keras.layers.Add(name="add")

    def call(self, inputs, training=False):
        outputs = inputs
        for op in self.ops:
            outputs = op(outputs, training)

        if self.add_op is not None:
            outputs = self.add_op([inputs, outputs])
        return outputs


if __name__ == "__main__":
    model = MobileNetV2()
    model.build(input_shape=(None,300, 300, 3))
    model.summary()

