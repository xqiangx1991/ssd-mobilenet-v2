# -*- coding: utf-8 -*-

# @Time   : 2019/11/18:20:14
# @Author : xuqiang

import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils


'''

不管是LabeledMatching还是UnLabeledMatching，都是输入的所有anchor的数据
这些anchor里面，有些是没人的，有些是没有person id的，但是并不影响计算，因
为这些数据的权重都是0，所以对于LabeledMatching，可以输出[24,1917,5532]，
对于UnLabeledMatching可以输出[24, 1917, 5000]，然后把特征合并，就可以得
到[24, 1917, 10532]，在UnLabeledMatching中，也计算了每一个anchor的特征
相对于所有的带有人的特征的距离。

'''

class OIM():

    def __init__(self,
                 capacity,
                 feature_size,
                 scalar = 0.1,
                 **kwargs):

        super(OIM, self).__init__()
        self.capacity = capacity
        self.feature_size = feature_size
        self.scalar = scalar

        kernel_shape = (self.capacity, self.feature_size)

        self.kernel = tf.Variable(initial_value=tf.zeros(shape=kernel_shape),
                                  trainable=False,
                                  dtype=tf.float32,
                                  name="kernel")

    def __call__(self, inputs, pids, **kwargs):
        self.inputs = inputs
        self.pids = pids

        outputs = tf.matmul(self.inputs, tf.transpose(self.kernel))
        outputs = outputs / self.scalar
        return outputs

    def update_weight(self):
        raise NotImplementedError()


class LabeledMatching(OIM):

    def __init__(self,
                 momentum = 0.5,
                 **kwargs):
        super(LabeledMatching, self).__init__(**kwargs)

        self.momentum = momentum

    def update_weight(self):

        features = self.inputs
        pids = self.pids
        # 所有满足要求的index
        pids_sq = tf.squeeze(pids,axis=-1)
        valid_index = tf.where(tf.logical_and(tf.greater_equal(pids_sq,0),
                                     tf.less(pids_sq, self.capacity)))

        # 获取label
        valid_labels = tf.gather_nd(pids, valid_index)
        valid_features = tf.gather_nd(features, valid_index)

        # 更新
        for idx in range(len(valid_labels)):
            valid_label = int(tf.squeeze(valid_labels[idx]))
            valid_feature = valid_features[idx]
            momentum_feature = self.kernel[valid_label] * self.momentum + \
                                            (1 - self.momentum) * valid_feature
            momentum_feature = momentum_feature / tf.linalg.norm(momentum_feature)
            self.kernel[valid_label].assign(momentum_feature)


class UnLabeledMatching(OIM):

    def __init__(self,
                 **kwargs):
        super(UnLabeledMatching, self).__init__(**kwargs)

        self.queue_tail_idx = tf.Variable(0, dtype=tf.int32,trainable=False)

    def update_weight(self):

        features = self.inputs
        pids = self.pids

        pids_sq = tf.squeeze(pids,axis=-1)
        valid_index = tf.where(tf.equal(pids_sq,-1))

        valid_features = tf.gather_nd(features, valid_index)

        for idx in range(len(valid_features)):
            valid_feature = valid_features[idx]

            normed_feature = valid_feature / tf.norm(valid_feature)
            self.kernel[self.queue_tail_idx].assign(normed_feature)

            self.queue_tail_idx.assign_add(1)

            if self.queue_tail_idx.value() >= self.capacity:
                self.queue_tail_idx.assign(0)
