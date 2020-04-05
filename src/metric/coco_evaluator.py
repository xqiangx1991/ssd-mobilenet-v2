# -*- coding: utf-8 -*-

# @Time   : 2020/3/9:16:01
# @Author : xuqiang

import tensorflow as tf
from src.ssd.target_assign_util import iou

class CocoEvaluator():
    '''
    整体的逻辑是以iou为分数
    按照groundtruth一个一个去对比
    每一个框只有一个label，就看匹配不匹配
    先筛选出
    '''

    def __init__(self):
        self.iou_th = 0.5
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def accumulate(self,
                 groundtruth_labels,
                 groundtruth_bboxes,
                 prediction_labels,
                 prediction_bbxoes):
        '''
        这里的真实值是没有进行target assign的，是原始的框。
        照理说应该还要加上真实值的width和height，这样可以区分目标物体的像素大小。
        '''
        # accuracy = (检测正确的pt)/(所有positive的pt)
        # recall = (检测正确的gt)/(所有的gt)

        batch_size = tf.shape(prediction_labels)[0]

        for batch in range(batch_size):
            groundtruth_bboxes_batch = groundtruth_bboxes[batch]
            prediction_bboxes_batch = prediction_bbxoes[batch]
            pt_labels_batch = tf.cast(prediction_labels[batch], tf.int32)
            pt_labels_batch = tf.reshape(pt_labels_batch,(-1,))
            gt_labels_batch = groundtruth_labels[batch]

            jaccards_batch = iou(groundtruth_bboxes_batch, prediction_bboxes_batch)
            jaccards_argmax_cross_pt_batch = tf.argmax(jaccards_batch,axis=0)
            jaccards_max_cross_pt_batch = tf.reduce_max(jaccards_batch, axis=0)

            jaccards_argmax_cross_gt_batch = tf.argmax(jaccards_batch, axis=1)
            jaccards_max_cross_gt_batch = tf.reduce_max(jaccards_batch, axis=1)

            # ========= 计算precision =========
            # 满足条件的才是有效的
            jaccards_max_cross_pt_mask = jaccards_max_cross_pt_batch > self.iou_th

            # 拿到gt labels
            gt_labels_cross_pt = tf.gather(gt_labels_batch, jaccards_argmax_cross_pt_batch)
            gt_labels_cross_pt = tf.squeeze(gt_labels_cross_pt, axis=-1)
            gt_labels_cross_pt = gt_labels_cross_pt * \
                                 tf.cast(jaccards_max_cross_pt_mask, gt_labels_cross_pt.dtype)
            gt_labels_cross_pt = tf.reshape(gt_labels_cross_pt,(-1,))
            # 比较一下
            # TODO:有可能出现一个gt对应几个pt，但是这种情况先不考虑

            self.precision.update_state(gt_labels_cross_pt, pt_labels_batch)
            # ========= 计算recall =========
            # iou要满足条件
            jaccards_max_cross_gt_mask = jaccards_max_cross_gt_batch > self.iou_th
            pt_labels_cross_gt = tf.gather(pt_labels_batch, jaccards_argmax_cross_gt_batch)
            pt_labels_cross_gt = pt_labels_cross_gt * \
                                 tf.cast(jaccards_max_cross_gt_mask,pt_labels_cross_gt.dtype)

            # 比较一下
            gt_labels_batch = tf.reshape(gt_labels_batch, (-1,))
            self.recall.update_state(gt_labels_batch,pt_labels_cross_gt)

    def evaluate(self):
        precision = self.precision.result()
        recall = self.recall.result()

        self.precision.reset_states()
        self.recall.reset_states()

        return precision, recall



