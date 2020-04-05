# -*- coding: utf-8 -*-

# @Time   : 2019/12/6:18:22
# @Author : xuqiang

import tensorflow as tf
from src.utils.logger import Logger
from src.utils.cv_utils import draw_detection
from src.utils.standard_fields import DetectionKeys
from src.utils.standard_fields import InputDataKeys

from src.metric.coco_evaluator import CocoEvaluator
from src.metric.reid_evaluator import ReIDEvaluator
from src.ssd.target_assign_util import iou

logger = Logger.getLogger()

# =========== Detection Eval =========== #

def coco_eval_step(model, test_dataset):

    coco_eval = CocoEvaluator()
    for idx,test_data in enumerate(test_dataset):
        coco_eval_one_step(model, test_data, coco_eval)

    step = tf.summary.experimental.get_step()
    if step is not None:
        precision, recall = coco_eval.evaluate()
        tf.summary.scalar("coco/test/precision", precision, step=step)
        tf.summary.scalar("coco/test/recall", recall, step=step)

def coco_eval_one_step(model, test_data, coco_eval):
    image = test_data[InputDataKeys.image]

    predict_dict = model(image)
    predict_raw_labels = predict_dict[DetectionKeys.detection_classes]
    predict_raw_bboxes = predict_dict[DetectionKeys.detection_bboxes]
    # 增加score的限制，必须满足一定条件才可以
    predict_dict = model.postprocess(predict_raw_labels, predict_raw_bboxes,
                                     score_threshold=0.4)

    groundtruth_labels = test_data[InputDataKeys.groundtruth_classes]
    groundtruth_bbxoes = test_data[InputDataKeys.groundtruth_bboxes]
    predict_labels = predict_dict[DetectionKeys.detection_classes]
    predict_bboxes = predict_dict[DetectionKeys.detection_bboxes]

    coco_eval.accumulate(groundtruth_labels, groundtruth_bbxoes,
                        predict_labels, predict_bboxes)


# =========== ReID Eval =========== #

def reid_eval_step(model,
                   test_dataset,
                   pid_name_map_path,
                   gallery_map_path,
                   use_reid = False):


    if not use_reid \
            or pid_name_map_path == '' \
            or gallery_map_path == '':
        return

    reid_eval = ReIDEvaluator(pid_name_map_path, gallery_map_path)
    for test_data in test_dataset:
        reid_eval_one_step(model, test_data, reid_eval)

    step = tf.summary.experimental.get_step()
    if step is not None:
        matched_count, total_query, precision = reid_eval.evaluate()
        images = reid_eval.evaluate_images()
        tf.summary.scalar("reid/test/matched_count", matched_count, step=step)
        tf.summary.scalar("reid/test/total_query", total_query, step=step)
        tf.summary.scalar("reid/test/reid_precision", precision, step=step)

        if len(images) > 0:
            tf.summary.image("reid/test/left:query,middle:gallery,right:gt", images, step=step)


def reid_eval_one_step(model, test_data, reid_eval):
    image_names = test_data[InputDataKeys.image_name]
    images = test_data[InputDataKeys.image]
    labels = test_data[InputDataKeys.groundtruth_classes]
    bboxes = test_data[InputDataKeys.groundtruth_bboxes]
    pids = test_data[InputDataKeys.pids]

    predict_dict = model(images)
    predict_labels = predict_dict[DetectionKeys.detection_classes]
    predict_bboxes = predict_dict[DetectionKeys.detection_bboxes]
    predict_reid_features = predict_dict[DetectionKeys.reid_feature]

    predict_dict = model.postprocess(predict_labels=predict_labels,
                                     predict_bboxes=predict_bboxes,
                                     score_threshold=0.4)
    predict_labels = predict_dict[DetectionKeys.detection_classes]
    predict_bboxes = predict_dict[DetectionKeys.detection_bboxes]

    predict_dict = label_pid_matching(image_names,
                                      images,
                                      labels,
                                      bboxes,
                                      pids,
                                      predict_labels,
                                      predict_bboxes,
                                      predict_reid_features)

    # 统计一下有多少预测的正样本
    step = tf.summary.experimental.get_step()
    if (step is not None) and (step % 100 == 0):
        sample = tf.reduce_sum(tf.cast(predict_dict[DetectionKeys.pids] > -2, tf.int32))
        tf.summary.scalar("reid/test/sample_count", sample, step=step)

    reid_eval.accumulate(predict_dict)


# =========== Utils =========== #

def coco_summary_images(model, test_dataset, step):

    images = list()
    for test_data in test_dataset.take(2):
        image = test_data[InputDataKeys.image]
        labels = tf.cast(test_data[InputDataKeys.groundtruth_classes], tf.float32)
        bboxes = test_data[InputDataKeys.groundtruth_bboxes]

        predict_dict = model(image)
        predict_raw_labels = predict_dict[DetectionKeys.detection_classes]
        predict_raw_bboxes = predict_dict[DetectionKeys.detection_bboxes]
        predict_dict = model.postprocess(predict_raw_labels, predict_raw_bboxes)

        predict_labels = predict_dict[DetectionKeys.detection_classes]
        predict_bboxes = predict_dict[DetectionKeys.detection_bboxes]
        predict_scores = predict_dict[DetectionKeys.detection_scores]

        for idx in range(len(image)):
            im = image[idx]
            im = (im + 1) / 2.0
            pt_im = draw_detection(im, predict_scores[idx], predict_bboxes[idx])
            gt_im = draw_detection(im, labels[idx], bboxes[idx],color=(0,0,255))
            concat_im = tf.concat([gt_im, pt_im], axis=1)
            images.append(concat_im)

    tf.summary.image("test/image/right:groundtruth,left:predict", images, max_outputs=10, step=step)


def label_pid_matching(
        image_names,
        images,
        groundtruth_labels,
        groundtruth_bboxes,
        groundtruth_pids,
        predict_labels,
        predict_bboxes,
        predict_reid_features,
        iou_threshold=0.5):
    labels = tf.squeeze(groundtruth_labels, axis=-1)
    bboxes = groundtruth_bboxes
    pids = groundtruth_pids

    batch_size = tf.shape(labels)[0]
    output_pid = tf.TensorArray(dtype=pids.dtype, size=batch_size)

    # 按照batch来处理
    for idx in range(batch_size):
        batch_labels = labels[idx]
        batch_bboxes = bboxes[idx]
        batch_pids = tf.squeeze(pids[idx], axis=-1)

        batch_selector = tf.squeeze(tf.where(batch_labels > 0), axis=-1)
        batch_labels_selected = tf.gather(batch_labels, batch_selector)
        batch_bboxes_selected = tf.gather(batch_bboxes, batch_selector)
        batch_pids_selected = tf.gather(batch_pids, batch_selector)

        batch_predict_labels = tf.squeeze(predict_labels[idx], axis=-1)
        batch_predict_bboxes = predict_bboxes[idx]

        batch_predict_selector = tf.squeeze(tf.where(batch_predict_labels > 0), axis=-1)
        batch_predict_bboxes_selected = tf.gather(batch_predict_bboxes, batch_predict_selector)

        # 所有选出来的predict bbox和原始的bbox的iou比较一下
        jac = iou(batch_predict_bboxes_selected, batch_bboxes_selected)

        # 每一个anchor找到对应的bbox，然后赋予对应的pid是多少
        jac_anchor_max = tf.reduce_max(jac, axis=1)
        jac_anchor_max_index = tf.argmax(jac, axis=1)
        # 阈值必须满足一定的条件的才拿出来
        jac_anchor_max_mask = tf.where(jac_anchor_max > iou_threshold) #tf.squeeze(tf.where(jac_anchor_max > iou_threshold), axis=-1)
        jac_anchor_max_index = tf.gather(jac_anchor_max_index, jac_anchor_max_mask)
        batch_predict_selector = tf.gather(batch_predict_selector, jac_anchor_max_mask)

        # 根据我们找到的对应label的index 找出对应的pid是多少
        jac_anchor_pid = tf.gather(batch_pids_selected, jac_anchor_max_index)

        # 拿到pid后写回到原来的索引中去
        batch_mask_pid = tf.reshape(tf.ones_like(batch_predict_labels, dtype=batch_pids.dtype) * -2, (-1, 1))
        batch_mask_pid = tf.tensor_scatter_nd_update(batch_mask_pid,
                                                     tf.reshape(tf.cast(batch_predict_selector, dtype=tf.int32),
                                                                (-1, 1)),
                                                     tf.reshape(jac_anchor_pid, (-1, 1)))

        # batch_mask_pid = tf.tensor_scatter_nd_update(batch_mask_pid,
        #                                              tf.cast(batch_predict_selector, dtype=tf.int32),
        #                                              jac_anchor_pid)

        batch_mask_pid = tf.squeeze(batch_mask_pid, axis=-1)

        output_pid = output_pid.write(idx, batch_mask_pid)

    predict_pids = output_pid.stack()
    valid_selector = tf.where(predict_pids > -1)
    valid_batch_selector = tf.unstack(valid_selector, axis=1)[0]
    predict_image_names = tf.gather(image_names, valid_batch_selector)
    predict_images = tf.gather(images, valid_batch_selector)
    predict_labels = tf.gather_nd(predict_labels, valid_selector)
    predict_bboxes = tf.gather_nd(predict_bboxes, valid_selector)
    predict_pids = tf.gather_nd(predict_pids, valid_selector)
    predict_reid_features = tf.gather_nd(predict_reid_features, valid_selector)

    outputs = {
        DetectionKeys.image_name: predict_image_names,
        DetectionKeys.image:predict_images,
        DetectionKeys.detection_classes: predict_labels,
        DetectionKeys.detection_bboxes: predict_bboxes,
        DetectionKeys.pids: predict_pids,
        DetectionKeys.reid_feature: predict_reid_features
    }

    return outputs