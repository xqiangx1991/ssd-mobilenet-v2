# -*- coding: utf-8 -*-

# @Time   : 2019/9/18:10:17
# @Author : xuqiang

class TFRecordKeys():
    image_encode = "image/encoded"
    image_filename = "image/filename"
    image_format = "image/format"
    image_height = "image/height"
    image_width = "image/width"
    object_bbox_xmax = "image/object/bbox/xmax"
    object_bbox_xmin = "image/object/bbox/xmin"
    object_bbox_ymax = "image/object/bbox/ymax"
    object_bbox_ymin = "image/object/bbox/ymin"
    object_class_label = "image/object/class/label"
    object_class_text = "image/object/class/text"
    image_source_id = "image/source_id"
    pid = "image/object/pid"

class InputDataKeys():
    image_name = "image_name"
    image = "image"
    raw_image = "raw_image"
    groundtruth_classes = "groundtruth_classes"
    groundtruth_bboxes = "groundtruth_bboxes"
    pids = "pids"
    image_width = "image_width"
    image_height = "image_height"
    score = "score"

class DetectionKeys():
    image_name = "image_name"
    image = "image"
    detection_bboxes = "detection_bboxes"
    detection_scores = "detection_scores"
    detection_classes = "detection_classes"
    pids = "pids"
    reid_feature = "reid_feature"
    classes_loss = "classes_loss"
    bboxes_loss = "bboxes_loss"

class LossKeys():
    labels_loss = "labels_loss"
    bboxes_loss = "bboxes_loss"
    pids_loss = "pids_loss"

