# -*- coding: utf-8 -*-

# @Time   : 2020/3/3:18:13
# @Author : xuqiang
import tensorflow as tf
from src.dataset.data_loader import DataLoader
from src.utils.standard_fields import TFRecordKeys
from src.utils.standard_fields import InputDataKeys
from src.dataset import tfrecord_decoder


class CocoLoader(DataLoader):

    def __init__(self):
        super().__init__()

    def key_to_features(self):
        k2f = {
            TFRecordKeys.image_encode: tf.io.FixedLenFeature([], tf.string),
            TFRecordKeys.image_filename: tf.io.FixedLenFeature([], tf.string),
            TFRecordKeys.image_format: tf.io.FixedLenFeature([], tf.string),
            TFRecordKeys.image_height: tf.io.FixedLenFeature([], tf.int64),
            TFRecordKeys.image_width: tf.io.FixedLenFeature([], tf.int64),
            TFRecordKeys.object_bbox_xmin: tf.io.VarLenFeature(tf.float32),
            TFRecordKeys.object_bbox_ymin: tf.io.VarLenFeature(tf.float32),
            TFRecordKeys.object_bbox_xmax: tf.io.VarLenFeature(tf.float32),
            TFRecordKeys.object_bbox_ymax: tf.io.VarLenFeature(tf.float32),
            TFRecordKeys.object_class_label: tf.io.VarLenFeature(tf.int64),
            TFRecordKeys.object_class_text: tf.io.VarLenFeature(tf.string),
            TFRecordKeys.image_source_id: tf.io.FixedLenFeature([], tf.string)
        }
        return k2f

    def decoders(self):
        image_decoder = tfrecord_decoder.Image(TFRecordKeys.image_encode, TFRecordKeys.image_format)
        bbox_decoder = tfrecord_decoder.BoundingBox([TFRecordKeys.object_bbox_ymin, TFRecordKeys.object_bbox_xmin,
                                                     TFRecordKeys.object_bbox_ymax, TFRecordKeys.object_bbox_xmax])
        label_decoder = tfrecord_decoder.Tensor(TFRecordKeys.object_class_label, tf.int32)
        width_decoder = tfrecord_decoder.Tensor(TFRecordKeys.image_width, tf.int32)
        height_decoder = tfrecord_decoder.Tensor(TFRecordKeys.image_height, tf.int32)

        decoder_map = {
            InputDataKeys.image:image_decoder,
            InputDataKeys.groundtruth_bboxes:bbox_decoder,
            InputDataKeys.groundtruth_classes:label_decoder
            # InputDataKeys.image_width:width_decoder,
            # InputDataKeys.image_height:height_decoder
        }

        return decoder_map