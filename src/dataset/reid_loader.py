# -*- coding: utf-8 -*-

# @Time   : 2020/3/3:19:17
# @Author : xuqiang

import tensorflow as tf
from src.utils.standard_fields import TFRecordKeys
from src.dataset.coco_loader import CocoLoader
from src.dataset import tfrecord_decoder
from src.utils.standard_fields import InputDataKeys

class ReIDLoader(CocoLoader):

    def __init__(self):
        super().__init__()

    def key_to_features(self):
        k2p = super().key_to_features()
        k2p[TFRecordKeys.pid] = tf.io.VarLenFeature(tf.int64)
        return k2p

    def decoders(self):
        decoder_map = super().decoders()
        pid_decoder = tfrecord_decoder.Tensor(TFRecordKeys.pid, tf.int32)
        decoder_map[InputDataKeys.pids] = pid_decoder
        image_name_decoder = tfrecord_decoder.Tensor(TFRecordKeys.image_filename, tf.string)
        decoder_map[InputDataKeys.image_name] = image_name_decoder

        return decoder_map

if __name__ == "__main__":
    reid = ReIDLoader()