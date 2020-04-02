# -*- coding: utf-8 -*-

# @Time   : 2020/3/3:22:02
# @Author : xuqiang

import tensorflow as tf
from src.utils.standard_fields import InputDataKeys
from src.dataset.resize_builder import build_image_resizer
from src.dataset.augument_builder import build_image_augument
from src.dataset.padding_builder import build_padding


class DataTransform():
    '''
    preprocess
    resize
    augumentation(training)
    padding
    '''

    def __init__(self,
                 preprocess_fn,
                 configs):
        self.preprocessor = preprocess_fn

        self.ops = list()

        if "augument" in configs:
            augument_config = configs['augument']
            self.ops.append(self.get_augument(augument_config))

        if "resize" in configs:
            resize_config = configs['resize']
            self.ops.append(self.get_resize(resize_config))

        if 'padding' in configs:
            padding_config = configs['padding']
            self.ops.append(self.get_padding(padding_config))



    def transform(self, tensor_dict):
        image = tensor_dict[InputDataKeys.image]
        image = tf.cast(image, tf.float32)
        image = self.preprocessor(image)
        tensor_dict[InputDataKeys.image] = image

        for op in self.ops:
            tensor_dict = op(tensor_dict)

        return tensor_dict

    def get_resize(self, resize_config):
        return build_image_resizer(resize_config)

    def get_augument(self, augument_config):
        return build_image_augument(augument_config)

    def get_padding(self, padding_config):
        return build_padding(padding_config)

