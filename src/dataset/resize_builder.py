# -*- coding: utf-8 -*-

# @Time   : 2019/9/17:20:25
# @Author : xuchen.xq

import tensorflow as tf
import functools

from src.utils.standard_fields import InputDataKeys

def build_image_resizer(resizer_config):

    def resize(tensor_dict, method, size):
        image = tensor_dict[InputDataKeys.image]
        image = tf.image.resize(image, size, method)
        tensor_dict[InputDataKeys.image] = image
        return tensor_dict

    resizer = None
    method_dict = {
        "BILINEAR":
            tf.image.ResizeMethod.BILINEAR,
        "NEAREST_NEIGHBOR":
            tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        "BICUBIC":
            tf.image.ResizeMethod.BICUBIC,
        "AREA":
            tf.image.ResizeMethod.AREA
    }

    type = resizer_config['type']
    if type == 'fixed_shape_resizer':
        resize_method = resizer_config["method"]
        resize_width = resizer_config["width"]
        resize_height = resizer_config['height']

        if not resize_method:
            resize_method = "BILINEAR"
        resize_method = resize_method.upper()

        if resize_method in method_dict:
            method = method_dict[resize_method]
        else:
            raise ValueError("Unkown resize method {}".format(resize_method))

        resizer = functools.partial(
            resize,
            size=(resize_height, resize_width),
            method=method
        )

    if resizer is None:
        raise ValueError("No resize function")

    return resizer