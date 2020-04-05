# -*- coding: utf-8 -*-

# @Time   : 2020/3/4:18:45
# @Author : xuqiang

import tensorflow as tf
import functools

from src.utils.standard_fields import InputDataKeys

def build_padding(padding_config):

    num_class = padding_config["num_class"]
    num_class_with_background = num_class + 1
    max_num_boxes = padding_config["max_num_boxes"]
    resized_shape = (padding_config["height"], padding_config["width"])

    func = functools.partial(pad_input_data_to_static_shapes,
                             num_class_with_background=num_class_with_background,
                             max_num_boxes=max_num_boxes,
                             resized_shape=resized_shape)

    return func


def pad_input_data_to_static_shapes(tensor_dict,
                                    num_class_with_background,
                                    max_num_boxes,
                                    resized_shape):
    height, width = resized_shape

    if InputDataKeys.image in tensor_dict:
        num_channels = tensor_dict[InputDataKeys.image].shape[2]
    else:
        num_channels = 3

    padding_shapes = {
        InputDataKeys.image: [height, width, num_channels],
        InputDataKeys.groundtruth_classes: [max_num_boxes, 1],
        InputDataKeys.groundtruth_bboxes: [max_num_boxes, 4],
        InputDataKeys.pids: [max_num_boxes, 1]
    }

    padding_tensor_dict = dict()
    for tensor_name in tensor_dict:

        if tensor_name not in padding_shapes:
            padding_tensor_dict[tensor_name] = tensor_dict[tensor_name]
            continue

        padding_tensor_dict[tensor_name] = pad_or_clip_nd(tensor_dict[tensor_name],
                                                          padding_shapes[tensor_name])

    return padding_tensor_dict

def pad_or_clip_nd(tensor, ouput_shape):

    tensor_shape = tf.shape(tensor)

    # 如果shape一致，直接返回
    # the_same = True
    # for idx, shape in enumerate(ouput_shape):
    #     the_same = the_same and (shape == tensor_shape.numpy()[idx])
    #
    # if(the_same):
    #     return tensor

    # clip
    clip_size = [
        tf.where(tensor_shape[i] - shape > 0, shape, -1)
        if shape is not None else -1 for i, shape in enumerate(ouput_shape)
    ]

    clipped_tensor = tf.slice(tensor,
                              begin = tf.zeros(len(clip_size), dtype=tf.int32),
                              size = clip_size)

    # padding
    clipped_tensor_shape = tf.shape(clipped_tensor)
    trailing_paddings = [
        shape - clipped_tensor_shape[i] if shape is not None else 0
        for i, shape in enumerate(ouput_shape)
    ]
    paddings = tf.stack(
        [
            tf.zeros(len(trailing_paddings), dtype=tf.int32),
            trailing_paddings
        ],
    axis = 1)
    padded_tensor = tf.pad(clipped_tensor, paddings=paddings)

    return padded_tensor



