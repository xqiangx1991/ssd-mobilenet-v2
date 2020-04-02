# -*- coding: utf-8 -*-

# @Time   : 2020/3/4:14:04
# @Author : xuqiang

from src.dataset.augment_functions import random_horizontal_flip_image
from src.dataset.augment_functions import ssd_random_crop
import functools

random_horizontal_flip_name = "random_horizontal_flip"
ssd_random_crop_name = "ssd_random_crop"

def build_image_augument(augument_config):
    '''

    :param augument_config:
    :return:
    '''
    # 会有多种的增强函数，把这些函数结合起来
    def process_fn(tensor_dict,funcs):
        for func in funcs:
            tensor_dict = func(tensor_dict)
        return tensor_dict

    funcs = list()
    if random_horizontal_flip_name in augument_config:
        func = functools.partial(random_horizontal_flip_image)
        funcs.append(func)

    if ssd_random_crop_name in augument_config:
        func = functools.partial(ssd_random_crop)
        funcs.append(func)

    total_func = functools.partial(process_fn, funcs=funcs)
    return total_func