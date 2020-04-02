# -*- coding: utf-8 -*-

# @Time   : 2019/9/18:15:28
# @Author : xuchen.xq

from src.ssd.ssd_mobilenet_v2 import SSDMobileNetV2
from src.ssd.ssd_mobilenet_v2_reid import SSDMobileNetV2ReID

def build_model(model_config):

    meta_architecture = model_config['type']
    if meta_architecture == "ssd":
        return _build_ssd_model(model_config)

    raise ValueError("Unknown meta architecture {}".format(meta_architecture))



def _build_ssd_model(ssd_config):

    num_class = ssd_config['num_class']
    resize_height = ssd_config["height"]
    resize_width = ssd_config['width']
    resize_channel = ssd_config["channel"]
    use_depthwise = ssd_config['use_depthwise']
    image_shape = (resize_height, resize_width, resize_channel)
    input_shape = (None, resize_height, resize_width, resize_channel)

    use_reid = ssd_config.get('reid', False)
    if use_reid:
        reid_dim = ssd_config.get('reid_dim', 256)
        ssdmobilenetv2 = SSDMobileNetV2ReID(num_class=num_class,
                                            reid_dim=reid_dim,
                                            inputs_shape=image_shape,
                                            use_depthwise=use_depthwise)
    else:
        ssdmobilenetv2 = SSDMobileNetV2(num_class=num_class,
                                        inputs_shape=image_shape,
                                        use_depthwise=use_depthwise)

    ssdmobilenetv2.build(input_shape)

    return ssdmobilenetv2

