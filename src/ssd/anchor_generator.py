# -*- coding: utf-8 -*-

# @Time   : 2019/9/9:11:17
# @Author : xuqiang

import tensorflow as tf


def create_scales_and_ratios(num_layers = 6,
                             min_scale=0.2,
                             max_scale=0.95,
                             aspect_ratios=[1.0, 2.0, 0.5, 3.0, 0.3333]
                             ):
    scales = [min_scale + (max_scale - min_scale) * i / (num_layers - 1)
              for i in range(num_layers)] + [1.0]

    total_scales = list()
    total_ratios = list()

    interp_scale_aspect_ratio = 1.0
    for layer, scale, scale_next in zip(range(num_layers), scales[:-1], scales[1:]):
        layer_scales = list()
        layer_ratios = list()

        if layer == 0:
            layer_scales = [0.1, scale, scale]
            layer_ratios = [1.0, 2.0, 0.5]
        else:
            for aspect_ratio in aspect_ratios:
                layer_scales.append(scale)
                layer_ratios.append(aspect_ratio)
            layer_scales.append(tf.sqrt(scale * scale_next))
            layer_ratios.append(interp_scale_aspect_ratio)

        total_scales.append(layer_scales)
        total_ratios.append(layer_ratios)

    return total_scales, total_ratios

def tile_anchors(grid_width,
                  grid_height,
                  scales,
                  aspect_ratios,
                  anchor_stride,
                  anchor_offset,
                  base_anchor_size=[1, 1]):
        '''
        '''

        ratio_sqrts = tf.sqrt(aspect_ratios)
        widths = scales * ratio_sqrts * base_anchor_size[0]
        heights = scales / ratio_sqrts * base_anchor_size[1]

        x_centers = tf.range(grid_width, dtype=tf.float32)
        x_centers = x_centers * anchor_stride[0] + anchor_offset[0]
        y_centers = tf.range(grid_height, dtype=tf.float32)
        y_centers = y_centers * anchor_stride[1] + anchor_offset[1]
        x_centers, y_centers = tf.meshgrid(x_centers, y_centers)

        width_grid, x_center_grid = tf.meshgrid(widths, x_centers)
        height_grid, y_center_grid = tf.meshgrid(heights, y_centers)

        x_center_grid = tf.reshape(x_center_grid, (-1, 1))
        y_center_grid = tf.reshape(y_center_grid, (-1, 1))
        width_grid = tf.reshape(width_grid, (-1, 1))
        height_grid = tf.reshape(height_grid, (-1, 1))

        x_min_grid = x_center_grid - 0.5 * width_grid
        y_min_grid = y_center_grid - 0.5 * height_grid
        x_max_grid = x_center_grid + 0.5 * width_grid
        y_max_grid = y_center_grid + 0.5 * height_grid

        # 切换anchor到[y, x, h, w]
        # boxs = tf.concat([y_center_grid, x_center_grid, height_grid, width_grid], axis=1)
        boxs = tf.concat([y_min_grid, x_min_grid, y_max_grid, x_max_grid], axis=1)

        return boxs

def generate_anchors_per_lications(num_layers = 6):
    num_anchors_per_location = list()

    total_scales, total_ratios = create_scales_and_ratios(num_layers=num_layers)
    for scales in total_scales:
        num_anchors_per_location.append(len(scales))
    return num_anchors_per_location


def generate_anchors(feature_map_shapes):
    """生成多尺度的anchor，确定图像大小和特征图大小以后生成就不变
                """
    total_scales, total_ratios = create_scales_and_ratios(num_layers=len(feature_map_shapes))

    anchor_strides = [(1.0 / float(pair[0]), 1.0 / float(pair[1])) for pair in feature_map_shapes]
    anchor_offsets = [(0.5 * stride[0], 0.5 * stride[1]) for stride in anchor_strides]

    anchors = list()
    for grid_size, scales, aspect_ratios, anchor_stride, anchor_offset \
            in zip(feature_map_shapes, total_scales, total_ratios, anchor_strides, anchor_offsets):
        tail_anchors = tile_anchors(grid_size[0], grid_size[1], scales,
                                          aspect_ratios, anchor_stride, anchor_offset)

        for idx in range(tail_anchors.shape[0]):
            tail_anchor = tail_anchors[idx]
            anchors.append(tail_anchor)

    anchors = tf.stack(anchors)
    return anchors
