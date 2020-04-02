# -*- coding: utf-8 -*-

# @Time   : 2019/11/11:19:52
# @Author : xuqiang

import tensorflow as tf

from src.utils.standard_fields import DetectionKeys
from src.utils.standard_fields import InputDataKeys
from src.utils.standard_fields import LossKeys

from src.layer.common import Conv2DBlock
from src.layer.common import DepthwiseConv2D
from src.mobilenet.mobilenet_v2 import MobileNetV2

from src.ssd.anchor_generator import generate_anchors_per_lications
from src.ssd.anchor_generator import generate_anchors
from src.layer.common import non_max_suppression_with_scores

from src.ssd.box_coder import BoxCoder
from src.ssd.target_assign import TargetAssign
from src.ssd.hard_example_mining import HardExampleMining

from src.layer.common import truncated_normal_initializer


class SSDMobileNetV2(tf.keras.Model):

    def __init__(self,
                 num_class,
                 inputs_shape,
                 use_depthwise = False,
                 **kwargs):
        super().__init__(**kwargs)

        self.num_class = num_class
        self.num_class_with_background = num_class + 1
        self.use_depthwise = use_depthwise
        self.inputs_shape = inputs_shape
        # 让属于网络的归网络，其他处理的归其他处理
        # 考虑可扩展性，比如增加reid的功能

        # backbone，但是其中的操作不是全用的，从分支中引入了一下
        self.backbone = MobileNetV2()
        self.ssd_extend = SSDFeatureExpandLayer(self.use_depthwise)

        self.anchors_per_lications = generate_anchors_per_lications()

        # 生成预测分支
        self.ssd_predictor = SSDPredictor(anchors_per_locations=self.anchors_per_lications,
                                          num_class_with_background=self.num_class_with_background,
                                          use_depthwise=self.use_depthwise)
        # 生成其他工具
        # 尝试进行一次推理
        self.hard_example_mining = HardExampleMining()
        self.calc_for_anchor_tools(inputs_shape)



    def call(self, inputs, training=False, mask=None):
        # 第一层的特征
        extend_features = self.infe_extend_features(inputs, training)
        labels_feature, bboxes_feature = self.ssd_predictor(extend_features, training)
        outputs = {
            DetectionKeys.detection_classes:labels_feature,
            DetectionKeys.detection_bboxes:bboxes_feature
        }

        return outputs

    def infe_extend_features(self, inputs, training=False):
        mbn_branch1, mbn_branch2 = self.backbone(inputs, training=training)
        extend_features = self.ssd_extend(mbn_branch2, training=training)

        extend_features = [mbn_branch1] + extend_features
        return extend_features

    def calc_for_anchor_tools(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        extend_features = self.infe_extend_features(inputs)

        feature_map_shapes = list()
        for feature_map in extend_features:
            feature_map_shape = feature_map.shape
            feature_map_shapes.append(feature_map_shape[1:3])

        self.anchors = generate_anchors(feature_map_shapes)
        self.box_coder = BoxCoder(self.anchors)
        self.target_assignor = TargetAssign(self.anchors)


    def preprocess(self, images):
        '''
        预处理
        :return:
        '''
        images = (2.0 / 255.0) * images - 1.0
        return images


    def postprocess(self,
                    predict_labels,
                    predict_bboxes,
                    score_threshold = 1e-8,
                    iou_threshold = 0.6,
                    max_detections_per_class = 100,
                    max_total_detections = 100):
        # 按照batch分别处理
        # 这部分主要用的就是bbox和label，其他的都是原封不动即可
        bboxes = self.box_coder.decode(predict_bboxes)
        scores = tf.sigmoid(predict_labels)
        # 剔除background
        scores = scores[:,:,1:]

        batch_size = tf.shape(predict_labels)[0]
        masks = tf.TensorArray(dtype=tf.float32, size=batch_size)
        for idx in range(batch_size):
            bbox = bboxes[idx]
            # 按照num_class进行处理
            # 每一个class是一列
            cls_indice = tf.TensorArray(dtype=tf.int32, size=self.num_class)
            cls_scores = tf.TensorArray(dtype=tf.float32, size=self.num_class)
            for cls_idx in range(self.num_class):
                score = scores[idx,:,cls_idx]
                max_select_size = tf.minimum(len(bbox), max_detections_per_class)
                selected_index, selected_scores = \
                    non_max_suppression_with_scores(bbox, score, max_select_size, iou_threshold, score_threshold)

                # 按照分数要进行重排，只选取其中的一部分，因此需要把分数和index都拿出来
                # 构建index和score
                # 对index和score进行扩充
                # index应该把对应的cls_index加进去
                num_selected = tf.shape(selected_index)[0]
                selected_index = tf.concat(
                    [selected_index, tf.zeros((max_select_size - num_selected), tf.int32)],
                    axis=0
                )
                selected_index = tf.stack([selected_index,
                                           tf.ones((max_select_size), tf.int32) * cls_idx],
                                          axis=1)


                selected_scores = tf.concat(
                    [selected_scores, tf.zeros((max_select_size - num_selected), tf.float32)],
                    axis=0
                )

                cls_indice = cls_indice.write(cls_idx, selected_index)
                cls_scores = cls_scores.write(cls_idx, selected_scores)

            cls_indice = cls_indice.concat()
            cls_scores = cls_scores.concat()

            top_values, top_index = tf.math.top_k(cls_scores,max_total_detections)
            # 选择其中的index
            selected_index_final = tf.gather(cls_indice,
                        tf.squeeze(tf.where(top_values > score_threshold),axis=-1))

            # 按照分数排序只取前几个
            mask = tf.scatter_nd(selected_index_final,
                                 tf.ones((tf.shape(selected_index_final)[0],), tf.float32),
                                 tf.shape(scores[idx]))


            # 合并所有的类别
            masks = masks.write(idx, mask)

        masks = masks.stack()
        bboxes = bboxes * masks
        scores = scores * masks

        output_dict = dict()
        output_dict[DetectionKeys.detection_classes] = masks
        # 把bbox全部弄成0
        output_dict[DetectionKeys.detection_bboxes] = bboxes
        output_dict[DetectionKeys.detection_scores] = scores

        return output_dict

    def calculate_loss(self, groundtruth_dict,predict_dict):
        target_assign_output = self.target_assign(groundtruth_dict)
        predict_labels, predict_bboxes = predict_dict[DetectionKeys.detection_classes], \
                predict_dict[DetectionKeys.detection_bboxes]
        predict_bboxes_decode = self.box_coder.decode(predict_bboxes)

        target_assign_labels = target_assign_output[InputDataKeys.groundtruth_classes]
        target_assign_bboxes = target_assign_output[InputDataKeys.groundtruth_bboxes]

        # 计算类别loss
        labels_loss = self._calculate_label_loss(target_assign_labels, predict_labels)
        hard_example_mining_mask = self.hard_example_mining.mining(target_assign_labels,
                                                                   predict_bboxes_decode,
                                                                   labels_loss)

        # 计算位置loss
        positive_mask = tf.squeeze(tf.cast(tf.greater(target_assign_labels, 0), tf.float32), -1)
        num_positive = tf.maximum(tf.reduce_sum(tf.cast(positive_mask, tf.float32)), 1)

        bboxes_loss = self._calculate_bbox_loss(target_assign_bboxes, predict_bboxes, positive_mask)
        labels_loss = labels_loss * hard_example_mining_mask
        bboxes_loss = bboxes_loss * hard_example_mining_mask
        labels_loss_ave = tf.divide(tf.reduce_sum(labels_loss),num_positive)
        bboxes_loss_ave = tf.divide(tf.reduce_sum(bboxes_loss), num_positive)

        output_dict = {
            LossKeys.labels_loss:labels_loss_ave,
            LossKeys.bboxes_loss:bboxes_loss_ave
        }

        return output_dict

    def target_assign(self, groundtruth_dict):
        target_assign_output = self.target_assignor.assign(groundtruth_dict)
        return target_assign_output


    def _calculate_label_loss(self, target_labels, predict_logits):

        target_class = tf.squeeze(target_labels, -1)
        target_class = tf.cast(target_class, tf.int32)
        target_class_one_hot = tf.one_hot(target_class, self.num_class_with_background, dtype=tf.float32)
        sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(target_class_one_hot, predict_logits)

        loss = tf.reduce_sum(sigmoid_loss, -1)

        return loss

    def _calculate_bbox_loss(self, target_assign_bboxes, predict_bboxes, mask = None):
        target_assign_bboxes_encode = self.box_coder.encode(target_assign_bboxes)
        absx = tf.abs(target_assign_bboxes_encode - predict_bboxes)
        minx = tf.minimum(absx, 1)
        loss = 0.5 * ((absx - 1) * minx + absx)
        # [batch_size, anchors]
        loss = tf.reduce_sum(loss, axis=-1)

        if mask is not None:
            loss = loss * mask

        return loss


# ========== Feature Expand ==========

class HeadOp(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_size,
                 filters,
                 stride,
                 use_bias,
                 ac,
                 bn,
                 use_depthwise = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.ops = list()

        if use_depthwise:
            dp1 = DepthwiseConv2D(
                kernel_size=kernel_size,
                strides=stride,
                padding='SAME',
                use_bias=False
            )

            cv1 = Conv2DBlock(
                filters=filters,
                kernel_size=1,
                strides=1,
                padding='SAME',
                use_bias=use_bias,
                ac=ac,
                bn=bn
            )
            self.ops.append(dp1)
            self.ops.append(cv1)
        else:
            cv2 = Conv2DBlock(
                filters=filters,
                kernel_size=kernel_size,
                strides=stride,
                padding='SAME',
                use_bias=use_bias,
                ac = ac,
                bn = bn,
                kernel_initializer=truncated_normal_initializer()
            )
            self.ops.append(cv2)

    def call(self, inputs, training = False):
        outputs = inputs
        for op in self.ops:
            outputs = op(outputs, training)
        return outputs

class SSDConv2DStack(tf.keras.layers.Layer):

    def __init__(self,
                 depth,
                 use_depthwise = False,
                 **kwargs):
        super().__init__(**kwargs)

        self.ops = list()

        name0 = "{}_1_1x1_{}".format(self.name, int(depth/2))
        conv0 = Conv2DBlock(
            filters=int(depth/2),
            kernel_size=1,
            strides=1,
            padding='SAME',
            use_bias=False,
            kernel_initializer=truncated_normal_initializer(),
            name=name0
        )
        self.ops.append(conv0)

        name1 = "{}_2_3x3_{}".format(self.name, depth)
        op2 = HeadOp(kernel_size=3,
                     filters=depth,
                     stride=2,
                     use_bias=False,
                     ac = False,
                     bn = False,
                     use_depthwise=use_depthwise,
                     name=name1)
        self.ops.append(op2)

    def call(self, inputs, training=False):
        outputs = inputs
        for op in self.ops:
            outputs = op(outputs, training)
        return outputs



class SSDFeatureExpandLayer(tf.keras.layers.Layer):
    '''特征提取
    '''

    def __init__(self,
                 use_depthwise=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.use_depthwise = use_depthwise
        expand_depthes = [512, 256, 256, 128]

        # 生成一系列的ops
        self.ops = list()
        for expand_depth in expand_depthes:
            name_ext = "{}_{}".format(self.name, expand_depth)
            op = SSDConv2DStack(depth=expand_depth,
                                use_depthwise=self.use_depthwise,
                                name=name_ext)
            self.ops.append(op)

    def call(self, inputs, training=False):
        outputs = inputs
        total_outputs = list()
        total_outputs.append(outputs)
        for op in self.ops:
            outputs = op(outputs, training)
            total_outputs.append(outputs)
        return total_outputs


# ============ Predictor ============
class LableBboxPredictor(tf.keras.layers.Layer):

    def __init__(self,
                 anchors_per_locations,
                 code_size,
                 use_depthwise = False,
                 **kwargs):
        super().__init__(**kwargs)

        kernel_size = 1 if not use_depthwise else 3
        strides = 1

        self.ops = list()
        for anchors_per_location in anchors_per_locations:
            filters = code_size * anchors_per_location

            if use_depthwise:
                dp1 = DepthwiseConv2D(
                    kernel_size=kernel_size,
                    strides=strides,
                    padding='SAME',
                    use_bias=False
                )

                cv1 = Conv2DBlock(
                    filters=filters,
                    kernel_size=1,
                    strides=1,
                    padding='SAME',
                    use_bias=True,
                    ac=False,
                    bn=False
                )
                self.ops.append([dp1, cv1])
            else:
                cv2 = Conv2DBlock(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding='SAME',
                    use_bias=True,
                    ac=False,
                    bn=False,
                    kernel_initializer=truncated_normal_initializer()
                )
                self.ops.append([cv2])

        self.reshape = tf.keras.layers.Reshape((-1, code_size))
        self.concatenate = tf.keras.layers.Concatenate(axis=1)

    def call(self, inputs, training = False):
        features = inputs
        if (len(features) != len(self.ops)):
            raise ValueError("feature map length not match ops length")

        outputs = list()
        for idx in range(len(features)):
            feature = features[idx]
            op = self.ops[idx]
            for p in op:
                feature = p(feature)
            feature = self.reshape(feature)
            outputs.append(feature)
        outputs = self.concatenate(outputs)
        return outputs



class SSDPredictor(tf.keras.layers.Layer):

    def __init__(self,
                 anchors_per_locations,
                 num_class_with_background,
                 box_code_size = 4,
                 use_depthwise = False,
                 **kwargs):
        super().__init__(**kwargs)

        self.bboxes_op = LableBboxPredictor(anchors_per_locations=anchors_per_locations,
                                            code_size=box_code_size,
                                            use_depthwise = use_depthwise)

        self.labels_op = LableBboxPredictor(anchors_per_locations=anchors_per_locations,
                                            code_size=num_class_with_background,
                                            use_depthwise=use_depthwise)


    def call(self, inputs, training = False):

        label_features = self.labels_op(inputs)
        bbox_features = self.bboxes_op(inputs)

        return label_features, bbox_features



if __name__ == "__main__":
    input_shape = (300,300,3)
    ssd = SSDMobileNetV2(2, input_shape)
    ssd.summary()




