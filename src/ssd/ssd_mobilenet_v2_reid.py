# -*- coding: utf-8 -*-

# @Time   : 2020/3/13:15:33
# @Author : xuqiang
import tensorflow as tf
from src.ssd.ssd_mobilenet_v2 import SSDMobileNetV2
from src.ssd.ssd_mobilenet_v2 import SSDPredictor

from src.layer.common import Conv2DBlock
from src.layer.common import DepthwiseConv2D
from src.layer.common import truncated_normal_initializer

from src.utils.standard_fields import DetectionKeys
from src.utils.standard_fields import InputDataKeys
from src.utils.standard_fields import LossKeys

from src.layer.oim_layer import LabeledMatching
from src.layer.oim_layer import UnLabeledMatching

class SSDMobileNetV2ReID(SSDMobileNetV2):
    def __init__(self,
                 num_class,
                 reid_dim,
                 inputs_shape,
                 use_depthwise = False,
                 **kwargs):
        super().__init__(num_class,
                         inputs_shape,
                         use_depthwise,
                         **kwargs)

        # reid计算构建
        self.reid_dims = reid_dim
        self.num_reid = 7929
        self.reid_unlabeled_queue_size = 3000
        self.reid_scalar = 0.07

        self.labeled_oim = LabeledMatching(capacity=self.num_reid,
                                           feature_size=self.reid_dims,
                                           scalar=self.reid_scalar)
        self.unlabeled_oim = UnLabeledMatching(capacity=self.reid_unlabeled_queue_size,
                                               feature_size=self.reid_dims,
                                               scalar=self.reid_scalar)

        # 增加reid分支
        self.reid_op = ReIDPredictor(anchors_per_locations=self.anchors_per_lications,
                                     reid_dims=self.reid_dims,
                                     use_depthwise=use_depthwise)

    def call(self, inputs, training=False, mask = None):
        extend_features = self.infe_extend_features(inputs, training)
        label_features, bboxes_features = self.ssd_predictor(extend_features, training)
        reid_output = self.reid_op(extend_features, training)

        outputs = {
            DetectionKeys.detection_classes:label_features,
            DetectionKeys.detection_bboxes:bboxes_features,
            DetectionKeys.reid_feature:reid_output
        }

        return outputs


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
        labels_loss_ave = tf.divide(tf.reduce_sum(labels_loss), num_positive)
        bboxes_loss_ave = tf.divide(tf.reduce_sum(bboxes_loss), num_positive)

        output_dict = {
            LossKeys.labels_loss: labels_loss_ave,
            LossKeys.bboxes_loss: bboxes_loss_ave
        }

        # 计算reid loss
        target_assign_pids = target_assign_output[InputDataKeys.pids]
        predict_reid_features = predict_dict[DetectionKeys.reid_feature]
        reid_loss = self._calculate_reid_loss(predict_reid_features, target_assign_pids)
        num_pids = tf.maximum(tf.reduce_sum(tf.cast(target_assign_pids > -1, tf.float32)), 1)
        reid_loss_ave = tf.reduce_sum(reid_loss) / num_pids
        output_dict[LossKeys.pids_loss] = reid_loss_ave

        return output_dict


    def update_oim(self):
        # 更新参数
        self.labeled_oim.update_weight()
        # self.unlabeled_oim.update_weight()


    def _calculate_reid_loss(self,reid_features, target_person_id):

        # 从特征中构建出分数
        labeled_matching = self.labeled_oim(reid_features, target_person_id)
        # unlabeled_matching = self.unlabeled_oim(reid_features, target_person_id)

        # oim_scores = tf.concat([labeled_matching, unlabeled_matching], axis=-1)
        oim_scores = labeled_matching
        oim_loss = self._reid_subsample_v2(oim_scores, target_person_id, k=100)
        pid_loss = oim_loss

        return pid_loss


    def _reid_nosample(self, oim_scores, reid_pids):

        oim_softmax_scores = tf.nn.softmax(oim_scores, axis=-1)
        #
        reid_pids_squeeze = tf.squeeze(reid_pids, -1)

        valid_pids_index = tf.where(tf.greater_equal(reid_pids_squeeze, 0))
        valid_pids = tf.gather_nd(reid_pids_squeeze, valid_pids_index)
        valid_pids_index = tf.cast(valid_pids_index, tf.int32)

        valid_score_index = tf.stack(tf.unstack(valid_pids_index, axis=1) + [valid_pids], 1)
        valid_scores = tf.gather_nd(oim_softmax_scores, valid_score_index)
        # 只有加了log操作以后才是越接近于1，loss越等于0
        valid_scores = - tf.math.log(valid_scores)

        pid_scores = tf.scatter_nd(valid_pids_index, valid_scores, tf.shape(reid_pids_squeeze))

        return pid_scores

    def _reid_subsample(self, reid_scores, reid_pids, k=50):
        '''
        在计算softmax的时候，其实不用从这么多的id中去找，能够把这个id和100个id分开已经很好了
        '''
        reid_pids = tf.squeeze(reid_pids, axis=-1)
        # 找出所有有效的anchor
        valid_anchor_indice = tf.where(tf.greater_equal(reid_pids, 0))

        selected_scores_list = tf.TensorArray(dtype=reid_scores.dtype, size=len(valid_anchor_indice))

        for idx in range(len(valid_anchor_indice)):
            valid_anchor_index = valid_anchor_indice[idx]
            batch_idx = valid_anchor_index[0]
            anchor_idx = valid_anchor_index[1]
            batch_pid = reid_pids[batch_idx, anchor_idx]

            batch_scores = reid_scores[batch_idx, anchor_idx]

            # 不能随机选取，应该把分数值大的选出来，选择有效的负样本
            val, ind = tf.nn.top_k(batch_scores, k=k)

            # 是否存在positive index
            exist_pos_index = tf.cast((ind == tf.expand_dims(batch_pid, axis=-1)), tf.int32)

            exclude_pos_index = tf.concat([ind[:-1], [batch_pid]], axis=0)
            sample_indice = ind * exist_pos_index + exclude_pos_index

            # test_indice = tf.concat([ind[:-1], [batch_pid]], axis=0)
            # test_indice,_ = tf.unique(test_indice)
            #
            # if(len(test_indice) < k -1):
            #     # 正样本已经在里面了
            #     sample_indice = ind
            # else:
            #     # 负样本+正样本
            #     sample_indice = test_indice
            # 把这些分数抽离出来计算loss
            selected_scores = tf.gather(batch_scores, sample_indice)
            # 计算softmax
            selected_scores_softmax = tf.nn.softmax(selected_scores)

            # 找到正样本在哪里
            positive_pid_idx = tf.squeeze(tf.where(sample_indice == batch_pid))

            # 获取pid对应的分数
            selected_pid_score = - tf.math.log(selected_scores_softmax[positive_pid_idx])
            selected_scores_list = selected_scores_list.write(idx, selected_pid_score)

        selected_scores_list = selected_scores_list.stack()
        selected_scores = tf.scatter_nd(valid_anchor_indice, selected_scores_list,
                                        tf.cast(tf.shape(reid_pids), tf.int64))

        return selected_scores

    def _reid_subsample_v2(self, reid_scores, reid_pids, k=50):

        squeeze_reid_pids = tf.squeeze(reid_pids, axis=-1)

        # 先找到有效的anchors [batch, anchor]
        valid_anchor_index = tf.where(tf.greater_equal(squeeze_reid_pids, 0))
        valid_anchor_index = tf.cast(valid_anchor_index, tf.int32)

        # 把对应index的scores和pid都选出来
        selected_anchor_scores = tf.gather_nd(reid_scores, valid_anchor_index)
        selected_pids = tf.gather_nd(squeeze_reid_pids, valid_anchor_index)
        selected_pid_scores_index = tf.stack(tf.unstack(valid_anchor_index, axis=1) + [selected_pids], 1)
        selected_pid_scores = tf.gather_nd(reid_scores, selected_pid_scores_index)
        selected_pid_scores = tf.math.exp(selected_pid_scores)

        # 选取前topk的value和index
        topk_values, topk_index = tf.nn.top_k(selected_anchor_scores, k=k)

        # 这些topk_index中是否有正样本的id
        exist_positive_pid = (topk_index == tf.expand_dims(selected_pids, axis=-1))
        exist_positive_pid = tf.cast(tf.reduce_sum(tf.cast(exist_positive_pid, tf.int32), axis=-1) > 0, tf.float32)

        # 是否存在正样本，其最后的分母不一样
        topk_values = tf.math.exp(topk_values)
        softmax_pos_denominator = tf.reduce_sum(topk_values, axis=-1)
        softmax_no_pos_denominator = tf.reduce_sum(topk_values[:, :-1], axis=-1) + selected_pid_scores

        # 如果已经存在positive pid，那就是这个index和value就可以，如果不存在，value的那行就需要加上
        softmax_denominator = softmax_pos_denominator * exist_positive_pid \
                              + softmax_no_pos_denominator * (1 - exist_positive_pid)

        softmax_scores = selected_pid_scores / softmax_denominator

        # top1的正确率
        correct_selected_score_count = (topk_index[:,0] == selected_pids)
        correct_selected_score_count = tf.reduce_sum(tf.cast(correct_selected_score_count, tf.int32))
        step = tf.summary.experimental.get_step()
        if step is not None and (step % 100 == 0):
            tf.summary.scalar("train/positive_selected_count", correct_selected_score_count, step=step)

        # 计算loss
        loss = - tf.math.log(softmax_scores)
        selected_loss = tf.scatter_nd(valid_anchor_index, loss, tf.shape(squeeze_reid_pids))

        return selected_loss

class ReIDPredictor(tf.keras.layers.Layer):

    def __init__(self,
                 anchors_per_locations,
                 reid_dims,
                 use_depthwise,
                 **kwargs):
        super().__init__(**kwargs)

        kernel_size = 1 if not use_depthwise else 3
        strides = 1
        self.ops = list()
        for anchors_per_location in anchors_per_locations:
            filters = reid_dims * anchors_per_location
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
        self.reshape = tf.keras.layers.Reshape((-1, reid_dims))
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
        outputs = tf.math.l2_normalize(outputs, axis=-1)
        return outputs

