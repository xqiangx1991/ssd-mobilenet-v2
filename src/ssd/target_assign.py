
import tensorflow as tf
from src.ssd.target_assign_util import iou
from src.ssd.target_assign_util import set_value_use_index
from src.utils.standard_fields import InputDataKeys

class TargetAssign():

    def __init__(self,
                 anchors,
                 iou_threshold = 0.5):
        super(TargetAssign, self).__init__()
        self._anchors = anchors
        self._iou_threshold = iou_threshold

    def assign(self, groundtruth_dict):
        bboxes = groundtruth_dict[InputDataKeys.groundtruth_bboxes]
        labels = groundtruth_dict[InputDataKeys.groundtruth_classes]

        jaccard = iou(bboxes, self._anchors)
        mask = self._compute_anchor_bbox_index(jaccard, labels)

        labels_output = self._compute_target_mask(labels, mask)
        bboxes_output =  self._compute_target_mask(bboxes, mask)
        scores_output = self._compute_scores_mask(jaccard, mask)

        outputs = {
            InputDataKeys.groundtruth_classes:labels_output,
            InputDataKeys.groundtruth_bboxes:bboxes_output,
            InputDataKeys.score:scores_output
        }

        if InputDataKeys.pids in groundtruth_dict.keys():
            # pid的计算稍微麻烦一点
            pids = groundtruth_dict[InputDataKeys.pids]
            pids_output = self._compute_target_mask(pids, mask, -2)
            outputs[InputDataKeys.pids] = pids_output

        return outputs

    def _compute_anchor_bbox_index(self, jaccard, labels):
        ''' 计算出每一个anchor对应的bbox的索引，如果没有对应的索引就置为-1
        应该要基于这个索引把对应的其他重组
        '''
        # [batch, bbox, anchor]
        # 给每一个满足条件的anchor都分配一个bbox
        matches = tf.argmax(jaccard,1)
        matched_values = tf.reduce_max(jaccard, 1)
        below_unmatched_threshold = tf.less(matched_values, self._iou_threshold)
        matches = set_value_use_index(matches, below_unmatched_threshold, -1)

        # 给每一个bbox都分配至少一个anchor
        force_match_rows = tf.argmax(jaccard,2)
        force_match_rows = set_value_use_index(force_match_rows, tf.squeeze(labels,-1) <=0, -1)
        force_match_rows_indicator = tf.one_hot(force_match_rows, len(self._anchors), axis=-1)
        force_match_rows_id = tf.argmax(force_match_rows_indicator, 1)
        force_match_rows_mask = tf.cast(
            tf.reduce_max(force_match_rows_indicator, 1), tf.bool
        )

        # [batch, anchor]
        final_matches = tf.where(force_match_rows_mask,
                                 force_match_rows_id, matches)

        final_matches = tf.cast(final_matches, labels.dtype)

        return final_matches


    def _compute_target_mask(self, targets, masks, bg_value = 0):
        batch_size = targets.shape[0]

        outputs = tf.TensorArray(targets.dtype, size=batch_size)
        for batch in range(batch_size):
            mask = masks[batch]
            fore_mask = tf.cast(tf.greater_equal(mask, 0), mask.dtype)
            bg_mask = tf.cast(tf.less(mask, 0), mask.dtype)
            mask = mask * fore_mask
            target = targets[batch]

            bg_mask = tf.reshape(bg_mask, [-1, 1])

            # 把每一个mask对应的label填充进去
            output = tf.gather(target, mask)
            output = set_value_use_index(output, bg_mask, bg_value)
            outputs = outputs.write(batch, output)
        outputs = outputs.stack()
        return outputs

    def _compute_scores_mask(self, jaccard, masks):
        batch_size = jaccard.shape[0]

        outputs = tf.TensorArray(dtype=jaccard.dtype, size=batch_size)
        for batch in range(batch_size):
            jac = jaccard[batch]
            mask = masks[batch]

            mask_hot = tf.one_hot(mask, jac.shape[0])
            mask_hot = tf.transpose(mask_hot)
            mask_jac = mask_hot * jac
            output = tf.reduce_max(mask_jac,0)
            outputs = outputs.write(batch, output)
        outputs = outputs.stack()
        return outputs
