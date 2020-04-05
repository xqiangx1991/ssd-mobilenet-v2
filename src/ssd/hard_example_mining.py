
import tensorflow as tf
from src.layer.common import non_max_suppression_with_scores

class HardExampleMining():

    def __init__(self,
                 num_hard_examples = 3000,
                 max_negatives_per_positive = 3,
                 iou_threshold = 0.99,
                 minimum_negatives = 3.0):
        super(HardExampleMining, self).__init__()
        self._num_hard_examples = num_hard_examples
        self._max_negatives_per_positive = max_negatives_per_positive
        self._iou_threshold = iou_threshold
        self._minimum_negatives = minimum_negatives

    def mining(self,
               target_labels,
               predict_bboxes_decode,
               labels_loss):

        batch_size = tf.shape(target_labels)[0]
        # 按照batch size的纬度进行处理
        masks = tf.TensorArray(tf.float32, size = batch_size)
        for batch in tf.range(batch_size):
            batch_class_loss = labels_loss[batch]
            batch_encode_bboxes = predict_bboxes_decode[batch]
            batch_encode_labels = target_labels[batch]

            # 用分类的loss排序，用处有多大？毕竟这个值很小
            selected_indices,_ = non_max_suppression_with_scores(batch_encode_bboxes,
                                                   batch_class_loss,
                                                   self._num_hard_examples,
                                                   self._iou_threshold)

            selected_indices = tf.expand_dims(selected_indices, -1)
            mask = self._subsample_positive_and_negative(selected_indices,
                                                   batch_encode_labels)
            masks = masks.write(batch, mask)

        masks = masks.stack()

        return masks

    def _subsample_positive_and_negative(self,
                                         selected_indices,
                                         encode_labels):

        # 1. 选择的样本中有多少是正样本
        positive_indicator = tf.gather(tf.greater(encode_labels, 0), selected_indices)
        positive_indicator = tf.reshape(positive_indicator, [-1])
        num_positive = tf.reduce_sum(tf.cast(positive_indicator, tf.float32))

        # 2. 选择负样本
        num_negative = tf.maximum(self._max_negatives_per_positive * num_positive,
                           self._minimum_negatives)
        negative_indicator = tf.gather(tf.less_equal(encode_labels, 0), selected_indices)
        negative_indicator = tf.reshape(negative_indicator, [-1, 1])
        negative_indicator = tf.less_equal(
            tf.cumsum(tf.cast(negative_indicator, tf.float32)), num_negative)
        negative_indicator = tf.reshape(negative_indicator, [-1])
        subsample_indicator = tf.where(tf.logical_or(positive_indicator, negative_indicator))
        subsample_indicator = tf.cast(subsample_indicator, tf.int32)
        subsample_indicator = tf.reshape(subsample_indicator, [-1])
        subsample_indicator = tf.gather(selected_indices, subsample_indicator)

        mask = tf.scatter_nd(subsample_indicator,
                             tf.ones_like(subsample_indicator, tf.float32),
                             tf.shape(encode_labels))
        mask = tf.squeeze(mask)
        return mask
