import tensorflow as tf
from src.utils.standard_fields import TFRecordKeys

class Tensor():
    def __init__(self,
                 tensor_key,
                 dtype,
                 default_value=0):
        if dtype is None:
            raise ValueError("dtype can't be None")
        self._tensor_key = tensor_key
        self._dtype = dtype
        self._default_value = default_value

    def tensors_to_item(self,keys_to_tensors):
        tensor = keys_to_tensors[self._tensor_key]

        if isinstance(tensor,tf.sparse.SparseTensor):
            # tensor = tf.sparse.to_dense(tensor, default_value=self._default_value)
            tensor = tensor.values
            tensor = tf.cast(tensor,self._dtype)
            tensor = tf.reshape(tensor,(-1,1))

        return tensor


class Image():
    def __init__(self,
                 image_key,
                 format_key,
                 channels=3,
                 dtype=tf.uint8):
        if not image_key:
            image_key = TFRecordKeys.image_encode

        if not format_key:
            format_key = TFRecordKeys.image_format

        self._image_key = image_key
        self._format_key = format_key
        self._channels = channels
        self._dtype = dtype

    def tensors_to_item(self, keys_to_tensors):
        image_buffer = keys_to_tensors[self._image_key]
        image_format = keys_to_tensors[self._format_key]

        return self._decode(image_buffer, image_format)

    def _decode(self, image_buffer, image_format):
        image = tf.cond(tf.image.is_jpeg(image_buffer),
                lambda: tf.image.decode_jpeg(image_buffer, self._channels),
                lambda: tf.image.decode_png(image_buffer, self._channels))

        return image



class BoundingBox():
    def __init__(self,
                 keys):

        if len(keys) != 4:
            raise ValueError("Bounding box expects 4 keys but got {}".format(len(keys)))
        self._keys = keys

    def tensors_to_item(self, keys_to_tensors):
        sides = []
        for key in self._keys:
            side = keys_to_tensors[key]

            if isinstance(side, tf.sparse.SparseTensor):
                side = side.values
            side = tf.expand_dims(side,0)
            sides.append(side)
        sides = tf.concat(sides,0)
        sides = tf.transpose(sides)
        return sides
