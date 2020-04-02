# -*- coding: utf-8 -*-

# @Time   : 2020/3/3:11:36
# @Author : xuqiang

import tensorflow as tf
import os
import glob
from src.utils.logger import Logger
logger = Logger.getLogger()

class DataLoader():
    def __init__(self):
        self.key2features = self.key_to_features()
        self.decoder_map = self.decoders()

    def key_to_features(self):
        '''获取key to features'''
        raise NotImplementedError("Not implement key_to_features")

    def decoders(self):
        '''生成一个map，name-decoder'''
        raise NotImplementedError("Not implement decoders")

    def parse(self, record):
        # 获取
        parsed = tf.io.parse_single_example(record, self.key2features)

        outputs = dict()
        for name in self.decoder_map.keys():
            decoder = self.decoder_map[name]
            outputs[name] = decoder.tensors_to_item(parsed)

        return outputs


    def load(self, config):
        dataset = self.read(config)
        dataset = dataset.map(self.parse)

        return dataset

    def read(self, config):
        paths = config['input_path']
        input_files = list()
        for path in paths:

            if not os.path.isfile(path):
                files = glob.glob(path)
                [input_files.append(file) for file in files]
            else:
                input_files.append(path)

        num_readers = config.get('num_readers', len(input_files))
        num_readers = max(1, num_readers)

        if (num_readers > len(input_files)):
            num_readers = len(input_files)
            logger.warn("num_readers has been reduced to "
                        "{} to match input file".format(num_readers))
        filename_dataset = tf.data.Dataset.from_tensor_slices(input_files)

        num_epochs = config.get('num_epochs', 0)
        if num_epochs > 0:
            filename_dataset = filename_dataset.repeat(config['num_epochs'])

        buffer_size = 8 * 1000 * 1000
        record_dataset = tf.data.TFRecordDataset(filename_dataset,
                                                 buffer_size=buffer_size,
                                                 num_parallel_reads=num_readers)
        shuffle = config.get("shuffle", False)
        if shuffle:
            shuffle_buffer_size = config.get('shuffle_buffer_size', 2048)
            record_dataset = record_dataset.shuffle(shuffle_buffer_size)

        record_dataset = record_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return record_dataset
