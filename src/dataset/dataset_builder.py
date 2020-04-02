# -*- coding: utf-8 -*-

# @Time   : 2019/9/17:14:59
# @Author : xuchen.xq

from src.dataset.coco_loader import CocoLoader
from src.dataset.reid_loader import ReIDLoader
from src.dataset.data_transform import DataTransform

def build_dataset(input_reader_config,
                  model_prerocess_fn,
                  use_reid = False):

    if use_reid:
        data_loader = ReIDLoader()
    else:
        data_loader = CocoLoader()

    dataset = data_loader.load(input_reader_config)

    if 'transform' in input_reader_config:
        transform_config = input_reader_config['transform']
        data_transform = DataTransform(model_prerocess_fn,
                                       transform_config)

        dataset = dataset.map(data_transform.transform)
        # for data in dataset:
        #     data_transform.transform(data)

    if 'sample_1_of_n_examples' in input_reader_config:
        dataset = dataset.shard(input_reader_config['sample_1_of_n_examples'], 0)

    num_prefetch_batches = input_reader_config.get('num_prefetch_batches', 2)
    dataset = dataset.prefetch(num_prefetch_batches)

    batch_size = input_reader_config.get('batch_size', 4)
    dataset = dataset.batch(batch_size)

    return dataset
