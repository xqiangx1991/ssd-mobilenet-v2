# -*- coding: utf-8 -*-

# @Time   : 2019/9/19:14:28
# @Author : xuchen.xq

import argparse

import tensorflow as tf
import yaml

from src.utils.logger import Logger

from src.train.model_builder import build_model
from src.train.optimizer_builder import build_optimizer
from src.dataset.dataset_builder import build_dataset
from src.train.train_step import train_step
from src.train.eval_step import coco_eval_step
from src.train.eval_step import coco_summary_images
from src.train.eval_step import reid_eval_step

parser = argparse.ArgumentParser(description="ssd-mobilenet-v2")
parser.add_argument("--output_path",help="Path to output model directory"
                    "where event and checkpoint files will be written")

parser.add_argument("--pipeline_config_path", help="Path to pipeline config file")
args = parser.parse_args()

def run():
    train_things = prepare()
    detection_model = train_things['model']
    optimizer = train_things['optimizer']
    config = train_things['config']
    train_dataset = train_things['train_dataset']
    eval_dataset = train_things['eval_dataset']

    # 增加reid的部分
    use_reid = config["model"].get("reid", False)
    pid_name_map_path = ''
    gallery_map_path = ''
    if 'eval_config' in config:
        pid_name_map_path = config["eval_config"].get("pid_name_map_path", "")
        gallery_map_path = config["eval_config"].get("gallery_map_path", "")


    model_path = args.output_path

    logger = Logger.getLogger(model_path)

    summary_writer = tf.summary.create_file_writer(model_path)
    num_step = config['train_config']['num_step']

    step = tf.Variable(0, dtype=tf.int64)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     model = detection_model,
                                     step = step)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, model_path, 3)

    if "fine_tune_checkpoint" in config["train_config"]:
        fine_tune_checkpoint = config["train_config"]["fine_tune_checkpoint"]
        checkpoint.restore(fine_tune_checkpoint)
        logger.info("load weight from {}".format(fine_tune_checkpoint))

    # tf.summary.trace_on(graph=True, profiler=True)
    tf.summary.experimental.set_step(step)
    with summary_writer.as_default():
        for epoch in range(num_step):
            logger.info("=============Epoch:{}==============".format(epoch))
            tf.keras.backend.set_learning_phase(True)
            train_step(detection_model, optimizer, train_dataset)

            # 生成graph
            # tf.summary.trace_export(name="ssd-mobilenet-v2",
            #                         step = step,
            #                         profiler_outdir=model_path)
            # tf.summary.flush()

            logger.info("=============Test==================")
            tf.keras.backend.set_learning_phase(False)
            coco_eval_step(detection_model, eval_dataset)
            coco_summary_images(detection_model, eval_dataset, step)

            reid_eval_step(detection_model,
                           eval_dataset,
                           pid_name_map_path,
                           gallery_map_path,
                           use_reid)

            saved_model = checkpoint_manager.save()
            logger.info("model saved to {}".format(saved_model))


# build
def prepare():
    pipeline_config_path = args.pipeline_config_path
    with open(pipeline_config_path, 'r') as pipe:
        config = yaml.load(pipe,Loader=yaml.Loader)
    # 生成模型、优化器和数据输入
    model_config = config['model']
    model = build_model(model_config)

    optimizer_config = config['optimizer']
    optimizer = build_optimizer(optimizer_config)

    use_reid = model_config.get('reid', False)
    train_input_config = config['train_input']
    train_dataset = build_dataset(train_input_config,
                                  model.preprocess,
                                  use_reid=use_reid)

    eval_input_config = config['eval_input']
    eval_dataset = build_dataset(eval_input_config,
                                 model.preprocess,
                                 use_reid=use_reid)

    output = {
        "model":model,
        "optimizer":optimizer,
        "train_dataset":train_dataset,
        "eval_dataset":eval_dataset,
        'config':config
    }

    return output

def reconstruct_input_config(model_config,
                             input_config):

    # 补充input config
    width = model_config.get("width", 300)
    height = model_config.get("height", 300)
    num_class = model_config.get("num_class", 1)

    if "transform" in input_config:
        transform_config = input_config['transform']
        transform_hw_steps = ["resize", "padding"]

        for step in transform_hw_steps:
            if step in transform_config:
                input_config['transform'][step]['width'] = width
                input_config['transform'][step]['height'] = height

        transform_nc_step = ['padding']
        for step in transform_nc_step:
            if step in transform_config:
                input_config["transform"][step]['num_class'] = num_class

    return input_config




if __name__ == "__main__":
    run()