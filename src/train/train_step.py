# -*- coding: utf-8 -*-

# @Time   : 2020/3/5:19:19
# @Author : xuqiang

import tensorflow as tf
from src.utils.standard_fields import DetectionKeys
from src.utils.standard_fields import InputDataKeys
from src.utils.logger import Logger


logger = Logger.getLogger()

def train_step(model, optimizer, train_dataset):

    for inputs in train_dataset:
        one_step(model, optimizer,inputs)

def one_step(model, optimizer, inputs):
    image = inputs[InputDataKeys.image]
    with tf.GradientTape() as tape:
        predict_dict = model(image, training=True)

        output_loss = model.calculate_loss(inputs, predict_dict)

        if (len(model.losses) > 0):
            regularization_losses = model.losses
            regularization_loss = tf.math.add_n(regularization_losses)
            output_loss["regularization_loss"] = regularization_loss
        total_loss = tf.math.add_n([loss for loss in output_loss.values()])
        output_loss["total_loss"] = total_loss

    # with tf.device("/cpu:0"):
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    model.update_oim()

    step = tf.summary.experimental.get_step()
    if step is not None and (step % 100 == 0):
        for loss_name in output_loss.keys():
            loss_value = output_loss[loss_name]
            tf.summary.scalar("train/{}".format(loss_name), loss_value, step=step)

    step.assign_add(1)
    return output_loss