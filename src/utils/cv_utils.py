
import cv2 as cv
import tensorflow as tf
import numpy as np

def draw_detection(im, labels, bboxes, color = (0,255,0), score_threshold=0.5):
    image = np.asarray(im)
    image = np.copy(image)
    image_shape = image.shape
    height, width = image_shape[0], image_shape[1]

    for i in range(len(labels)):
        label = labels[i]
        bbox = bboxes[i]

        if(label > score_threshold):
            image = draw_bbox(image, bbox, width, height, color)

    return image


def draw_rectangle(image, rect, color):
    ymin = rect[0]
    xmin = rect[1]
    ymax = rect[2]
    xmax = rect[3]
    image_rect = cv.rectangle(image,(xmin,ymin),(xmax,ymax),color)
    return image_rect

def draw_bbox(image, bbox, width, height, color):

    ymin = int(bbox[0] * height)
    xmin = int(bbox[1] * width)
    ymax = int(bbox[2] * height)
    xmax = int(bbox[3] * width)

    rect = [ymin, xmin, ymax, xmax]
    image_rect = draw_rectangle(image ,rect, color)
    return image_rect

def imshow(image, name, delay = 0):

    if(tf.is_tensor(image)):
        image = image.numpy()

    cv.imshow(name, image)
    cv.waitKey(delay)

def save_image(image, name):
    if (tf.is_tensor(image)):
        image = image.numpy()

    cv.imwrite(name, image)
