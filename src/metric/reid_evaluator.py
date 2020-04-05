# -*- coding: utf-8 -*-

# @Time   : 2020/3/17:19:58
# @Author : xuqiang

import tensorflow as tf
from src.utils.standard_fields import DetectionKeys
from collections import defaultdict
import random
from src.utils.logger import Logger
from src.utils.cv_utils import draw_detection

logging = Logger.getLogger()

def construct_key(image_name, pid_name):
    return "{}-{}".format(image_name, pid_name)

def deconstruct_key(key):
    k = key.split("-")
    image_name = k[0]
    pid_name = k[1]
    return image_name, pid_name

def read_pid_name_map(pid_name_map_path):
    pid_map = dict()
    with open(pid_name_map_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split(" ")
            pid, pid_name = int(line[0]), line[1].strip()
            pid_map[pid] = pid_name
    return pid_map

def read_gallery_map(gallery_map_path):
    gallery_map = defaultdict(list)
    with open(gallery_map_path, "r") as gallery_file:
        lines = gallery_file.readlines()
        for line in lines:
            line = line.strip().split(" ")
            query_reid = "{}-{}".format(line[0][:-4], line[1])
            gallery_reid = "{}-{}".format(line[2][:-4], line[3])
            gallery_map[query_reid].append(gallery_reid)
    return gallery_map


class ReIDEvaluator():
    '''先要把所有的特征都采集到，然后再根据query和gallery进行搜索，
    找到最相似的，并且相似度需要满足一定的条件才可以。然后再看看准确率有多少
    '''

    def __init__(self,
                 pid_name_map_path,
                 gallery_map_path,
                 score_threshold = 0.7):
        self.score_threshold = score_threshold
        self.image_names = list()
        self.labels = list()
        self.pids = list()
        self.features = list()
        self.pid_feature_map = dict()
        # 不同的key，但是同一个pid
        self.pidname_keys_map = defaultdict(list)

        self.pid_name_map = read_pid_name_map(pid_name_map_path)
        self.gallery_map = read_gallery_map(gallery_map_path)

        # 从所有的候选集中选取10个query
        self.im_keys = list()
        self.im_query = list()
        self.im_images = dict()
        self.im_bboxes = dict()
        self.im_query_map = dict()

        query_keys = list(self.gallery_map.keys())[:10]
        for query_key in query_keys:
            keys = [query_key] + self.gallery_map[query_key]
            [self.im_keys.append(k) for k in keys]
            self.im_query.append(query_key)


    def accumulate(self, predict_dict):
        # 输入预测值，构建查询table
        image_names = predict_dict[DetectionKeys.image_name]
        images = predict_dict[DetectionKeys.image]
        predict_labels = predict_dict[DetectionKeys.detection_classes]
        predict_bboxes = predict_dict[DetectionKeys.detection_bboxes]
        predict_features = predict_dict[DetectionKeys.reid_feature]
        predict_pids = predict_dict[DetectionKeys.pids]

        image_names = image_names.numpy()
        predict_pids = predict_pids.numpy()

        for idx, predict_pid in enumerate(predict_pids):
            image_name = image_names[idx]
            image_name = str(image_name)[2:-5]
            predict_pid = int(predict_pid)
            predict_pid_name = "p{}".format(predict_pid)
            if predict_pid in self.pid_name_map:
                predict_pid_name = self.pid_name_map[predict_pid]

            key = construct_key(image_name, predict_pid_name)

            if key in self.im_keys:
                self.im_images[key] = images[idx]
                self.im_bboxes[key] = predict_bboxes[idx]

            self.pid_feature_map[key] = predict_features[idx]
            self.pidname_keys_map[predict_pid_name].append(key)


    def evaluate_images(self):
        output_images = list()
        for query_key in self.im_query_map.keys():
            gallery_key = self.im_query_map[query_key]['gallery_key']
            query_gt_key = self.im_query_map[query_key]['gt_key']

            keys = [query_key, gallery_key, query_gt_key]
            colors = [(255,0,0),(0,255,0),(0,0,255)]
            images = list()
            bboxes = list()

            for key in keys:
                if key in self.im_images:
                    images.append(self.im_images[key])
                    bboxes.append(self.im_bboxes[key])

            serial_images = list()
            for idx, image in enumerate(images):
                image = (image + 1.0)/2.0
                # bboxes = [size, 4]
                # labels = [size, ]
                bbox = bboxes[idx]
                bbox = tf.constant(bbox, shape=(1,4))
                label = tf.ones((1,1))

                image = draw_detection(image, label, bbox, color=colors[idx])
                serial_images.append(image)

            if len(serial_images) > 0:
                serial_images = tf.concat(serial_images, axis=1)
                output_images.append(serial_images)

        return output_images



    def evaluate(self, K = 50):
        # 所有已经提取了特征的
        logging.info("featured {} pid".format(len(self.pid_feature_map.keys())))
        total_keys = list(self.pid_feature_map.keys())
        K = min(K, max(0, len(total_keys) - 2))

        total_valid_query_count = 0
        total_matched_count = 0
        for query_key in self.gallery_map.keys():
            # 如果querykey没有提取特征，略过
            query_gt_key = query_key
            if query_key not in self.pid_feature_map:
                continue
            _, query_pid_name = deconstruct_key(query_key)
            # 从池中随机选择出候选项
            random.shuffle(total_keys)
            gallery_pool_keys = total_keys[:K]
            # 把候选集加入
            candicate_keys = self.pidname_keys_map[query_pid_name]
            # 如果只有query，那就略过
            if len(candicate_keys) <=1:
                logging.warn("pid {} does not have candidate id".format(query_pid_name))
                continue

            # 一方面要加入一个候选项进去，同时也要保证其他的候选项不在里面
            for ck in candicate_keys:
                if ck in gallery_pool_keys:
                    gallery_pool_keys.remove(ck)

            for ck in candicate_keys:
                if ck != query_key:
                    gallery_pool_keys.append(ck)
                    query_gt_key = ck
                    break
            # 把候选集对应的特征都捞出来进行对比
            valid_gallery_pool_features = list()
            valid_gallery_pool_pid_names = list()
            valid_gallery_pool_keys = list()
            for gallery_pool_key in gallery_pool_keys:
                # 如果没有提取特征
                if gallery_pool_key not in self.pid_feature_map:
                    continue

                valid_gallery_pool_keys.append(gallery_pool_key)
                gallery_pool_feature = self.pid_feature_map[gallery_pool_key]
                valid_gallery_pool_features.append(gallery_pool_feature)

                _, gallery_pid_name = deconstruct_key(gallery_pool_key)
                valid_gallery_pool_pid_names.append(gallery_pid_name)

            if query_pid_name not in valid_gallery_pool_pid_names:
                # 里面并没有有效的查询
                logging.warn("----> not found pid {} in gallery <-----".format(query_pid_name))
                continue

            # 寻找对应的pid
            valid_gallery_pool_features = tf.stack(valid_gallery_pool_features, axis=0)
            query_feature = self.pid_feature_map[query_key]
            scores = tf.reduce_sum(query_feature * valid_gallery_pool_features,axis=1)
            max_score = tf.reduce_max(scores)
            max_score_index = tf.argmax(scores)

            matched_gallery_pool_key = valid_gallery_pool_keys[max_score_index]
            matched_gallery_pool_pid_name = valid_gallery_pool_pid_names[max_score_index]
            logging.info("[query] {} [found] {} [score] {} from {} items".format(
                query_key, matched_gallery_pool_key, max_score, len(valid_gallery_pool_keys)))

            if (matched_gallery_pool_pid_name == query_pid_name) and \
                (max_score > self.score_threshold):
                total_matched_count += 1

            total_valid_query_count += 1

            # 处理图片数据
            if query_key in self.im_query:
                self.im_query_map[query_key] = {"gallery_key":matched_gallery_pool_key,
                                                "gt_key":query_gt_key}


        total_valid_query_count = max(total_valid_query_count, 1)
        accuracy = total_matched_count / total_valid_query_count

        return total_matched_count, total_valid_query_count, accuracy