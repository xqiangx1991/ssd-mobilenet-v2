# -*- coding: utf-8 -*-

# @Time   : 2019/9/17:15:49
# @Author : xuqiang

import logging
import os

class Logger():

    __instance = None

    @staticmethod
    def getLogger(path=None):

        if path is not None:

            if os.path.isdir(path):
                parent_path = os.path.dirname(path)
                if not os.path.exists(parent_path):
                    os.makedirs(path)

                Logger(path)

        if Logger.__instance is None:
            Logger(path)

        return Logger.__instance._logger


    def __init__(self, path=None):
        super(Logger, self).__init__()

        FORMAT = '%(asctime)s - %(name)s - %(levelname)s: - %(message)s'

        formatter = logging.Formatter(FORMAT,datefmt='%Y-%m-%d %H:%M:%S')

        self._logger = logging.getLogger()
        self._logger.setLevel(logging.INFO)

        if path:
            path = os.path.join(path, "logger.txt")
            file_handler = logging.FileHandler(path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

        Logger.__instance = self