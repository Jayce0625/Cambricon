# Copyright (C) [2020-2023] The Cambricon Authors. All Rights Reserved.

import os
from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

__all__ = ["Config"]


def config_decorator(func):
    """ decorator for config """
    def wrapper(self, *args, **kwargs):
        """ wrap loader """
        config_file = func(self, *args, **kwargs)
        with open(config_file, "r") as f:
            return load(f.read(), Loader=Loader)

    return wrapper


class Config(object):
    """ Config class for different framework. """

    def __init__(self):
        self.__curdir = os.path.dirname(os.path.abspath(__file__))

    @property
    @config_decorator
    def caffe(self):
        return self.__curdir + "/caffe.yaml"

    @property
    @config_decorator
    def onnx(self):
        return self.__curdir + "/onnx.yaml"

    @property
    @config_decorator
    def pytorch(self):
        return self.__curdir + "/pytorch.yaml"

    @property
    @config_decorator
    def tensorflow(self):
        return self.__curdir + "/tensorflow.yaml"
