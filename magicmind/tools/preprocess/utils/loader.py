# Copyright (C) [2020-2023] The Cambricon Authors. All Rights Reserved.

import os
import cv2


class Loader(object):
    """ Simple data loader for cv images. """

    def __init__(self, file_name):
        """ init function with file path """
        if isinstance(file_name, list):
            self.init(file_name)
        else:
            self.__path = file_name
            files = os.listdir(self.__path)
            self.__filenames = []
            for file in files:
                if os.path.isfile(os.path.join(self.__path,
                                               file)) and (".JPEG" in file or
                                                           ".jpg" in file):
                    self.__filenames.append(os.path.join(self.__path, file))
            self.__idx = 0
            if len(self) == 0:
                raise ValueError("There is not any image files in {}".format(
                    self.__path))

    def init(self, image_list):
        """ init function with file list """
        self.__filenames = image_list
        self.__idx = 0
        if len(self) == 0:
            raise ValueError("There is not any image files in list")

    def __len__(self):
        """ return num of file """
        return len(self.__filenames)

    def __getitem__(self, idx):
        """ get idx file name """
        assert 0 <= idx < len(self)
        return self.__filenames[idx]

    def __iter__(self):
        """ just ret self """
        return self

    def __next__(self):
        """ return next file """
        if self.__idx == len(self):
            raise StopIteration
        self.__idx = self.__idx + 1
        return cv2.imread(self[self.__idx - 1])
