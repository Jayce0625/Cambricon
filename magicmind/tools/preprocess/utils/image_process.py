# Copyright (C) [2020-2023] The Cambricon Authors. All Rights Reserved.

from typing import Callable, List, Optional, Sequence, Set, Union, Tuple
import numpy as np
import numbers
import cv2


# image resize
def resize(image: np.ndarray,
           size: Union[Tuple[int, ...], List[int], int],
           interpolation=cv2.INTER_LINEAR):
    """ Resize an image.  """
    assert isinstance(image, np.ndarray)

    if isinstance(size, int):
        h, w = image.shape[:2]
        if (w <= h and w == size) or (h <= w and h == size):
            return image
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return cv2.resize(image, (ow, oh), interpolation=interpolation)
    elif isinstance(size, tuple) or isinstance(size, list):
        return cv2.resize(image, tuple(size[::-1]), interpolation=interpolation)
    else:
        raise TypeError(
            "size(type: {}) must be type of int or tuple/list.".format(
                type(size)))


# center crop
def center_crop(image: np.ndarray, size: Union[Tuple[int, ...], int, float]):
    """ Crop center of image with target size."""
    assert isinstance(image, np.ndarray)

    if isinstance(size, numbers.Number):
        size = (int(size), int(size))

    h, w = image.shape[:2]
    oh, ow = size
    x = int(round((w - ow) / 2.))
    y = int(round((h - oh) / 2.))
    return image[y:y + oh, x:x + ow]


def pre_process_yolo(img,
                     need_transpose=False,
                     resize=(416, 416),
                     mean=None,
                     std=None):
    """ Preprocess for yolo."""
    import math
    size = img.shape
    min_side = max(resize[0], resize[1])
    h, w = size[0], size[1]

    scale = float(min_side) / float(max(h, w))
    new_w, new_h = int(math.floor(float(w) * scale)), int(
        math.floor(float(h) * scale))
    resize_img = cv2.resize(img, (new_w, new_h), cv2.INTER_LINEAR)

    if new_w % 2 != 0 and new_h % 2 == 0:
        top, bottom = (min_side - new_h) / 2, (min_side - new_h) / 2
        left, right = (min_side - new_w) / 2 + 1, (min_side - new_w) / 2
    elif new_h % 2 != 0 and new_w % 2 == 0:
        top, bottom = (min_side - new_h) / 2 + 1, (min_side - new_h) / 2
        left, right = (min_side - new_w) / 2, (min_side - new_w) / 2
    elif new_h % 2 == 0 and new_w % 2 == 0:
        top, bottom = (min_side - new_h) / 2, (min_side - new_h) / 2
        left, right = (min_side - new_w) / 2, (min_side - new_w) / 2
    else:
        top, bottom = (min_side - new_h) / 2 + 1, (min_side - new_h) / 2
        left, right = (min_side - new_w) / 2 + 1, (min_side - new_w) / 2
    print("new_h, new_w : ", new_h, new_w, scale, top, bottom, left, right)
    pad_img = cv2.copyMakeBorder(resize_img,
                                 int(top),
                                 int(bottom),
                                 int(left),
                                 int(right),
                                 cv2.BORDER_CONSTANT,
                                 value=[128, 128, 128])
    rgb_img = cv2.cvtColor(pad_img, cv2.COLOR_BGR2RGB)
    float32_img = np.float32(rgb_img)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    norm_img = float32_img - mean
    norm_img = norm_img * std
    if need_transpose:
        norm_img = norm_img.transpose([2, 0, 1])
    return norm_img


# preprocess
def image_preprocess(image: np.ndarray,
                     resize_size: Union[Tuple[int, ...], int] = None,
                     center_crop_size: Union[Tuple[int, ...], int,
                                             float] = None,
                     need_transpose: bool = False,
                     need_normalize: bool = False,
                     need_bgr2rgb: bool = False,
                     mean: List[float] = None,
                     std: List[float] = None):
    """ image preprocess functon. """
    assert isinstance(image, np.ndarray)
    if need_bgr2rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if resize_size is not None:
        image = resize(image, resize_size[0])
    if center_crop_size is not None:
        image = center_crop(image, center_crop_size)
    image = image.astype(np.float32)
    if need_normalize:
        image = image / 255.
    if mean:
        image -= mean
    if std:
        image /= std
    if need_transpose:
        image = np.transpose(image, (2, 0, 1))
    return image
