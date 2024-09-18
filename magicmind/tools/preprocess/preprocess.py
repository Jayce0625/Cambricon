# Copyright (C) [2020-2023] The Cambricon Authors. All Rights Reserved.

import os
import numpy as np
import sys
import argparse
from utils import image_preprocess, pre_process_yolo, Loader
from config import Config


def args():
    """Function for defining arguments."""
    parser = argparse.ArgumentParser(description='ImageNet Preprocess')

    parser.add_argument(
        '-f',
        '--framework',
        required=True,
        metavar='STR',
        help='framework to use (caffe/onnx/tensorflow/pytorch). ')
    parser.add_argument('-i',
                        '--image_path',
                        required=True,
                        metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-n',
                        '--max_image_number',
                        metavar='INT',
                        type=int,
                        default=100,
                        help='Max image number to preprocess (default: 100)')
    parser.add_argument('-l',
                        '--labels',
                        metavar='DIR',
                        help='Path to labels.txt')
    parser.add_argument('-s',
                        '--save_path',
                        required=True,
                        metavar='DIR',
                        help='path to save dataset')
    parser.add_argument('-m',
                        '--model_name',
                        required=True,
                        metavar='DIR',
                        help='network e.g. resnet50')
    return parser.parse_args()


def read_label(path, labels):
    """To read labels from file."""
    f = open(labels)
    line = f.readline()
    file_tail_list = ['.JPG', '.JPEG', '.PNG', '.BMP', '.TIF', '.GIF']
    name_list = []
    while line:
        is_tail_exist = False
        file_name = line[0:line.find(' ')]
        dot_pos = line.rfind('.')
        if dot_pos == -1:
            print("\nERROR! labels.txt contains illegal line:", line,
                  "which doesn't contain a filename suffix\n")
            sys.exit()
        else:
            for file_tail_i in file_tail_list:
                if line.upper().find(file_tail_i) != -1:
                    is_tail_exist = True
                    break
            if not is_tail_exist:
                print("\nWARNING: labels.txt contains illegal line.\n")
                print(line, " doesn't contain appropriate filename suffix.\n")
                print("Perhaps this will cause the failure of loading file.\n")
                sys.exit()
        if os.path.isfile(os.path.join(path, file_name)):
            name_list.append(os.path.join(path, file_name))
            line = f.readline()
        else:
            print("\nERROR! labels.txt contains illegal line:", line,
                  "which doesn't contain a existed file\n")
            sys.exit()
    return name_list


def preprocess_images(loader, model_conf, args):
    """To call preprocess."""
    file_list = []
    if not os.path.exists(args.save_path):
        print("Creates a folder: ", args.save_path)
        os.makedirs(args.save_path)
    for idx, image in enumerate(loader):
        if idx == args.max_image_number:
            break
        print("Preprocess image({}): {}".format(idx, loader[idx]))
        if args.model_name == "yolov3":
            image = pre_process_yolo(image, True, model_conf["resize_size"],
                                     model_conf["mean"], model_conf["std"])
        else:
            image = image_preprocess(
                image,
                resize_size=model_conf["resize_size"],
                center_crop_size=model_conf["center_crop_size"],
                mean=model_conf["mean"],
                std=model_conf["std"],
                need_normalize=model_conf["need_normalize"],
                need_bgr2rgb=model_conf["need_bgr2rgb"],
                need_transpose=model_conf["need_transpose"])
        image_save_path = os.path.join(args.save_path,
                                       os.path.basename(loader[idx]))
        shape = list(image.shape)
        shape.insert(0, 1)
        image.astype(np.float32).flatten().tofile(image_save_path)
        file_list.append(os.path.basename(loader[idx]) + " shape" + str(shape).replace(' ',''))
    with open(os.path.join(args.save_path, "file_list"), "w") as f:
        for image_path in file_list:
            f.write("{}\n".format(image_path))
    print("Calibration data:")
    print("   \\__ file_list: ",
          os.path.abspath(os.path.join(args.save_path, "file_list")))
    print("   \\__ calibration_data_path: ", os.path.abspath(args.save_path))


def main(args):
    """Main process."""
    config = Config()
    if args.labels:
        namelist = read_label(args.image_path, args.labels)
        loader = Loader(namelist)
    else:
        loader = Loader(args.image_path)
    if args.framework == "caffe":
        if args.model_name not in config.caffe:
            raise ValueError("model({}) not in configuration of caffe.".format(
                args.model_name))
        model_conf = config.caffe[args.model_name]
        preprocess_images(loader, model_conf, args)
    elif args.framework == "onnx":
        if args.model_name not in config.onnx:
            raise ValueError("model({}) not in configuration of onnx.".format(
                args.model_name))
        model_conf = config.onnx[args.model_name]
        preprocess_images(loader, model_conf, args)
    elif args.framework == "pytorch":
        if args.model_name not in config.pytorch:
            raise ValueError(
                "model({}) not in configuration of pytorch.".format(
                    args.model_name))
        model_conf = config.pytorch[args.model_name]
        preprocess_images(loader, model_conf, args)
    elif args.framework == "tensorflow":
        if args.model_name not in config.tensorflow:
            raise ValueError(
                "model({}) not in configuration of tensorflow.".format(
                    args.model_name))
        model_conf = config.tensorflow[args.model_name]
        preprocess_images(loader, model_conf, args)
    else:
        raise ValueError("framework name: {} not support!".format(
            args.framework))


if __name__ == "__main__":
    args = args()
    main(args)
