import glob
import re
import datetime
import logging
import math
import platform
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import torch
import os

from PyQt5 import QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from config import *
import cv2
import numpy as np
from util.details import *
from pathlib import Path


# 工具函数

# 把图片自适应尺寸到label上 缺失的部分补充黑边


# 获取当前路径文件的上一个或者下一个文件   -- #
# 判断当前文件是否是图片或者视频        -- #
# id_ 0 --> last    id_ 1 --> next   -- #
# 返回上一个或下一个文件                -- #
def getLastOrNextFile(id_, current_path):
    # 判断current_path是否合法
    if not "/" in current_path and not "\\" in current_path:
        return "**error**"

    # 先判断是否是目录
    if os.path.isdir(current_path):
        return "**dir**"
    # 获取文件的目录地址
    parent_path = Path(current_path).resolve().parent
    file_list = os.listdir(parent_path)
    file_list_len = len(file_list)
    index = file_list.index(current_path.split('/')[-1])
    if id_:
        index += 1
        if index >= file_list_len:
            index = file_list_len - 1
        while index < file_list_len and not file_list[index].split('.')[-1] in picture_suffix_list:
            index += 1
        if index == file_list_len:
            # 下一张没有图片
            return "**last-none**"
    else:
        index -= 1
        if index <= 0:
            index = 0
        while index >= 0 and not file_list[index].split('.')[-1] in picture_suffix_list:
            index -= 1
        if index < 0:
            # 上一张没有图片
            return "**next-none**"

    return parent_path / file_list[index]


# --------------- 根据秒数获取 当前h:m:s --------------- #
def getTimeLabelInfo(secs):
    s = secs % 60
    secs //= 60
    m = secs % 60
    secs //= 60
    h = secs
    h_str = f"{h}"
    m_str = f"{m}" if m // 10 > 0 else f"0{m}"
    s_str = f"{s}" if s // 10 > 0 else f"0{s}"

    return h_str + ":" + m_str + ":" + s_str


# ----------------- 获取最新exp ----------------- #
def get_latest_exp():
    directory = str(Path(os.getcwd())) + "\\runs\\detect"
    name_list = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            name_list.append(item)

    print(name_list)
    max_int = 0
    for name in name_list:
        tmp_int = int(name.split('p')[-1]) if name.split('p')[-1] != "" else 0
        max_int = tmp_int if max_int <= tmp_int else max_int

    second_str = str(max_int) if max_int else ""
    result_dir = os.path.join(directory, "exp" + second_str)
    return result_dir


# ------------------ 把numpy的图像数组 转换成 QPixmap形式 ------------------------------- #
def load_image1(file_path):
    # 使用OpenCV加载图像
    img = cv2.imread(file_path)

    # 将图像从BGR格式转换为RGB格式
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def numpy_to_pixmap(numpy_img):
    """Converts a NumPy array (image) to QPixmap."""
    # height, width, channel = numpy_img.shape
    # bytes_per_line = 3 * width
    # q_image = QtGui.QImage(numpy_img.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
    # pixmap = QtGui.QPixmap.fromImage(q_image)
    # return pixmap
    img = numpy_img  # numpy类型图片
    img = img.astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = QtGui.QImage(img[:], img.shape[1], img.shape[0], img.shape[1] * 3, QtGui.QImage.Format_RGB888)
    pixmap = QtGui.QPixmap(img)
    return pixmap


# -------------------- 把detect后图片的信息以txt格式存入exp文件夹下 ------------------- #
# -------------------- [{"xyxy": xyxy, "cls": names[int(cls)], "conf": conf}, .... ]---- #
def store_txt(detect_list, which_frame):
    exp_dir = get_latest_exp()
    new_dir = str(exp_dir) + "\\" + "TXT"
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    file_name = "info"
    # file_name_id = 0
    file_path = os.path.join(new_dir, file_name + str(which_frame) + ".txt")
    # while os.path.exists(file_path):
    #     file_name_id += 1
    #     file_path = os.path.join(new_dir, file_name + str(file_name_id) + ".txt")

    for _dict in detect_list:
        xyxy_normal_list = [tensor.item() for tensor in _dict["xyxy"]]
        conf_normal = _dict["conf"].item()
        with open(file_path, 'a') as file:
            file.write(str(xyxy_normal_list) + "\n" + str(_dict["cls"]) + "\n" + str(conf_normal) + "\n")


# ---------------- ---------------------------- #
def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


# ----------------- 设置GPU ---------------------- #
def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    # 打印一些版本信息
    s = f'YOLOR 🚀 {git_describe() or date_modified()} torch {torch.__version__} '
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    logger.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    return torch.device('cuda:0' if cuda else 'cpu')


if __name__ == "__main__":
    result_dir = get_latest_exp()
    print(result_dir)