import argparse
import os.path
import sys

import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from config import *
from util.utils import *
from util.details import *

import cv2
from my_treeView import *
from my_dataWidget import *
from show_widget import *
from models.common import *
from models.experimental import *
from models.yolo import *
from util.general import *
from util.torch_utils import *
from util.datasets import *


class SubVideoWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.outer_layout = QVBoxLayout()

        self.vid_label = QLabel("")

        self.bot_label = QLabel("")

        # 设置控件属性 和 布局控件
        self.resize(800, 600)

        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Window)

        self.setParam()
        self.initUI()

    def initUI(self):
        self.outer_layout.addWidget(self.vid_label)
        self.outer_layout.addWidget(self.bot_label)
        self.setLayout(self.outer_layout)

    def setParam(self):
        self.outer_layout.setContentsMargins(0, 0, 0, 0)
        self.vid_label.setMinimumHeight(400)
        self.vid_label.setMinimumWidth(400)
        self.vid_label.setStyleSheet("background-color: black;")

        self.bot_label.setMaximumHeight(200)

    def setPicture(self, pix):
        self.vid_label.setScaledContents(True)
        self.vid_label.setPixmap(pix)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    sub = SubVideoWindow()

    im0 = load_image1("./icons/bg.png")
    im1 = im0.tolist()
    im2 = np.array(im1)
    print(np.array_equal(im0, im2))
    # 将数据类型设置为 uint8
    im2 = im2.astype(np.uint8)
    sub.setPicture(numpy_to_pixmap(im2))
    sub.show()
    sys.exit(app.exec_())
