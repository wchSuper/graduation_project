import sys
import os
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from qfluentwidgets import *
from config import *
from QSS.qss import *


class MyTreeView(TreeView):

    def __init__(self):
        super(TreeView, self).__init__()

        self.setMouseTracking(True)

        # 获取系统所有文件
        self.model01 = QFileSystemModel()
        # 进行筛选只显示文件夹，不显示文件和特色文件
        # self.model01.setFilter(QDir.Dirs | QDir.NoDotAndDotDot)
        self.model01.setRootPath('')

        self.setModel(self.model01)
        # 隐藏文件系统中1-3列的信息 只展示文件名
        for col in range(1, 4):
            self.setColumnHidden(col, True)
        # 双击事件
        self.doubleClicked.connect(self.onLeftTreeDoubleClicked)

        # 设置样式
        # self.setStyleSheet(tree_scroll)

    def onLeftTreeDoubleClicked(self, index):
        filePath = self.model01.filePath(index)
        if os.path.isdir(filePath):
            self.model01.setRootPath(filePath)
        else:
            suffix = filePath.split('.')[-1]
            # 属于图片
            if suffix in picture_suffix_list:
                print(filePath)
            # 属于视频
            if suffix in video_suffix_list:
                print("double clicked: ", filePath)

    def mouseMoveEvent(self, event):
        index = self.indexAt(event.pos())
        if index.isValid():
            file_path = self.model().filePath(index)
            tooltip = f"{file_path}"
            QToolTip.showText(self.mapToGlobal(event.pos()), tooltip, self)

        super().mouseMoveEvent(event)