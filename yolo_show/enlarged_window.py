# 图片放大的窗口类

import sys
import os
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from qfluentwidgets import *
from config import *
from QSS.qss import *


class EnlargedWindow(QWidget):

    def __init__(self):
        super(EnlargedWindow, self).__init__()

        self.big_label = QLabel()
        # self.pixmap = QPixmap()
        # self.big_label.setScaledContents(True)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.big_label)
        self.setLayout(self.layout)

        self.resize(1920, 1080)

        self.update()

    def setPath(self, img):
        self.big_label.setPixmap(QPixmap(img))

    def imgScaled(self):
        w, h = self.big_label.pixmap().width(), self.big_label.pixmap().height()
        w1, h1 = self.big_label.width(), self.big_label.height()
        ratio = max(w/w1, h/h1)
        self.big_label.pixmap().setDevicePixelRatio(ratio)
        self.big_label.setAlignment(Qt.AlignCenter)
        self.big_label.setPixmap(self.big_label.pixmap())

    def resizeEvent(self, event):
        self.imgScaled()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EnlargedWindow()
    window.setPath("F:/test_for_software/000000000009.jpg")
    window.show()

    app.exit(app.exec_())
