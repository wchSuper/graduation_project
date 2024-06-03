import sys
import os
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from qfluentwidgets import *
from qfluentwidgets.multimedia import StandardMediaPlayBar
from config import *
from enlarged_window import *
from QSS.qss import *


class ShowWidget(QWidget):
    # ------- 暂停和播放的信号 默认暂停 ----- #
    vid_stopped_signal, vid_started_signal = pyqtSignal(), pyqtSignal()

    # 开始实时检测信号
    vid_shishi_detect = pyqtSignal()

    def __init__(self):
        super(ShowWidget, self).__init__()

        # 设置背景
        self.layout0 = QVBoxLayout()

        # ------- top层 3label -------- #
        self.top_widget = QWidget()
        self.top_label1 = BodyLabel()
        self.top_label2 = BodyLabel()
        self.top_label3 = BodyLabel()
        self.top_layout = QHBoxLayout()

        # ------- show层 1label--------- #
        self.show_label = QLabel()

        # ------- 底部播放栏层 ------------------- #
        self.bottom_widget = QWidget()
        self.bottom_layout = QVBoxLayout()
        self.bar = StandardMediaPlayBar()
        self.video_play_btn = self.bar.playButton
        self.volume_btn = self.bar.volumeButton
        self.back_btn = self.bar.skipBackButton
        self.forward_btn = self.bar.skipForwardButton
        self.video_proc_slider = self.bar.progressSlider
        self.show_time_label_left = self.bar.currentTimeLabel
        self.show_time_label_right = self.bar.remainTimeLabel

        # 浮动标签
        self.float_label = QLabel(self.show_label)

        # 菜单
        self.menu = RoundMenu(self)
        self.action1 = Action(FluentIcon.ZOOM_IN, "大屏")
        self.action2 = Action(FluentIcon.COPY, "复制")
        self.action3 = Action(FluentIcon.SETTING, "设置")
        self.action4 = Action(FluentIcon.CAMERA, "实时检测")
        self.menu.addActions([
            self.action1,
            self.action2,
            self.action3,
            self.action4
        ])

        # enlarged window
        self.enlarged_window = EnlargedWindow()

        # ------- 方法 -------------------- #
        # ------- 0未播放  1播放 ---------- #
        self.current_play_id = 0
        self.setPara()
        self.initUI()
        self.signalSlots()

    def initUI(self):
        # -------- top layout -------- #
        self.top_layout.addWidget(self.top_label1)
        self.top_layout.addWidget(self.top_label2)
        self.top_layout.addWidget(self.top_label3)
        self.top_widget.setLayout(self.top_layout)

        # -------- bottom layout ----- #
        self.bottom_layout.addWidget(self.bar, 1)
        self.bottom_widget.setLayout(self.bottom_layout)

        # ---------- 最外层 -------------- #
        self.layout0.addWidget(self.top_widget)
        self.layout0.addWidget(self.show_label)
        self.layout0.addWidget(self.bottom_widget)
        self.setLayout(self.layout0)

    def setPara(self):
        self.setStyleSheet("background-color:white; border-radius:10px;")
        self.layout0.setContentsMargins(0, 0, 0, 0)

        self.top_label1.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.top_label2.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.top_label3.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.setTopLabelContent(self.top_label1, "video🏀image🏀camera")
        self.setTopLabelContent(self.top_label2, "📄 none")
        self.setTopLabelContent(self.top_label3, "⏳ none")

        self.top_layout.setContentsMargins(0, 0, 0, 0)
        self.top_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.top_widget.setMaximumHeight(35)

        self.video_proc_slider.setOrientation(Qt.Horizontal)
        self.video_proc_slider.setRange(0, 100)
        self.video_proc_slider.setStyleSheet(video_slider_style)

        self.bottom_widget.setStyleSheet("background-color:white; border-radius:10px;")
        self.bottom_layout.setContentsMargins(0, 0, 0, 0)
        self.bottom_widget.setMaximumHeight(150)
        self.bar.setMinimumHeight(120)
        # -------------------- 设置bar内控件属性 -------------------------- #
        self.video_play_btn.setIconSize(QSize(25, 25))
        self.back_btn.setIconSize(QSize(25, 25))
        self.forward_btn.setIconSize(QSize(25, 25))
        self.volume_btn.setIconSize(QSize(25, 25))
        self.show_time_label_left.setFont(QFont("", 11))
        self.show_time_label_right.setFont(QFont("", 11))

    def setTopLabelContent(self, _label_, content):
        _label_.setText(content)
        _label_.setFont(QFont("", 16))

    def setTopAllLabels(self, str1, str2, str3):
        self.top_label1.setText(str1)
        self.top_label2.setText(f"📄 {str2}")
        self.top_label3.setText(f"{str3}")
        self.update()

    def setTimeLabel(self, time_str):
        self.top_label3.setText(f"⏳ {time_str}s")
        self.update()

    def getShowLabel(self):
        return self.show_label

    def changeVideoPlayBtn(self):
        if self.current_play_id == 0:
            self.video_play_btn.setIcon(FluentIcon.PAUSE_BOLD)
            self.current_play_id = 1
            # 发出开始视频信号
            self.vid_started_signal.emit()
        else:
            self.video_play_btn.setIcon(FluentIcon.PLAY_SOLID)
            self.current_play_id = 0
            # 发出结束视频信号
            self.vid_stopped_signal.emit()

    def signalSlots(self):
        self.video_play_btn.clicked.connect(self.changeVideoPlayBtn)

        # 菜单项事件
        self.action1.triggered.connect(self.enLargeImage)
        self.action4.triggered.connect(self.cameraDetectStart)

    # 给box增亮
    def highLightBoxes(self, plot_list):
        x1, y1, x2, y2 = plot_list[0], plot_list[1], plot_list[2], plot_list[3]
        dx, dy = plot_list[4], plot_list[5]
        conf = plot_list[6]
        cls = plot_list[7]
        if not self.show_label.pixmap() is None:
            pix = self.show_label.pixmap()
            box_painter = self.setBoxPainter(pix)
            rect_x, rect_y = min(x1, x2), min(y1, y2)
            rect_w, rect_h = abs(x1 - x2), abs(y1 - y2)
            box_painter.drawRect(rect_x, rect_y, rect_w, rect_h)

            # 浮动标签
            self.float_label.setText(cls + ":" + str(conf)[:4])
            self.float_label.adjustSize()
            self.float_label.setStyleSheet("background-color:black; color:white; border-radius:5px")
            self.float_label.move(rect_x + dx - 5, rect_y + dy - 5)
            self.update()

    # 后续会根据设置的宽度来
    def setBoxPainter(self, father):
        box_painter = QPainter(father)
        pen = QPen()
        pen.setColor(Qt.red)
        pen.setWidth(5)
        box_painter.setPen(pen)
        box_painter.setBrush(Qt.NoBrush)
        box_painter.setRenderHint(QPainter.Antialiasing)
        return box_painter

    def contextMenuEvent(self, event):
        action = self.menu.exec_(self.mapToGlobal(event.pos()))

    def enLargeImage(self):
        img_path = str(self.top_label2.text())[2:]
        self.enlarged_window = EnlargedWindow()
        self.enlarged_window.setPath(img_path)
        self.enlarged_window.show()

    def cameraDetectStart(self):
        self.vid_shishi_detect.emit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    show_widget = ShowWidget()
    show_widget.show()
    rect = show_widget.show_label.setPixmap(QPixmap("./icons/bg.png"))
    show_widget.highLightBoxes([0, 0, 100, 100, 0, 0, 0.12, "dick"])
    sys.exit(app.exec_())