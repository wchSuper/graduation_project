import sys
import os

import PyQt5.QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from qfluentwidgets import *
from config import *
from QSS.qss import *
from coco_select_dialog import *


class DataWidget(QWidget):

    on_iou_conf_changed = pyqtSignal(list)

    def __init__(self):
        super(DataWidget, self).__init__()
        self.title_label = QLabel("setting⚙")
        self.para_widget1 = QWidget()
        self.para_widget2 = QWidget()
        self.para_widget3 = QWidget()
        self.para_widget4 = QWidget()
        # 添加水平分隔线
        self.result_widget = QWidget()
        self.res_label = QLabel("results🏆")
        self.res_list_widget = ListWidget()
        self.res_layout = QVBoxLayout()
        self.layout1 = QVBoxLayout()

        # ------- para_widget1 内部 ------- #
        self.select_model_label = QLabel("model selecting✅")
        self.select_model_combo = ComboBox()
        self.para_layout1 = QHBoxLayout()

        # ------- para_widget2 内部 ---------------- #
        self.iou_label = QLabel("IoU")
        self.iou_double_spin_box = DoubleSpinBox()
        self.iou_slider = Slider()
        self.para_layout2 = QGridLayout()

        # ------- para_widget3 内部 ---------------- #
        self.conf_label = QLabel("Conf")
        self.conf_double_spin_box = DoubleSpinBox()
        self.conf_slider = Slider()
        self.para_layout3 = QGridLayout()

        # ------- para_widget4 内部 ---------------- #
        self.coco_label = QLabel("类别")
        self.coco_btn = PushButton()
        self.coco_input = LineEdit()
        self.para_layout4 = QHBoxLayout()

        # ----------- 变量 -------------------------- #
        self.coco_cls = {}
        self.coco_names = coco_names
        self.coco_names_zh = coco_names_zh
        self.category_dialog = None          # CategorySelector(self.coco_names_zh)

        # ----------- 方法 -------------------------- #
        self.setPara()
        self.initUI()
        self.signalSlots()

        self.iou_double_spin_box.setValue(45)
        self.conf_double_spin_box.setValue(25)

    def setPara(self):
        self.layout1.setContentsMargins(0, 0, 10, 5)
        font = QFont("Arial", 18, QFont.Bold)
        self.title_label.setFont(font)
        self.title_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("color: 		#2F4F4F")

        self.select_model_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.select_model_label.setAlignment(Qt.AlignLeft)
        self.select_model_label.setStyleSheet("color: 		#2F4F4F")
        self.para_layout1.setContentsMargins(0, 10, 0, 0)

        self.select_model_combo.addItem("yolov7-tiny.pt")
        self.select_model_combo.addItem("yolov7.pt")
        self.select_model_combo.addItem("yolov5s.pt")
        self.select_model_combo.addItem("yolov8n.pt")
        self.select_model_combo.addItem("yolov8s.pt")

        self.iou_label.setStyleSheet(label_style1)
        self.iou_slider.setOrientation(Qt.Horizontal)
        self.iou_slider.setRange(0, 100)
        self.iou_slider.setTickInterval(1)
        self.iou_slider.setTickPosition(QSlider.TicksBelow)
        self.iou_double_spin_box.setRange(0, 100)
        self.iou_double_spin_box.setSingleStep(0.1)
        self.para_layout2.setContentsMargins(0, 10, 0, 0)

        self.conf_label.setStyleSheet(label_style1)
        self.conf_slider.setOrientation(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setTickInterval(1)
        self.conf_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_double_spin_box.setRange(0, 100)
        self.conf_double_spin_box.setSingleStep(0.1)
        self.para_layout3.setContentsMargins(0, 10, 0, 10)

        self.coco_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.coco_btn.setIcon(FluentIcon.SEARCH)
        self.coco_btn.setIconSize(QSize(20, 20))
        self.coco_btn.setStyleSheet("background:transparent;")

        # ---------- 水平分割线添加阴影效果 ----------- #
        # self.horizon_line.setFrameShape(QFrame.HLine)
        # self.horizon_line.setFrameShadow(QFrame.Sunken)
        self.res_list_widget.setStyleSheet(res_list_widget_style)
        self.result_widget.setStyleSheet("background-color: white; border-radius: 10px;")
        self.res_label.setFont(font)
        self.res_label.setStyleSheet("color: 		#2F4F4F")
        self.res_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.res_label.setAlignment(Qt.AlignCenter)

    def initUI(self):

        # -------- para layout1  ------- #
        self.para_layout1.addWidget(self.select_model_label)
        self.para_layout1.addWidget(self.select_model_combo)
        self.para_widget1.setLayout(self.para_layout1)

        # -------- para layout2  ------- #
        self.para_layout2.addWidget(self.iou_label, 0, 0)
        self.para_layout2.addWidget(self.iou_double_spin_box, 1, 0)
        self.para_layout2.addWidget(self.iou_slider, 1, 1)
        self.para_widget2.setLayout(self.para_layout2)

        # -------- para layout3  ------- #
        self.para_layout3.addWidget(self.conf_label, 0, 0)
        self.para_layout3.addWidget(self.conf_double_spin_box, 1, 0)
        self.para_layout3.addWidget(self.conf_slider, 1, 1)
        self.para_widget3.setLayout(self.para_layout3)

        # -------- para layout4  ------- #
        self.para_layout4.addWidget(self.coco_label)
        self.para_layout4.addWidget(self.coco_btn)
        self.para_layout4.addWidget(self.coco_input)
        self.para_widget4.setLayout(self.para_layout4)

        # --------- result widget -------- #
        self.res_layout.addWidget(self.res_label)
        self.res_layout.addWidget(self.res_list_widget)
        self.result_widget.setLayout(self.res_layout)

        # 最外层layout
        self.layout1.addWidget(self.title_label)
        self.layout1.addWidget(self.para_widget1)
        self.layout1.addWidget(self.para_widget2)
        self.layout1.addWidget(self.para_widget3)
        self.layout1.addWidget(self.para_widget4)
        # self.layout1.addWidget(self.horizon_line)
        self.layout1.addWidget(self.result_widget)

        self.setLayout(self.layout1)

    # -------------- 设置layout的margin填充 -------------------------- #
    def setLayoutPad(self, _layout):
        _layout.setSpacing(0)

    def signalSlots(self):
        self.iou_double_spin_box.valueChanged.connect(self.updateIoUSliderValue)
        self.iou_slider.valueChanged.connect(self.updateIouSpinValue)
        self.conf_double_spin_box.valueChanged.connect(self.updateConfSliderValue)
        self.conf_slider.valueChanged.connect(self.updateConfSpinValue)

        self.coco_btn.clicked.connect(self.cocoSelection)

    def updateIoUSliderValue(self, value):
        self.iou_slider.setValue(value)
        self.on_iou_conf_changed.emit([self.iou_slider.value()/100, self.conf_slider.value()/100])

    def updateIouSpinValue(self, value):
        self.iou_double_spin_box.setValue(value)

    def updateConfSliderValue(self, value):
        self.conf_slider.setValue(value)
        self.on_iou_conf_changed.emit([self.iou_slider.value() / 100, self.conf_slider.value() / 100])

    def updateConfSpinValue(self, value):
        self.conf_double_spin_box.setValue(value)

    def getIoU_Conf_WValue(self):
        return self.iou_slider.value(), self.conf_slider.value(), self.select_model_combo.currentText(), \
               self.coco_cls

    def cocoSelection(self):
        if self.category_dialog is None:
            self.category_dialog = CategorySelector(self.coco_names_zh)
            self.category_dialog.update_coco_cls_signal.connect(self.updateCOCOCls)
        # self.coco_cls = self.category_dialog.selected_categories

        self.category_dialog.show()

    def updateCOCOCls(self, coco_list):
        self.coco_cls = coco_list[0]
        s = ""
        for key in self.coco_cls:
            s += key
            s += ";"
        self.coco_input.setText(s)

    # ------------------- 在res_list_widget内部添加结果参数 ------------------------- #
    def addParamList(self, file_name):
        three_lines, all_lines = [], []
        index = 1
        with open(file_name, 'r') as file:
            for line in file.readlines():
                line = line.strip()
                if index == 1:
                    three_lines.append(eval(line))
                elif index == 2:
                    three_lines.append(line)
                else:
                    three_lines.append(line)
                    all_lines.append(three_lines)
                    three_lines = []
                    index = 0
                index += 1
        # 获取到了all_lines 接下来就是把信息展示出来啦
        self.res_list_widget.clear()
        for line in all_lines:
            # self.res_list_widget.addItem(str(line[0]))
            # self.res_list_widget.addItem(str(line[1]))
            # self.res_list_widget.addItem(str(line[2]))
            self.res_list_widget.addItem(f"class {str(line[1])}   conf {str(line[2])}\n\n"
                                         f"左上 ({str(line[0][0])}, {str(line[0][1])})\n"
                                         f"右下 ({str(line[0][2])}, {str(line[0][3])})")


class PowerSettingCard(ExpandGroupSettingCard):

    def __init__(self, parent=None):
        super().__init__(FluentIcon.SPEED_OFF, "节电模式", "通过限制某些通知和后台活动降低电池消耗", parent)

        # 第一组
        self.modeButton = PushButton("立即启用")
        self.modeLabel = BodyLabel("节电模式")
        self.modeButton.setFixedWidth(135)

        # 第二组
        self.autoLabel = BodyLabel("自动开启节电模式")
        self.autoComboBox = ComboBox()
        self.autoComboBox.addItems(["10%", "20%", "30%"])
        self.autoComboBox.setFixedWidth(135)

        # 第三组
        self.lightnessLabel = BodyLabel("使用节电模式时屏幕亮度较低")
        self.lightnessSwitchButton = SwitchButton("关", self, IndicatorPosition.RIGHT)
        self.lightnessSwitchButton.setOnText("开")

        # 调整内部布局
        self.viewLayout.setContentsMargins(0, 0, 0, 0)
        self.viewLayout.setSpacing(0)

        # 添加各组到设置卡中
        self.add(self.modeLabel, self.modeButton)
        self.add(self.autoLabel, self.autoComboBox)
        self.add(self.lightnessLabel, self.lightnessSwitchButton)

    def add(self, label, widget):
        w = QWidget()
        w.setFixedHeight(60)

        layout = QHBoxLayout(w)
        layout.setContentsMargins(48, 12, 48, 12)

        layout.addWidget(label)
        layout.addStretch(1)
        layout.addWidget(widget)

        # 添加组件到设置卡
        self.addGroupWidget(w)


