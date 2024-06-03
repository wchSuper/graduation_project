import sys
import time

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from qfluentwidgets.components.settings.setting_card import SettingIconWidget

from config import *
from qfluentwidgets import *


class TrainDataWindow(QWidget):

    # train params
    train_param_signal = pyqtSignal(list)
    # widget params
    widget_param_signal = pyqtSignal(list)

    def __init__(self):
        super(TrainDataWindow, self).__init__()
        self.top_widget = QWidget()

        # ----------- top_widget  中控件 -------------#
        self.icon_label, self.title_label = SettingIconWidget(FluentIcon.SETTING), QLabel("title_label")
        self.select_combo = ComboBox()

        # ================    bottom_widget  =============== #
        self.param_widget = QStackedWidget()
        self.page1, self.page2 = QWidget(), QWidget()
        self.page1_layout, self.page2_layout = QVBoxLayout(self.page1), QVBoxLayout(self.page2)
        self.scroll_area1, self.scroll_area2 = ScrollArea(), ScrollArea()
        # ------------ page1 中控件 ------------------------ #
        self.default_widget1 = QWidget()
        self.add_btn, self.del_btn, self.clear_btn = PushButton(), PushButton(), PushButton()
        self.default_layout1 = QHBoxLayout(self.default_widget1)
        # ------------ page2 中控件 ------------------------ #

        # ---------- 最外层 | 顶部 | 底部  layout ---------------- #
        self.vlayout = QVBoxLayout()
        self.hlayout1 = QHBoxLayout()
        # self.vlayout1 = QVBoxLayout()  底部用别的代替了=> page1_layout | page2_layout

        # page1中的small widgets list AND current selected content
        self.small_widgets_list = []
        self.small_widgets_list2 = []
        # page2中的small widgets list AND current selected content
        self.small_widgets_content_dict = {}
        self.small_widgets_content_dict2 = {}
        # page1 and page2  name: 控件
        self.small_widgets_dict1 = {}
        self.small_widgets_dict2 = {}

        # 特殊控件
        self.color_btn = PushButton()
        self.color_choose = QColor(0, 255, 0)

        # 初始化
        self.initParam()
        self.initStructure()
        self.slotConnection()

    def initStructure(self):
        # -------------- 顶部 -------------------- #
        self.hlayout1.addWidget(self.icon_label)
        self.hlayout1.addWidget(self.title_label)
        self.hlayout1.addWidget(self.select_combo)
        self.top_widget.setLayout(self.hlayout1)
        # -------------- 底部 page1-------------------- #
        self.default_layout1.addWidget(self.add_btn)
        self.default_layout1.addWidget(self.del_btn)
        self.default_layout1.addWidget(self.clear_btn)
        self.page1_layout.addWidget(self.default_widget1)
        # ----  test 界面
        # for i in range(100):
        #     widget_i = self.createParams()
        #     self.page1_layout.addWidget(widget_i)
        # ---- page1填充参数小widget ------ #
        self.createSmallWidgets()

        self.scroll_area1.setWidget(self.page1)
        # -------------- 底部 page2-------------------- #

        # ---- page2填充参数小widget ------ #
        self.createSmallWidgets2()
        self.scroll_area2.setWidget(self.page2)

        # -------------- stackedWidget ---------------- #
        self.param_widget.addWidget(self.scroll_area1)
        self.param_widget.addWidget(self.scroll_area2)

        # 最外层
        self.vlayout.addWidget(self.top_widget)
        self.vlayout.addWidget(self.param_widget)
        self.setLayout(self.vlayout)

    def initParam(self):
        self.vlayout.setContentsMargins(0, 0, 0, 0)
        self.hlayout1.setContentsMargins(10, 0, 10, 0)
        self.icon_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.title_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        self.top_widget.setMaximumHeight(100)

        # 滚动条属性

        # icon label size  在hlayout1中
        self.icon_label.setFixedSize(25, 25)
        title_font = self.customFont(weight="1")
        self.title_label.setText("参数所属")
        self.title_label.setFont(title_font)
        self.select_combo.addItem("训练参数")
        self.select_combo.addItem("界面参数")

        # btn size
        self.add_btn.setIcon(FluentIcon.ADD)
        self.add_btn.setIconSize(QSize(20, 20))
        self.del_btn.setIcon(FluentIcon.DELETE)
        self.del_btn.setIconSize(QSize(20, 20))
        self.clear_btn.setIcon(CLEAR_ICON)
        self.clear_btn.setIconSize(QSize(20, 20))

        # style_sheet
        self.top_widget.setStyleSheet("background-color: white; border-radius: 10px")
        self.param_widget.setStyleSheet("background-color: white; border-radius: 10px")
        self.add_btn.setStyleSheet("background:transparent")
        self.del_btn.setStyleSheet("background:transparent")
        self.clear_btn.setStyleSheet("background:transparent")

        # set default value

    def slotConnection(self):
        self.select_combo.currentIndexChanged.connect(self.stackIndexChanged)

        for small_widget in self.small_widgets_list:
            labels = small_widget.findChildren(QLabel)
            combos = small_widget.findChildren(ComboBox)
            spins = small_widget.findChildren(SpinBox)
            sliders = small_widget.findChildren(Slider)
            if len(combos):
                self.small_widgets_content_dict[f"{str(labels[0].text())}-c"] = combos[0].currentText()
                self.small_widgets_dict1[f"{str(labels[0].text())}-c"] = combos[0]
                combos[0].currentIndexChanged.connect(self.updateValues)
            else:
                self.small_widgets_content_dict[f"{str(labels[0].text())}-s"] = [spins[0].value(), sliders[0].value()]
                self.small_widgets_dict1[f"{str(labels[0].text())}-s"] = [spins[0], sliders[0]]
                spins[0].valueChanged.connect(sliders[0].setValue)
                sliders[0].valueChanged.connect(spins[0].setValue)
                spins[0].valueChanged.connect(self.updateValues)
                sliders[0].valueChanged.connect(self.updateValues)
        #
        self.color_btn.clicked.connect(self.chooseColor)

        # spinbox: epoch | batch-size | workers changed
        self.small_widgets_dict1["epoch-s"][0].valueChanged.connect(self.updateTrainParams1)
        self.small_widgets_dict1["batch-size-s"][0].valueChanged.connect(self.updateTrainParams2)
        self.small_widgets_dict1["workers-s"][0].valueChanged.connect(self.updateTrainParams3)


    def createSmallWidgets(self):
        """
            1. weight(combo)  2. cfg(combo) 3. data(combo) 4.device(combo) 5.optimizer(combo)
            6. epoch(spinbox-slider) 7. batch_size(spinbox-slider) 8. workers(spinbox-slider)
        """
        small_widget1 = self.createParams2(param_name="weight",
                                           combo_content=["yolov7-tiny.pt", "yolov7.pt", "yolov7x.pt"])
        small_widget2 = self.createParams2(param_name="cfg",
                                           combo_content=["training/yolov7-tiny.yaml", "training/yolov7.yaml"])
        small_widget3 = self.createParams2(param_name="data",
                                           combo_content=["training/coco.yaml", "training/coco128.yaml"])
        small_widget4 = self.createParams2(param_name="lr",
                                           combo_content=["0.01", "0.001"])
        small_widget5 = self.createParams2(param_name="device", combo_content=["GPU", "CPU"])
        small_widget6 = self.createParams2(param_name="optimizer", combo_content=["adam", "SGD"])
        small_widget7 = self.createParams(param_name="epoch", set_range=[100, 301], default=100)
        small_widget8 = self.createParams(param_name="batch-size", set_range=[1, 33], default=4)
        small_widget9 = self.createParams(param_name="workers", set_range=[1, 9], default=8)
        # 加入到列表中
        self.small_widgets_list = [small_widget1, small_widget2, small_widget3, small_widget4, small_widget5,
                                   small_widget6, small_widget7, small_widget8, small_widget9]
        for small_widget_i in self.small_widgets_list:
            self.page1_layout.addWidget(small_widget_i)

    def createSmallWidgets2(self):
        small_widget1 = self.createParams2(param_name="主题",
                                           combo_content=["白灰色", "黑白", "浅蓝"])
        small_widget2 = self.createParams(param_name="标题字号", set_range=[1, 61], default=14)
        small_widget3 = self.createParams(param_name="线粗", set_range=[1, 61], default=1)
        small_widget4 = self.createParams3(param_name="线色")
        self.small_widgets_list2 = [small_widget1, small_widget2, small_widget3, small_widget4]
        for small_widget_i in self.small_widgets_list2:
            self.page2_layout.addWidget(small_widget_i)

    # 动态创建参数小widget  === spin-slider
    def createParams(self, param_name="参数name", set_range=[0, 100], default=8):
        h_widget_i = QWidget()
        h_layout_i = QHBoxLayout(h_widget_i)
        name_label_i = QLabel(param_name)
        name_label_i.setFont(self.customFont(12))
        name_label_i.setFixedWidth(100)
        spin_box_i = SpinBox()
        slider_i = Slider(Qt.Horizontal)
        spin_box_i.setRange(set_range[0], set_range[1])
        spin_box_i.setFixedWidth(120)
        slider_i.setRange(set_range[0], set_range[1])
        spin_box_i.setValue(default)
        slider_i.setValue(default)
        h_layout_i.addWidget(name_label_i, 1)
        h_layout_i.addWidget(spin_box_i, 1)
        h_layout_i.addWidget(slider_i, 2)
        return h_widget_i

    # 动态创建第二种参数小widget  ==== combo
    def createParams2(self, param_name="参数name", combo_content=["1", "2"]):
        h_widget_i = QWidget()
        h_layout_i = QHBoxLayout(h_widget_i)
        name_label_i = QLabel(param_name)
        name_label_i.setFont(self.customFont(12))
        name_label_i.setFixedWidth(100)
        combo_i = ComboBox()
        for content in combo_content:
            combo_i.addItem(content)
        h_layout_i.addWidget(name_label_i)
        h_layout_i.addWidget(combo_i)
        return h_widget_i

    # 动态创建第三种参数小widget  ==== color selection
    def createParams3(self, param_name="参数name"):
        h_widget_i = QWidget()
        h_layout_i = QHBoxLayout(h_widget_i)
        name_label_i = QLabel(param_name)
        name_label_i.setFont(self.customFont(12))
        name_label_i.setFixedWidth(100)
        self.color_btn.setFixedSize(QSize(15, 15))
        self.color_btn.setStyleSheet(f"background-color:{self.color_choose.name()}")
        h_layout_i.addWidget(name_label_i, 1)
        h_layout_i.addWidget(self.color_btn, 2)
        return h_widget_i

    # choose color
    def chooseColor(self):
        color_ = QColorDialog.getColor()
        if color_.isValid():
            self.color_choose = color_
            self.updateBtnColor()

    # update color-btn background color
    def updateBtnColor(self):
        self.color_btn.setStyleSheet(f"background-color:{self.color_choose.name()}")
        self.update()

    # 自定义字体属性
    def customFont(self, font_size=16, font_color=QColor(Qt.black), font_family="Arial", weight=""):
        font_i = QFont(font_family, font_size)
        if len(weight) >= 1:
            font_i.setBold(True)
        return font_i

    # 改变stackWidget的page
    def stackIndexChanged(self, index):
        self.param_widget.setCurrentIndex(index)

    def updateValues(self):
        if len(self.small_widgets_content_dict) == 0:
            return
        # for key, values in self.small_widgets_content_dict.items():
        #     print(f"{key} : {values}")
        for small_widget in self.small_widgets_list:
            labels = small_widget.findChildren(QLabel)
            combos = small_widget.findChildren(ComboBox)
            spins = small_widget.findChildren(SpinBox)
            sliders = small_widget.findChildren(Slider)
            if len(combos):
                self.small_widgets_content_dict[f"{str(labels[0].text())}-c"] = combos[0].currentText()
            else:
                self.small_widgets_content_dict[f"{str(labels[0].text())}-s"] = [spins[0].value(), sliders[0].value()]
        # print(self.small_widgets_content_dict)

    # epoch
    def updateTrainParams1(self, val1):
        epoch, batch, worker = val1, self.small_widgets_content_dict["batch-size-s"][0], \
                               self.small_widgets_content_dict["workers-s"][0]
        self.train_param_signal.emit([epoch, batch, worker])

    # batch-size
    def updateTrainParams2(self, val2):
        epoch, batch, worker = self.small_widgets_content_dict["epoch-s"][0], val2, \
                               self.small_widgets_content_dict["workers-s"][0]
        self.train_param_signal.emit([epoch, batch, worker])

    # workers
    def updateTrainParams3(self, val3):
        epoch, batch, worker = self.small_widgets_content_dict["epoch-s"][0], \
                               self.small_widgets_content_dict["batch-size-s"][0], val3
        self.train_param_signal.emit([epoch, batch, worker])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    train_data_window = TrainDataWindow()
    train_data_window.show()
    sys.exit(app.exec_())
