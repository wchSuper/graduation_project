
from training_data_window import *
from train_plot_window import *
from main_window import *
from util.details import *
from models.yolo import *


class TrainWindow(QWidget):
    # from training_data_window.py
    # train params
    train_param_signal = pyqtSignal(list)
    # widget params
    widget_param_signal = pyqtSignal(list)

    def __init__(self):
        super(TrainWindow, self).__init__()

        # # # # # # # 大布局 # # # # # # # # #
        #         顶部 top_widget            # ---- 顶部放一个label
        #                                   #          /左侧plot_widget
        #         中部 show_widget           # ---- 中部
        # # # # # # # # # # # # # # # # # # #          \右侧data_widget
        self.top_widget, self.show_widget = QWidget(), QWidget()
        self.outer_layout = QVBoxLayout()

        # 顶部 top_widget  内部控件
        self.top_label = BodyLabel("模拟训练演示-COCO版")
        self.top_layout = QVBoxLayout()

        # 中部 show_widget 内部控件
        self.plot_widget = TrainPlotWindow()
        self.data_widget = TrainDataWindow()
        self.show_layout = QHBoxLayout(self.show_widget)

        # ----------- 一些字体的size color family ---------------- #
        self.size1, self.size2, self.size3 = 18, 18, 18
        self.color1, self.color2, self.color3 = QColor(Qt.black), QColor(Qt.black), QColor(Qt.black)
        self.family1, self.family2, self.family3 = "", "", ""

        #

        self.initStructure()
        self.initParam()
        self.slotConnection()

    def initStructure(self):
        self.resize(1200, 800)
        #  ----------- top ------------------ #
        self.top_label.setAlignment(Qt.AlignCenter)
        self.top_layout.addWidget(self.top_label, Qt.AlignCenter)
        self.top_widget.setLayout(self.top_layout)
        # ----------- middle ---------------- #
        self.show_layout.addWidget(self.plot_widget)
        self.show_layout.addWidget(self.data_widget)

        # ----------- outer ------------------- #
        self.outer_layout.addWidget(self.top_widget)
        self.outer_layout.addWidget(self.show_widget)
        self.setLayout(self.outer_layout)

    def initParam(self):
        # -----------------------------  设置样式 ---------------------------------- #
        self.top_widget.setStyleSheet("background-color: white; border-radius: 10px")
        self.show_widget.setStyleSheet("background-color: white; border-radius: 10px")

        # -----------------------------  设置尺寸 ---------------------------------- #
        self.top_widget.setMinimumHeight(80)
        self.top_widget.setMaximumHeight(80)

        self.data_widget.setMaximumWidth(300)

    def slotConnection(self):
        # draw plots
        # self.plot_widget.train_widget.on_plot_signal.connect(self.drawPlots)

        # 接收 data 界面发过来的信号list
        self.data_widget.train_param_signal.connect(self.updateTrainParams)

        self.data_widget.widget_param_signal.connect(self.updateWidgetParams)

    # update train params
    def updateTrainParams(self, train_param_list):
        epoch, batch, worker = train_param_list[0], train_param_list[1], train_param_list[2]
        # print({"epoch": epoch, "batch-size": batch, "workers": worker})
        self.plot_widget.setTrainParams({"epoch": epoch, "batch-size": batch, "workers": worker})

    # update widget params
    def updateWidgetParams(self, widget_param_list):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    train_window = TrainWindow()
    train_window.show()
    sys.exit(app.exec_())

