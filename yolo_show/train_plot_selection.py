from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from qfluentwidgets import *
import sys
from config import *


class TrainPlotSelectedWidget(QWidget):

    def __init__(self):
        super(TrainPlotSelectedWidget, self).__init__()

        # top_widget
        #
        # selected_list
        self.outer_layout = QVBoxLayout(self)
        self.top_widget = QWidget()
        self.selected_list = ListWidget()

        # top_widget => (hlayout) 3 button(add, delete, clear)
        self.top_layout = QHBoxLayout(self.top_widget)
        self.add_btn, self.del_btn, self.clr_btn = PushButton(), PushButton(), PushButton()

        # save item
        self.item_info_dict = {}

        self.initStructure()
        self.initParam()

    def initStructure(self):
        # top layout
        self.top_layout.addWidget(self.add_btn)
        self.top_layout.addWidget(self.del_btn)
        self.top_layout.addWidget(self.clr_btn)

        # add item  default setting
        self.defaultPlotName()

        # outer layout
        self.outer_layout.addWidget(self.top_widget)
        self.outer_layout.addWidget(self.selected_list)

    def initParam(self):
        self.setFixedSize(QSize(400, 500))
        # style sheet
        self.add_btn.setStyleSheet("background: transparent;")
        self.del_btn.setStyleSheet("background: transparent;")
        self.clr_btn.setStyleSheet("background: transparent;")
        self.top_widget.setStyleSheet("background-color: white; border-radius: 10px")
        self.selected_list.setStyleSheet("background-color: white; border-radius: 10px")
        # icon
        self.add_btn.setIcon(FluentIcon.ADD)
        self.del_btn.setIcon(FluentIcon.DELETE)
        self.clr_btn.setIcon(CLEAR_ICON)
        self.add_btn.setIconSize(QSize(20, 20))
        self.del_btn.setIconSize(QSize(20, 20))
        self.clr_btn.setIconSize(QSize(20, 20))

    # 自定义添加item
    def addItem(self, text):
        # 创建QListWidgetItem
        item = QListWidgetItem()
        self.selected_list.addItem(item)

        custom_widget = QWidget()
        custom_layout = QHBoxLayout(custom_widget)

        check_box = CheckBox()
        plot_name = QLabel(text)
        plot_name.setFont(QFont("Arial", 14))

        # save plot_name: check_box
        self.item_info_dict[str(text)] = [check_box, plot_name]
        custom_layout.addWidget(check_box, 1)
        custom_layout.addWidget(plot_name, 2)

        # 设置自定义的widget为item的widget
        self.selected_list.setItemWidget(item, custom_widget)

        # 调整item的大小以适应自定义widget
        item.setSizeHint(custom_widget.sizeHint())

    def defaultPlotName(self):
        self.addItem("epoch-Precision")
        self.addItem("epoch-Recall")
        self.addItem("epoch-mAP@0.5")
        self.addItem("epoch-mAP@0.5:0.95")
        self.addItem("epoch-Loss")

    def getItemInfo(self):
        cnt = self.selected_list.count()
        list1 = []
        for idx in range(cnt):
            item = self.selected_list.item(idx)
            custom_widget = self.selected_list.itemWidget(item)
            check_box = custom_widget.layout().itemAt(0).widget()
            plot_label = custom_widget.layout().itemAt(1).widget()
            list1.append(str(plot_label.text()))
        return list1


if __name__ == "__main__":
    app = QApplication(sys.argv)
    train_plot_select = TrainPlotSelectedWidget()
    train_plot_select.show()
    sys.exit(app.exec_())
