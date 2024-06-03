from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from config import *
from qfluentwidgets import ComboBox
import sys


class TrainResultWidget(QWidget):

    def __init__(self, all_train_data_list):
        super(TrainResultWidget, self).__init__()

        # data analyse
        train_times = len(all_train_data_list)
        all_train_param_list = all_train_data_list[train_times-1]
        epochs = len(all_train_param_list)
        self.page1_param_list = []
        self.page2_param_list = []
        self.page3_param_list = []
        self.category_list = []

        self.setPageParams(all_train_param_list)
        self.setCategory(epochs)
        self.line1, self.col1 = len(self.category_list), len(self.page1_param_list[0])
        self.line2, self.col2 = len(self.page2_param_list[0]), 1
        self.line3, self.col3 = len(self.category_list), len(self.page3_param_list[0])

        # layout
        self.outer_layout = QVBoxLayout(self)

        self.top_widget = QWidget()
        self.top_layout = QHBoxLayout(self.top_widget)
        # top
        self.left_btn, self.right_btn = QPushButton("<-"), QPushButton("->")
        self.top_label = QLabel("Result")

        # bottom
        self.bottom_widget = QStackedWidget()

        # page1
        self.page1_title = QLabel("Precision---Recall---mAP50---mAP95---Loss   Table")
        self.table_widget1 = TableWidget(self.line1, self.col1, self.category_list, param_names1, self.page1_param_list)
        self.page1 = QWidget()
        self.page1_layout = QVBoxLayout(self.page1)
        # page2
        self.page2_title_widget = QWidget()
        self.page2_title_layout = QHBoxLayout(self.page2_title_widget)
        self.page2_title_label = QLabel("epoch")
        self.page2_title_combo = ComboBox()
        self.table_widget2 = TableWidget(self.line2, self.col2, class_names, param_names2,
                                         self.Reshape(self.page2_param_list[0]))
        self.page2 = QWidget()
        self.page2_layout = QVBoxLayout(self.page2)
        # page3
        self.page3_title = QLabel("detection time---NMS time---all time Table")
        self.table_widget3 = TableWidget(self.line3, self.col3, self.category_list, param_names3, self.page3_param_list)
        self.page3 = QWidget()
        self.page3_layout = QVBoxLayout(self.page3)

        self.initParams()
        self.initStructure()
        self.slotConnection()

        self.resize(800, 600)

    def setPageParams(self, all_train_param_list):
        for train_param_list in all_train_param_list:
            epoch = train_param_list[0]
            results = train_param_list[1]
            maps = train_param_list[2]  # all classes  -- page2 params
            times = train_param_list[3]  # test time    -- page3 params

            # page1 params
            m_precision = str(results[0])[:4]
            m_recall = str(results[1])[:4]
            map_05 = str(results[2])[:4]
            map_95 = str(results[3])[:4]
            loss = str(results[4] + results[5] + results[6])[:4]
            self.page1_param_list.append([m_precision, m_recall, map_05, map_95, loss])
            self.page2_param_list.append(maps)
            self.page3_param_list.append([str(times[0])[:4], str(times[1])[:4], str(times[2])[:4]])

    def setCategory(self, epoch_num):
        for epoch in range(epoch_num):
            self.category_list.append(f"epoch-{epoch}")

    def initStructure(self):
        # top
        self.top_layout.addWidget(self.left_btn)
        self.top_layout.addWidget(self.top_label)
        self.top_layout.addWidget(self.right_btn)

        # bottom
        # page1
        self.page1_layout.addWidget(self.page1_title)
        self.page1_layout.addWidget(self.table_widget1)
        # page2
        # page2 - title
        self.page2_title_layout.addWidget(self.page2_title_label)
        self.page2_title_layout.addWidget(self.page2_title_combo)
        self.page2_layout.addWidget(self.page2_title_widget)
        self.page2_layout.addWidget(self.table_widget2)
        # page3
        self.page3_layout.addWidget(self.page3_title)
        self.page3_layout.addWidget(self.table_widget3)
        # add to stack
        self.bottom_widget.addWidget(self.page1)
        self.bottom_widget.addWidget(self.page2)
        self.bottom_widget.addWidget(self.page3)

        # outer
        self.outer_layout.addWidget(self.top_widget)
        self.outer_layout.addWidget(self.bottom_widget)

    def initParams(self):
        self.top_label.setFont(QFont("Arial", 14))
        self.page1_title.setFont(QFont("Arial", 14))
        self.top_label.setAlignment(Qt.AlignCenter)
        self.page1_title.setAlignment(Qt.AlignCenter)
        self.left_btn.setFont(QFont("Arial", 14))
        self.right_btn.setFont(QFont("Arial", 14))

        for category in self.category_list:
            self.page2_title_combo.addItem(category)
        self.page2_title_combo.setCurrentIndex(0)

    def slotConnection(self):
        self.left_btn.clicked.connect(self.changeCurrentPage1)
        self.right_btn.clicked.connect(self.changeCurrentPage2)

        self.page2_title_combo.currentIndexChanged.connect(self.updatePage2)

    # from [1, 2, 3, 4, 5....] to [[1], [2], [3], ...]
    def Reshape(self, list1):
        list2 = []
        for i in list1:
            list2.append([i])
        return list2

    def changeCurrentPage1(self):
        current_index = self.bottom_widget.currentIndex()
        nums = self.bottom_widget.count()
        current_index -= 1
        if current_index == -1:
            current_index = nums - 1
        self.bottom_widget.setCurrentIndex(current_index)

    def changeCurrentPage2(self):
        current_index = self.bottom_widget.currentIndex()
        nums = self.bottom_widget.count()
        current_index += 1
        if current_index == nums:
            current_index = 0
        self.bottom_widget.setCurrentIndex(current_index)

    def updatePage2(self, index):
        # TableWidget(line2, col2, class_names, ["mAP"], self.Reshape(self.page2_param_list[0]))
        self.table_widget2.update_table(self.line2, self.col2, class_names, ["mAP"],
                                        self.Reshape(self.page2_param_list[index]))


class TableWidget(QWidget):
    def __init__(self, lines=1, cols=1, categories=['class1'], param_names=['param1'], param_list=[[1]]):
        super().__init__()
        self.setWindowTitle('Parameter Table')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        # 创建表格widget
        self.table = QTableWidget(lines + 1, cols + 1, self)

        # horizontal 标签
        self.table.setHorizontalHeaderLabels(['类别名'] + param_names)

        # vertical   标签
        self.table.setVerticalHeaderLabels([''] * (lines + 1))

        # 类别名
        for i, category in enumerate(categories):
            self.table.setItem(i + 1, 0, QTableWidgetItem(category))

        # 参数名
        for i in range(lines):
            for j in range(cols):
                self.table.setItem(i + 1, j + 1, QTableWidgetItem(str(param_list[i][j])))

        layout.addWidget(self.table)
        self.setLayout(layout)

        self.setStyleSheet("background-color: white; border-radius:10px")

    def update_table(self, lines, cols, categories, param_names, param_list):
        self.table.setRowCount(lines + 1)
        self.table.setColumnCount(cols + 1)

        # 更新 horizontal 标签
        self.table.setHorizontalHeaderLabels(['类别名'] + param_names)

        # 更新 vertical 标签
        self.table.setVerticalHeaderLabels([''] * (lines + 1))

        # 更新类别名
        for i, category in enumerate(categories):
            self.table.setItem(i + 1, 0, QTableWidgetItem(category))

        # 更新参数值
        for i in range(lines):
            for j in range(cols):
                self.table.setItem(i + 1, j + 1, QTableWidgetItem(str(param_list[i][j])))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    train_result_widget = TrainResultWidget(all_train_data_list)
    train_result_widget.show()
    sys.exit(app.exec_())
