import sys
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton, QComboBox, \
    QListWidgetItem, QLabel, QWidget
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPalette, QFont
from config import *


class CategorySelector(QDialog):
    # update coco cls
    update_coco_cls_signal = pyqtSignal(list)

    def __init__(self, categories):
        super().__init__()
        self.setWindowTitle('COCO Category Selector')
        self.setGeometry(100, 100, 800, 600)

        self.selected_categories = {}
        self.list_widgets = [QListWidget(self), QListWidget(self), QListWidget(self)]
        self.current_list_widget = 0
        self.selected_items = {list_widget: [] for list_widget in self.list_widgets}

        # 主布局
        main_layout = QVBoxLayout()

        # 顶部布局
        top_layout = QHBoxLayout()

        # 类别选择下拉框
        self.combo_box = QComboBox(self)
        self.combo_box.addItems(categories)
        self.combo_box.setFont(QFont("Arial", 12))
        top_layout.addWidget(self.combo_box)

        # 添加类别按钮
        add_button = QPushButton('添加类别', self)
        add_button.setFont(QFont("Arial", 12))
        add_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 6px; border-radius: 5px;")
        add_button.clicked.connect(self.add_category)
        top_layout.addWidget(add_button)

        # 删除类别按钮
        remove_button = QPushButton('删除类别', self)
        remove_button.setFont(QFont("Arial", 12))
        remove_button.setStyleSheet("background-color: #f44336; color: white; padding: 6px; border-radius: 5px;")
        remove_button.clicked.connect(self.remove_category)
        top_layout.addWidget(remove_button)

        main_layout.addLayout(top_layout)

        # 底部三个水平排列的ListWidget
        list_layout = QHBoxLayout()
        for list_widget in self.list_widgets:
            list_widget.setSpacing(10)
            list_widget.itemDoubleClicked.connect(self.select_item)
            list_widget.setSelectionMode(QListWidget.NoSelection)  # 禁止单击选中
            list_layout.addWidget(list_widget)

        main_layout.addLayout(list_layout)

        self.setLayout(main_layout)

        # 默认添加类别
        self.default_categories = ['汽车', '公交车', '卡车', '人']
        for category in self.default_categories:
            self.add_category(category, default=True)

    def add_category(self, category=None, default=False):
        if not default:
            category = self.combo_box.currentText()

        if category not in self.selected_categories:
            self.selected_categories[category] = 1

            list_widget = self.list_widgets[self.current_list_widget]

            # 新行
            item_widget = QWidget()
            item_layout = QHBoxLayout()
            item_layout.setContentsMargins(5, 5, 5, 5)  # 增加边距
            item_layout.setSpacing(10)  # 设置间距
            item_widget.setLayout(item_layout)

            label = QLabel(category, self)
            label.setFixedSize(100, 30)
            label.setAlignment(Qt.AlignCenter)

            # 设置标签背景颜色为黑色，字体颜色为白色
            palette = label.palette()
            palette.setColor(QPalette.Window, QColor(Qt.black))
            palette.setColor(QPalette.WindowText, QColor(Qt.white))
            label.setAutoFillBackground(True)
            label.setPalette(palette)

            # 设置更好看的字体
            font = QFont("Arial", 12, QFont.Bold)
            label.setFont(font)

            item_layout.addWidget(label)

            list_item = QListWidgetItem(list_widget)
            list_item.setSizeHint(item_widget.sizeHint())
            list_widget.addItem(list_item)
            list_widget.setItemWidget(list_item, item_widget)

            # 切换到下一个 ListWidget
            self.current_list_widget = (self.current_list_widget + 1) % 3
        # print(self.selected_categories)
        self.update_coco_cls_signal.emit([self.selected_categories])

    def remove_category(self):
        for list_widget, items in self.selected_items.items():
            for item in items:
                row = list_widget.row(item)
                if row != -1:
                    item_widget = list_widget.itemWidget(item)
                    layout = item_widget.layout()
                    for j in range(layout.count()):
                        widget = layout.itemAt(j).widget()
                        if widget:
                            category_name = widget.text()
                            if category_name in self.selected_categories:
                                del self.selected_categories[category_name]
                    list_widget.takeItem(row)
            self.selected_items[list_widget] = []
        # print(self.selected_categories)
        self.update_coco_cls_signal.emit([self.selected_categories])

    def select_item(self, item):
        for list_widget in self.list_widgets:
            if list_widget.indexFromItem(item).isValid():
                if item not in self.selected_items[list_widget]:
                    self.selected_items[list_widget].append(item)
                    item.setSelected(True)
                    break


