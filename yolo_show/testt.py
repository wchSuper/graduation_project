import sys
import os
import random
import shutil
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


class DatasetSplitter(QWidget):
    def __init__(self):
        super().__init__()

        self.left_widget = DatasetSelector()
        self.middle_widget = FileBrowser()
        self.right_widget = DirectoryCreator()
        self.h_layout = QHBoxLayout(self)
        self.h_layout.addWidget(self.left_widget)
        self.h_layout.addWidget(self.middle_widget)
        self.h_layout.addWidget(self.right_widget)

        # slot
        self.middle_widget.on_change_path.connect(self.set_dataset_path)

        self.resize(1200, 600)

    def set_dataset_path(self, path_str):
        param_dict = self.left_widget.get_info()
        dataset_name = param_dict.get("dataset_name")
        translate_dict = {"训练集": "train", "测试集": "test", "验证集": "val"}
        self.left_widget.set_lineEdit2(path_str, translate_dict[dataset_name])


class FileBrowser(QWidget):
    on_change_path = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        # 设置窗口布局
        self.layout = QVBoxLayout()

        # 文件选择布局
        self.path_layout = QHBoxLayout()
        self.path_label = QLabel("选择文件夹:", self)
        self.path_edit = QLineEdit(self)
        self.path_edit.setPlaceholderText("输入文件夹路径...")
        self.path_edit.returnPressed.connect(self.update_tree_from_path)
        self.browse_button = QPushButton("Browse", self)
        self.browse_button.clicked.connect(self.browse_folder)
        self.path_layout.addWidget(self.path_label)
        self.path_layout.addWidget(self.path_edit)
        self.path_layout.addWidget(self.browse_button)

        # 文件树视图
        self.tree_view = QTreeView(self)
        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(['Name'])
        self.tree_view.setModel(self.model)
        self.tree_view.setEditTriggers(QTreeView.NoEditTriggers)  # 禁止编辑

        # 添加组件到主布局
        self.layout.addLayout(self.path_layout)
        self.layout.addWidget(self.tree_view)
        self.setLayout(self.layout)

    def browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.path_edit.setText(folder_path)
            self.populate_tree(folder_path)

    def update_tree_from_path(self):
        folder_path = self.path_edit.text()
        if QDir(folder_path).exists():
            self.populate_tree(folder_path)
        else:
            self.path_edit.setText("")

    def populate_tree(self, folder_path):
        self.model.clear()
        self.model.setHorizontalHeaderLabels(['Name'])
        root = QStandardItem(folder_path)
        self.model.appendRow(root)
        self.add_items(root, folder_path)
        # 发出信号
        self.on_change_path.emit(str(self.path_edit.text()))

    def add_items(self, parent, path):
        for file_name in QDir(path).entryList(QDir.NoDotAndDotDot | QDir.AllEntries):
            file_path = QDir(path).filePath(file_name)
            item = QStandardItem(file_name)
            parent.appendRow(item)
            if QDir(file_path).exists():
                self.add_items(item, file_path)

    def get_file_path(self):
        return str(self.path_edit.text())


class DatasetSelector(QWidget):
    def __init__(self):
        super().__init__()

        # 设置主布局
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(20, 20, 20, 20)

        # ComboBox布局
        self.combo_layout = QHBoxLayout()
        self.combo_label = QLabel("选择数据集:")
        self.combo_box = QComboBox()
        self.combo_box.addItems(["训练集", "测试集", "验证集"])
        self.combo_layout.addWidget(self.combo_label)
        self.combo_layout.addWidget(self.combo_box)
        self.combo_layout.setAlignment(Qt.AlignLeft)

        # LineEdit布局
        self.path_layout = QFormLayout()
        self.train_path_edit = QLineEdit()
        self.train_path_edit.setPlaceholderText("输入训练集路径...")
        self.test_path_edit = QLineEdit()
        self.test_path_edit.setPlaceholderText("输入测试集路径...")
        self.val_path_edit = QLineEdit()
        self.val_path_edit.setPlaceholderText("输入验证集路径...")
        self.path_layout.addRow("训练集路径:", self.train_path_edit)
        self.path_layout.addRow("测试集路径:", self.test_path_edit)
        self.path_layout.addRow("验证集路径:", self.val_path_edit)

        # SpinBox布局
        self.spinbox_layout = QFormLayout()
        self.train_spinbox = QSpinBox()
        self.train_spinbox.setRange(32, 99999)
        self.train_spinbox.setSingleStep(32)
        self.test_spinbox = QSpinBox()
        self.test_spinbox.setRange(32, 99999)
        self.test_spinbox.setSingleStep(32)
        self.val_spinbox = QSpinBox()
        self.val_spinbox.setRange(32, 99999)
        self.val_spinbox.setSingleStep(32)
        self.spinbox_layout.addRow("训练集图片张数:", self.train_spinbox)
        self.spinbox_layout.addRow("测试集图片张数:", self.test_spinbox)
        self.spinbox_layout.addRow("验证集图片张数:", self.val_spinbox)

        # 非目录文件数量标签
        self.file_count_label = QLabel("")
        self.update_file_counts()

        # 确定按钮
        self.confirm_button = QPushButton("确定")
        self.confirm_button.setStyleSheet("background-color: #4CAF50; "
                                          "color: white; padding: 10px 20px; font-size: 16px;")
        self.confirm_button.clicked.connect(self.confirm_action)

        # 写入按钮
        self.write_button = QPushButton("写入")
        self.write_button.setStyleSheet("background-color: #FF5733; "
                                        "color: white; padding: 10px 20px; font-size: 16px;")
        self.write_button.clicked.connect(self.write_selected_images)
        self.write_button.setEnabled(False)

        # 添加组件到主布局
        self.layout.addLayout(self.combo_layout)
        self.layout.addLayout(self.path_layout)
        self.layout.addLayout(self.spinbox_layout)
        self.layout.addWidget(self.file_count_label, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.confirm_button, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.write_button, alignment=Qt.AlignCenter)

        self.setLayout(self.layout)
        self.setStyleSheet("""
            QLabel {
                font-size: 14px;
            }
            QLineEdit {
                padding: 5px;
                font-size: 14px;
            }
            QSpinBox {
                padding: 5px;
                font-size: 14px;
            }
            QComboBox {
                padding: 5px;
                font-size: 14px;
            }
        """)

        # Connect signals to update file counts
        self.train_path_edit.textChanged.connect(self.update_file_counts)
        self.test_path_edit.textChanged.connect(self.update_file_counts)
        self.val_path_edit.textChanged.connect(self.update_file_counts)

        # 一共多少张
        self.train_files, self.val_files, self.test_files = 0, 0, 0
        self.selected_images = {"train": [], "test": [], "val": []}
        self.selected_labels = {"train_label": [], "test_label": [], "val_label": []}

    def confirm_action(self):
        train_path = self.train_path_edit.text()
        test_path = self.test_path_edit.text()
        val_path = self.val_path_edit.text()

        train_count = self.train_spinbox.value()
        test_count = self.test_spinbox.value()
        val_count = self.val_spinbox.value()

        if train_path and test_path and val_path and train_count > 0 and test_count > 0 and val_count > 0:
            self.selected_images["train"] = self.select_images(train_path, train_count, "train")
            self.selected_images["test"] = self.select_images(test_path, test_count, "test")
            self.selected_images["val"] = self.select_images(val_path, val_count, "val")
            print(self.selected_images)
            print(self.selected_labels)

            # QMessageBox.information(self, "选择结果", f"训练集图片:\n{', '.join(self.selected_images['train'])}\n\n"
            #                                     f"测试集图片:\n{', '.join(self.selected_images['test'])}\n\n"
            #                                     f"验证集图片:\n{', '.join(self.selected_images['val'])}")
            QMessageBox.information(self, "选择结果", "OK")

            self.write_button.setEnabled(True)

    def update_file_counts(self):
        train_path = self.train_path_edit.text()
        test_path = self.test_path_edit.text()
        val_path = self.val_path_edit.text()

        self.train_files = self.get_file_count(train_path)
        self.test_files = self.get_file_count(test_path)
        self.val_files = self.get_file_count(val_path)

        self.file_count_label.setText(
            f"训练集文件数: {self.train_files} | 测试集文件数: {self.test_files} | 验证集文件数: {self.val_files}"
        )

    def select_images(self, path, count, status):
        images = [f for f in os.listdir(path)
                  if os.path.isfile(os.path.join(path, f))
                  and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        if len(images) < count:
            QMessageBox.warning(self, "错误", f"指定路径中图片数量不足 {count} 张")
            return []

        # path上两级目录路径
        last2_directory = os.path.abspath(os.path.join(path, "../.."))
        label_directory = os.path.join(last2_directory, "labels")
        train_label_directory, val_label_directory, train_labels, val_labels = "", "", "", ""
        selected_labels = []

        # 从 0 - len(images) 选出 count 个不重复的数字
        random_numbers = random.sample(range(0, len(images)), count)
        selected_images = [images[i] for i in random_numbers]

        if status == "train":
            train_label_directory = os.path.join(label_directory, "train2017")
            train_labels = [f for f in os.listdir(train_label_directory)
                            if os.path.isfile(os.path.join(train_label_directory, f))
                            and f.lower().endswith('.txt')]
            selected_labels = [train_labels[i] for i in random_numbers]
            self.selected_labels["train_label"] = selected_labels
        elif status == "val":
            val_label_directory = os.path.join(label_directory, "val2017")
            val_labels = [f for f in os.listdir(val_label_directory)
                          if os.path.isfile(os.path.join(val_label_directory, f))
                          and f.lower().endswith('.txt')]
            selected_labels = [val_labels[i] for i in random_numbers]
            self.selected_labels["val_label"] = selected_labels

        return selected_images

    def write_selected_images(self):
        base_dir = self.selected_images_dir()
        if not base_dir:
            return

        for category in ["train", "test", "val"]:
            src_path = self.get_dataset_path(category)
            dst_images_path = os.path.join(base_dir, "images", category)
            dst_description_path = os.path.join(base_dir, "des")
            dst_labels_path = os.path.join(base_dir, "labels", category)

            if not os.path.exists(dst_images_path):
                os.makedirs(dst_images_path)
            if not os.path.exists(dst_description_path):
                os.makedirs(dst_description_path)
            if not os.path.exists(dst_labels_path):
                os.makedirs(dst_labels_path)

            description_file_path = os.path.join(dst_description_path, f"{category}.txt")
            with open(description_file_path, "w") as label_file:
                for i in range(len(self.selected_images[category])):
                    image = self.selected_images[category][i]
                    src_file1 = os.path.join(src_path, image)
                    dst_file1 = os.path.join(dst_images_path, image)
                    print(image, src_file1, dst_file1)
                    shutil.copy(src_file1, dst_file1)
                    print(self.selected_labels[f"{category}_label"])
                    if len(self.selected_labels[f"{category}_label"]):
                        label = self.selected_labels[f"{category}_label"][i]
                        src_file2 = os.path.join(src_path, label)
                        dst_file2 = os.path.join(dst_labels_path, label)
                        shutil.copy(src_file2, dst_file2)
                        print(label, src_file2, dst_file2)
                    label_file.write(dst_file1 + "\n")

        QMessageBox.information(self, "成功", "选定的图片已成功复制到目标文件夹中，并生成了标签文件")

    def get_dataset_path(self, dataset_type):
        if dataset_type == "train":
            return self.train_path_edit.text()
        elif dataset_type == "test":
            return self.test_path_edit.text()
        elif dataset_type == "val":
            return self.val_path_edit.text()
        return ""

    def selected_images_dir(self):
        if not self.selected_images["train"] and not self.selected_images["test"] and \
                not self.selected_images["val"]:
            QMessageBox.warning(self, "错误", "没有选择任何图片")
            return None

        directory_creator = self.parent().right_widget
        if not isinstance(directory_creator, DirectoryCreator):
            QMessageBox.warning(self, "错误", "无法获取目标文件夹")
            return None

        return directory_creator.get_latest_dataset_dir()

    def get_file_count(self, directory):
        if not QDir(directory).exists():
            return 0
        dir = QDir(directory)
        dir.setFilter(QDir.NoDotAndDotDot | QDir.Files)
        file_count = len(dir.entryList())
        return file_count

    def get_info(self):
        train_path = self.train_path_edit.text()
        test_path = self.test_path_edit.text()
        val_path = self.val_path_edit.text()
        train_count = self.train_spinbox.value()
        test_count = self.test_spinbox.value()
        val_count = self.val_spinbox.value()
        return {"dataset_name": self.combo_box.currentText(), "train_path": train_path, "test_path": test_path,
                "val_path": val_path, "train_count": train_count, "test_count": test_count, "val_count": val_count}

    def set_lineEdit(self, path1, path2, path3):
        self.train_path_edit.setText(path1)
        self.test_path_edit.setText(path2)
        self.val_path_edit.setText(path3)

    def set_lineEdit2(self, path, dataset_str):
        if dataset_str not in ["train", "test", "val"]:
            return
        if dataset_str == "train":
            self.train_path_edit.setText(path)
        elif dataset_str == "test":
            self.test_path_edit.setText(path)
        else:
            self.val_path_edit.setText(path)


class DirectoryCreator(QWidget):
    def __init__(self):
        super().__init__()

        # 设置主布局
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(20, 20, 20, 20)

        # 路径选择布局
        self.path_layout = QHBoxLayout()
        self.path_label = QLabel("目的文件夹路径:")
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("输入文件路径或选择...")
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_folder)
        self.path_layout.addWidget(self.path_label)
        self.path_layout.addWidget(self.path_edit)
        self.path_layout.addWidget(self.browse_button)

        # 目录创建按钮
        self.create_button = QPushButton("创建目录")
        self.create_button.setStyleSheet(
            "background-color: #4CAF50; color: white; padding: 10px 20px; font-size: 16px;")
        self.create_button.clicked.connect(self.create_directories)

        # 可视化展示创建过程的列表框
        self.visualization_list = QListWidget()

        # 添加组件到主布局
        self.layout.addLayout(self.path_layout)
        self.layout.addWidget(self.create_button, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.visualization_list)

        self.setLayout(self.layout)
        self.setWindowTitle("Directory Creator")
        self.setStyleSheet("""
            QLabel {
                font-size: 14px;
            }
            QLineEdit {
                padding: 5px;
                font-size: 14px;
            }
            QPushButton {
                padding: 5px;
                font-size: 14px;
            }
        """)

        self.latest_dataset_dir = ""

    def browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.path_edit.setText(folder_path)

    def create_directories(self):
        base_path = self.path_edit.text()
        if not base_path:
            QMessageBox.warning(self, "错误", "请输入有效的文件路径")
            return

        try:
            # 创建目录并记录创建过程
            self.visualization_list.clear()
            dataset_dir = self.find_next_dataset_dir(base_path)
            os.makedirs(os.path.join(dataset_dir, "images"))
            os.makedirs(os.path.join(dataset_dir, "labels"))
            self.latest_dataset_dir = dataset_dir
            self.visualization_list.addItem(f"创建目录: {dataset_dir}")
            self.visualization_list.addItem(f"创建目录: {os.path.join(dataset_dir, 'images')}")
            self.visualization_list.addItem(f"创建目录: {os.path.join(dataset_dir, 'labels')}")
            QMessageBox.information(self, "成功", "目录创建完成")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"创建目录时出错: {e}")

    def find_next_dataset_dir(self, base_path):
        i = 1
        while True:
            dataset_dir = os.path.join(base_path, f"dataset{i}")
            if not os.path.exists(dataset_dir):
                return dataset_dir
            i += 1

    def get_latest_dataset_dir(self):
        return self.latest_dataset_dir


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DatasetSplitter()
    window.setWindowTitle("数据集分割重组界面")
    window.show()
    sys.exit(app.exec_())
