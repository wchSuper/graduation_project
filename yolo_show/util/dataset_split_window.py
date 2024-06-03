import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, \
    QFileDialog, QMessageBox
from PyQt5.QtCore import Qt
from dataset_splitter import split_coco_dataset


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('COCO Dataset Splitter')
        self.setGeometry(100, 100, 600, 200)

        # Layouts
        layout = QVBoxLayout()
        input_layout = QHBoxLayout()
        ratio_layout = QHBoxLayout()
        button_layout = QHBoxLayout()

        # Widgets
        self.input_label = QLabel('COCO Dataset Directory:')
        self.input_edit = QLineEdit()
        self.input_button = QPushButton('Browse')
        self.input_button.clicked.connect(self.browseDirectory)

        self.ratio_label = QLabel('Train/Val/Test Ratio (e.g. 0.7, 0.2, 0.1):')
        self.ratio_edit = QLineEdit('0.7, 0.2, 0.1')

        self.split_button = QPushButton('Split Dataset')
        self.split_button.clicked.connect(self.splitDataset)

        # Assemble layouts
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_edit)
        input_layout.addWidget(self.input_button)

        ratio_layout.addWidget(self.ratio_label)
        ratio_layout.addWidget(self.ratio_edit)

        button_layout.addStretch(1)
        button_layout.addWidget(self.split_button)

        layout.addLayout(input_layout)
        layout.addLayout(ratio_layout)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def browseDirectory(self):
        directory = QFileDialog.getExistingDirectory(self, 'Select COCO Dataset Directory')
        if directory:
            self.input_edit.setText(directory)

    def splitDataset(self):
        dataset_dir = self.input_edit.text()
        ratio = self.ratio_edit.text()

        if not os.path.isdir(dataset_dir):
            QMessageBox.warning(self, 'Error', 'Invalid COCO Dataset Directory')
            return

        try:
            train_ratio, val_ratio, test_ratio = map(float, ratio.split(','))
        except ValueError:
            QMessageBox.warning(self, 'Error', 'Invalid Ratio Format')
            return

        if train_ratio + val_ratio + test_ratio != 1.0:
            QMessageBox.warning(self, 'Error', 'The sum of the ratios must be 1')
            return

        self.splitCocoDataset(dataset_dir, train_ratio, val_ratio, test_ratio)
        QMessageBox.information(self, 'Success', 'Dataset Split Successfully')

    def splitCocoDataset(self, dataset_dir, train_ratio, val_ratio, test_ratio):
        split_coco_dataset(dataset_dir, train_ratio, val_ratio, test_ratio)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
