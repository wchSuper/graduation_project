import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QPainter, QPen, QBrush, QPolygon, QFont
from PyQt5.QtCore import Qt, QPoint


class DynamicLineChart(QWidget):
    def __init__(self, epoch_range=(0, 100), precision_range=(0.50, 1.00), plot_name="plot name"):
        super(DynamicLineChart, self).__init__()

        self.plot_name = plot_name
        self.plot_name_label = QLabel(self.plot_name)
        self.outer_layout = QVBoxLayout(self)
        self.plot_widget = DynamicLineChart2(epoch_range, precision_range)

        self.plot_name_label.setFont(QFont("Arial", 14))
        self.plot_name_label.setMaximumHeight(20)
        self.plot_name_label.setAlignment(Qt.AlignCenter)
        self.outer_layout.setContentsMargins(0, 0, 0, 0)

        self.outer_layout.addWidget(self.plot_name_label)
        self.outer_layout.addWidget(self.plot_widget)

        self.resize(800, 600)

    def add_data_point(self, epoch, precision):
        self.plot_widget.add_data_point(epoch, precision)

    def setCoordinateRange(self, x_range, y_range):
        self.plot_widget.epoch_range = x_range
        self.plot_widget.precision_range = y_range


class DynamicLineChart2(QWidget):
    def __init__(self, epoch_range=(0, 100), precision_range=(0.50, 1.00)):
        super().__init__()
        self.setWindowTitle('Dynamic Line Chart')
        self.setGeometry(100, 100, 800, 600)

        self.epoch_range = epoch_range
        self.precision_range = precision_range
        self.data_points = []

        self.setStyleSheet("background-color: white; border-radius: 10px")
        self.initUI()

    def setCoordinateRange(self, x_range, y_range):
        self.epoch_range = x_range
        self.precision_range = y_range

    def initUI(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

    def add_data_point(self, epoch, precision):
        # Simulate receiving a new data point
        self.data_points.append((epoch, precision))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        self.drawCoordinateSystem(painter)
        self.drawDataPoints(painter)

    def drawCoordinateSystem(self, painter):
        # Set up the painter
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(Qt.black, 2, Qt.SolidLine)
        painter.setPen(pen)

        # Get widget dimensions
        width = self.width()
        height = self.height()

        # Draw X and Y axis
        painter.drawLine(50, height - 50, width - 50, height - 50)  # X axis
        painter.drawLine(50, 50, 50, height - 50)  # Y axis

        # Draw solid arrows
        self.drawSolidArrow(painter, QPoint(width - 50, height - 50), QPoint(width - 60, height - 55), QPoint(width - 60, height - 45))  # X axis arrow
        self.drawSolidArrow(painter, QPoint(50, 50), QPoint(45, 60), QPoint(55, 60))  # Y axis arrow

        # Draw coordinate lines and labels
        self.drawCoordinates(painter, width, height)

    def drawSolidArrow(self, painter, p1, p2, p3):
        arrow_head = QPolygon([p1, p2, p3])
        painter.setBrush(Qt.black)
        painter.drawPolygon(arrow_head)

    def drawCoordinates(self, painter, width, height):
        pen = QPen(Qt.gray, 1, Qt.DotLine)
        painter.setPen(pen)

        # Draw lines and text along the X axis
        for i in range(self.epoch_range[0], self.epoch_range[1] + 1, 10):
            x = 50 + (i - self.epoch_range[0]) * (width - 100) // (self.epoch_range[1] - self.epoch_range[0])
            painter.drawLine(x, height - 50, x, 50)
            painter.drawText(x - 10, height - 30, str(i))

        # Draw lines and text along the Y axis
        for i in range(5, 11):
            precision = self.precision_range[0] + (i - 5) * (self.precision_range[1] - self.precision_range[0]) / 5
            y = height - 50 - int((precision - self.precision_range[0]) * (height - 100) / (self.precision_range[1] - self.precision_range[0]))
            painter.drawLine(50, y, width - 50, y)
            painter.drawText(20, y + 5, f'{precision:.2f}')

    def drawDataPoints(self, painter):
        if not self.data_points:
            return

        pen = QPen(Qt.red, 1, Qt.SolidLine)  # Set line thickness to 1
        brush = QBrush(Qt.red)
        painter.setPen(pen)
        painter.setBrush(brush)

        width = self.width()
        height = self.height()

        for i in range(len(self.data_points) - 1):
            x1 = int(50 + (self.data_points[i][0] - self.epoch_range[0]) * (width - 100) / (self.epoch_range[1] - self.epoch_range[0]))
            y1 = int(height - 50 - (self.data_points[i][1] - self.precision_range[0]) * (height - 100) / (self.precision_range[1] - self.precision_range[0]))
            x2 = int(50 + (self.data_points[i + 1][0] - self.epoch_range[0]) * (width - 100) / (self.epoch_range[1] - self.epoch_range[0]))
            y2 = int(height - 50 - (self.data_points[i + 1][1] - self.precision_range[0]) * (height - 100) / (self.precision_range[1] - self.precision_range[0]))
            painter.drawLine(x1, y1, x2, y2)

        # Draw points
        for point in self.data_points:
            x = int(50 + (point[0] - self.epoch_range[0]) * (width - 100) / (self.epoch_range[1] - self.epoch_range[0]))
            y = int(height - 50 - (point[1] - self.precision_range[0]) * (height - 100) / (self.precision_range[1] - self.precision_range[0]))
            painter.drawEllipse(x - 1, y - 1, 2, 2)  # Make the points smaller


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DynamicLineChart(epoch_range=(0, 100), precision_range=(0.00, 1.00))
    window.add_data_point(1, 0.56)
    window.add_data_point(2, 0.66)
    window.show()
    sys.exit(app.exec_())
