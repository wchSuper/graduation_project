import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QPainter, QPen, QBrush, QFont, QColor, QPolygonF, QRadialGradient
from PyQt5.QtCore import Qt, QPoint, QRectF, QPointF
import math


class CircularGauge(QWidget):
    def __init__(self, param_range=(0.40, 1.00), current_value=0.56, chart_name="chart name"):
        super(CircularGauge, self).__init__()

        self.chart_name = chart_name
        self.chart_name_label = QLabel(self.chart_name)
        self.outer_layout = QVBoxLayout(self)
        self.chart_widget = CircularGauge2(param_range, current_value)

        self.chart_name_label.setFont(QFont("Arial", 14))
        self.chart_name_label.setMaximumHeight(20)
        self.chart_name_label.setAlignment(Qt.AlignCenter)
        self.outer_layout.setContentsMargins(0, 0, 0, 0)

        self.outer_layout.addWidget(self.chart_name_label)
        self.outer_layout.addWidget(self.chart_widget)

        self.resize(200, 200)

    def setCurrentValue(self, val):
        self.chart_widget.setCurrentValue(val)
        self.chart_name_label.setText(f"{self.chart_name} ({val})")
        self.update()


class CircularGauge2(QWidget):
    def __init__(self, param_range=(0.40, 1.00), current_value=0.56):
        super().__init__()
        self.setWindowTitle('Circular Gauge')
        self.setGeometry(100, 100, 400, 400)

        self.param_range = param_range
        self.current_value = current_value

        self.setStyleSheet("background-color: white; border-radius: 10px")

    def setCurrentValue(self, val):
        self.current_value = val
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        self.drawGauge(painter)
        self.drawPointer(painter)

    def drawGauge(self, painter):
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        center = QPoint(width // 2, height // 2)
        radius = min(width, height) // 3

        # Draw gradient circle
        gradient = QRadialGradient(center, radius * 1.1)
        gradient.setColorAt(0, QColor(255, 255, 255))
        gradient.setColorAt(1, QColor(200, 200, 200))
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, radius, radius)

        # Draw circle border
        pen = QPen(Qt.black, 2, Qt.SolidLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(center, radius, radius)

        # Draw parameter labels around the circle
        font = QFont("Arial", 10, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QPen(Qt.black))

        num_intervals = 10  # Number of intervals (10 steps)
        for i in range(num_intervals + 1):
            value = self.param_range[0] + i * (self.param_range[1] - self.param_range[0]) / num_intervals
            angle = 360 * (i / num_intervals)
            radian = math.radians(angle - 90)
            x = center.x() + radius * 1.15 * math.cos(radian)
            y = center.y() + radius * 1.15 * math.sin(radian)

            if i == 0:
                text = f'{self.param_range[0]:.2f}({self.param_range[1]:.2f})'
                # Move the text slightly to avoid overlap
                painter.drawText(QRectF(x - 20, y - 10, 60, 20), Qt.AlignCenter, text)
            elif i == num_intervals:
                continue  # Skip the last label
            else:
                text = f'{value:.2f}'
                # Draw shadow
                shadow_offset = 1
                painter.setPen(QPen(Qt.gray))
                painter.drawText(QRectF(x - 10 + shadow_offset, y - 10 + shadow_offset, 30, 20), Qt.AlignCenter, text)

                # Draw text
                painter.setPen(QPen(Qt.black))
                painter.drawText(QRectF(x - 10, y - 10, 30, 20), Qt.AlignCenter, text)

    def drawPointer(self, painter):
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(Qt.red, 2, Qt.SolidLine)
        painter.setPen(pen)
        brush = QBrush(Qt.red)
        painter.setBrush(brush)

        width = self.width()
        height = self.height()
        center = QPoint(width // 2, height // 2)
        radius = min(width, height) // 3

        # Calculate the angle of the pointer
        value_percentage = (self.current_value - self.param_range[0]) / (self.param_range[1] - self.param_range[0])
        angle = 360 * value_percentage
        radian = math.radians(angle - 90)

        # Draw pointer
        pointer_length = radius * 0.9
        pointer_base = radius * 0.1
        x = center.x() + pointer_length * math.cos(radian)
        y = center.y() + pointer_length * math.sin(radian)

        # Create the pointer polygon (arrow)
        pointer_head = QPointF(x, y)
        pointer_left = QPointF(
            center.x() + pointer_base * math.cos(radian + math.pi / 2),
            center.y() + pointer_base * math.sin(radian + math.pi / 2)
        )
        pointer_right = QPointF(
            center.x() + pointer_base * math.cos(radian - math.pi / 2),
            center.y() + pointer_base * math.sin(radian - math.pi / 2)
        )

        pointer_polygon = QPolygonF([pointer_left, pointer_head, pointer_right])
        painter.drawPolygon(pointer_polygon)

        # Draw center circle
        center_circle_radius = 5
        painter.setBrush(QBrush(Qt.black))
        painter.drawEllipse(center, center_circle_radius, center_circle_radius)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    result_widget = CircularGauge((0.40, 1.00),
                                  0.80, "chart_name")
    result_widget.show()
    sys.exit(app.exec_())
