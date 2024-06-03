import sys
import random
import math
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QPainterPath, QLinearGradient
from PyQt5.QtCore import Qt, QRectF, QTimer, QPoint, QSize

class HeartWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Heart')
        self.setGeometry(100, 100, 800, 800)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.t = 0
        self.dragging = False
        self.drag_start_position = QPoint(0, 0)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateAnimation)
        self.timer.start(50)

        self.text = "高斐斐是大猪！"
        self.displayed_text = ""
        self.text_timer = QTimer(self)
        self.text_timer.timeout.connect(self.showNextCharacter)

        self.hearts = []
        self.heart_timer = QTimer(self)
        self.heart_timer.timeout.connect(self.addHeart)

    def updateAnimation(self):
        self.t += 0.1
        if self.t > 2 * 3.14159:
            self.t = 0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw the heart
        painter.setBrush(QBrush(Qt.red, Qt.SolidPattern))
        painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))

        path = self.createHeartPath()
        painter.drawPath(path)

        painter.setPen(QPen(Qt.white, 2, Qt.SolidLine))
        painter.setFont(QFont("Arial", 30, QFont.Bold))
        painter.drawText(self.rect(), Qt.AlignCenter, "520")

        gradient = QLinearGradient(0, 0, 400, 0)
        gradient.setColorAt(0.0, Qt.blue)
        gradient.setColorAt(1.0, Qt.green)

        painter.setPen(QPen(gradient, 2, Qt.SolidLine))
        painter.setFont(QFont("Arial", 20, QFont.Bold))
        painter.drawText(QRectF(50, 400, 200, 50), Qt.AlignCenter, "王辰浩")
        painter.drawText(QRectF(550, 400, 200, 50), Qt.AlignCenter, "高斐斐")

        # Draw the text above the heart
        if self.displayed_text:
            text_gradient = QLinearGradient(0, 0, self.width(), 0)
            text_gradient.setColorAt(0.0, Qt.red)
            text_gradient.setColorAt(1.0, Qt.blue)

            painter.setPen(QPen(text_gradient, 0))
            painter.setFont(QFont("Arial", 40, QFont.Bold))
            text_rect = QRectF(0, 150, self.width(), 100)
            painter.drawText(text_rect, Qt.AlignCenter, self.displayed_text)

        # Draw small hearts
        for heart in self.hearts:
            painter.setBrush(QBrush(heart['color'], Qt.SolidPattern))
            painter.setPen(Qt.NoPen)
            heart_path = self.createSmallHeartPath(heart['x'], heart['y'], heart['size'])
            painter.drawPath(heart_path)

    def createHeartPath(self):
        path = QPainterPath()
        centerX, centerY = 400, 400
        size = 150 + 30 * abs(math.sin(self.t))

        path.moveTo(centerX, centerY - size / 4)
        path.cubicTo(centerX + size, centerY - size, centerX + size, centerY + size, centerX, centerY + size)
        path.cubicTo(centerX - size, centerY + size, centerX - size, centerY - size, centerX, centerY - size / 4)
        return path

    def createSmallHeartPath(self, x, y, size):
        path = QPainterPath()
        path.moveTo(x, y - size / 4)
        path.cubicTo(x + size, y - size, x + size, y + size, x, y + size)
        path.cubicTo(x - size, y + size, x - size, y - size, x, y - size / 4)
        return path

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_start_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self.dragging:
            self.move(event.globalPos() - self.drag_start_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
            event.accept()

    def mouseDoubleClickEvent(self, event):
        self.displayed_text = ""
        self.text_timer.start(300)  # Start showing text one character at a time
        self.hearts = []  # Clear any existing hearts
        self.heart_timer.start(100)  # Start adding small hearts

    def showNextCharacter(self):
        if len(self.displayed_text) < len(self.text):
            self.displayed_text += self.text[len(self.displayed_text)]
            self.update()
        else:
            self.text_timer.stop()  # Stop the timer once all characters are displayed

    def addHeart(self):
        if len(self.displayed_text) < len(self.text):
            return  # Wait until the text is fully displayed

        x = random.randint(0, self.width())
        y = random.randint(0, self.height())
        size = random.randint(10, 30)
        color = QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.hearts.append({'x': x, 'y': y, 'size': size, 'color': color})

        if len(self.hearts) > 100:  # Limit the number of hearts
            self.hearts.pop(0)
        self.update()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Love Confession')
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        self.label = QLabel('Click the button to reveal the heart')
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.button = QPushButton('Show Heart')
        self.button.clicked.connect(self.showHeart)
        layout.addWidget(self.button)

        self.setLayout(layout)

    def showHeart(self):
        self.heartWindow = HeartWidget()
        self.heartWindow.show()

    def closeEvent(self, event):
        if hasattr(self, 'heartWindow') and self.heartWindow is not None:
            self.heartWindow.close()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
