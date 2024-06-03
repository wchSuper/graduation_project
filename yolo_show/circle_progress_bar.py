import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import math


def Animation(parent, type=b"windowOpacity", from_value=0, to_value=1, ms=1000, connect=None):
    anim = QPropertyAnimation(parent, type)
    anim.setDuration(ms)
    anim.setStartValue(from_value)
    anim.setEndValue(to_value)
    if connect:
        anim.finished.connect(connect)
    anim.start()
    return anim


class ProgressThread(QThread):
    signal = pyqtSignal()

    def run(self):
        for p in range(100):
            self.signal.emit()
            self.msleep(100)


class RoundProgress(QWidget):
    m_waterOffset = 0.05
    m_offset = 50
    bg_color = QColor(255, 0, 0)
    fsize = 10

    def __init__(self, t, parent=None):
        super(RoundProgress, self).__init__(parent)
        self.resize(*t)
        self.size = t
        self.setWindowFlags(Qt.FramelessWindowHint)  # 去边框
        self.setAttribute(Qt.WA_TranslucentBackground)  # 设置窗口背景透明
        self.percent = 0
        self.pen = QPen()
        gradient = QConicalGradient(50, 50, 91)
        gradient.setColorAt(0, QColor(255, 10, 10))
        gradient.setColorAt(1, QColor(255, 201 ,14))
        # gradient.setColorAt(0.5, QColor(255, 201 ,14))
        self.pen.setBrush(gradient)  # 设置画刷渐变效果
        self.pen.setWidth(8)
        self.pen.setCapStyle(Qt.RoundCap)
        self.font = QFont()
        self.font.setFamily("Share-TechMono")
        self.font.setPointSize(self.size[0] // 4)

    def paintEvent(self, event):
        width, height = self.size
        rect = QRectF(self.fsize, self.fsize, width-self.fsize*2, height-self.fsize*2)
        painter = QPainter(self)
        rotateAngle = 360 * self.percent / 100
        # 绘制准备工作，启用反锯齿
        painter.setRenderHints(QPainter.Antialiasing)
        painter.setPen(self.pen)

        painter.drawArc(rect, (90 - 0) * 16, -rotateAngle * 16)  # 画圆环
        painter.setFont(self.font)
        painter.setPen(QColor(153 - 1.53*self.percent ,
                              217-0.55*self.percent,
                              234-0.02*self.percent)) # r:255, g:201 - 10/100 * percent, b: 14-4 /100*percent 当前渐变
        painter.drawText(rect, Qt.AlignCenter, "%d%%" % self.percent) # 显示进度条当前进度
        self.update()

    def update_percent(self, p):
        self.percent = p


class WaterProgress(QWidget):
    fsize = 10

    def __init__(self,t,parent=None):
        super(WaterProgress, self).__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint)  # 去边框
        self.setAttribute(Qt.WA_TranslucentBackground)  # 设置窗口背景透明
        self.resize(*t)
        self.size = t
        self.layout = QGridLayout(self)

        #设置进度条颜色
        self.bg_color = QColor("#95BBFF")
        self.m_waterOffset = 0.005
        self.m_offset = 50
        self.m_borderwidth = 10
        self.percent = 0

    def paintEvent(self, event):
        painter = QPainter()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.begin(self)
        painter.setPen(Qt.NoPen)

        #获取窗口的宽度和高度
        width,height = self.size
        percentage = 1 - self.percent/100
        # 水波走向：正弦函数 y = A(wx+l) + k
        # w 表示 周期，值越大密度越大
        w = 2 * math.pi / (width)
        # A 表示振幅 ，理解为水波的上下振幅
        A = height * self.m_waterOffset
        # k 表示 y 的偏移量，可理解为进度
        k = height *percentage

        water1 = QPainterPath()
        water2 = QPainterPath()

        #起始点
        water1.moveTo(5,height)
        water2.moveTo(5,height)
        self.m_offset += 0.6

        if self.m_offset > (width/2):
            self.m_offset = 0
        i = 5

        rect = QRectF(self.fsize,self.fsize,width - self.fsize*2, height - self.fsize * 2)
        while i < width-5:
            waterY1 = A*math.sin(w*i +self.m_offset ) + k
            waterY2 = A*math.sin(w*i + self.m_offset + width/2*w) + k

            water1.lineTo(i, waterY1)
            water2.lineTo(i, waterY2)
            i += 1

        water1.lineTo(width-5,height)
        water2.lineTo(width-5,height)

        totalpath = QPainterPath()
        #totalpath.addRect(rect)
        #painter.setBrush(Qt.gray)
        painter.drawRect(self.rect())
        painter.save()
        totalpath.addEllipse(rect)
        totalpath.intersected(water1)
        painter.setPen(Qt.NoPen)

        #设置水波的透明度
        watercolor1 =QColor(self.bg_color)
        watercolor1.setAlpha(100)
        watercolor2 = QColor(self.bg_color)
        watercolor2.setAlpha(150)
        path = totalpath.intersected(water1)
        painter.setBrush(watercolor1)
        painter.drawPath(path)


        path = totalpath.intersected(water2)
        painter.setBrush(watercolor2)
        painter.drawPath(path)
        painter.restore()
        painter.end()
        #self.update()

    def update_percent(self, p):
        self.percent = p
        if self.m_waterOffset < 0.05:
            self.m_waterOffset += 0.001
        return p


class Progress(QDialog):
    percent = 0

    def __init__(self, text="",parent=None):
        super(Progress, self).__init__(parent)
        Font = QFont()
        Font.setFamily("Consolas")
        Font.setPointSize(12)
        self.setFont(Font)
        self.setWindowFlags(Qt.FramelessWindowHint)  # 去边框
        self.setAttribute(Qt.WA_TranslucentBackground)  # 设置窗口背景透明
        self.ProgressThread = ProgressThread()
        self.ProgressThread.signal.connect(self.percentUpdate)
        self.ProgressThread.start()
        width, height = 230, 230
        self.resize(width, height)
        self.water = WaterProgress((width, height), self)
        self.round = RoundProgress((width, height), self)
        self.label = QLabel(self)
        self.label.setText(QCoreApplication.translate("Dialog", text))
        print(self.label.width())
        self.label.move((width-self.label.width())/2, height/3*2)
        QMetaObject.connectSlotsByName(self)
        self.anim = Animation(self, )

    def connect(self, link):
        print(type(link))
        #self.setWindowFlags(Qt.WindowStaysOnTopHint) #置顶
        self.show()

    def percentUpdate(self):
        self.percent += 1
        self.water.update_percent(self.percent)
        self.water.paintEvent(None)
        self.round.update_percent(self.percent)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    r = Progress("<a link='b'>write something</a>")
    r.show()
    sys.exit(app.exec_())
