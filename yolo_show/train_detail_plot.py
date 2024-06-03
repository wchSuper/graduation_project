import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsRectItem, QGraphicsTextItem
from PyQt5.QtGui import QBrush, QPen, QPainterPath
from PyQt5.QtCore import Qt, QPointF

class LayerItem(QGraphicsRectItem):
    def __init__(self, name, info, x, y, width=100, height=50):
        super().__init__(x, y, width, height)
        self.name = name  # Add the name attribute here
        self.setBrush(QBrush(Qt.lightGray))
        self.setPen(QPen(Qt.black))
        self.label = QGraphicsTextItem(name + '\n' + info, self)
        self.label.setPos(x + 5, y + 5)

class BackboneScene(QGraphicsScene):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.items = []  # To keep track of items for connections
        self.draw_backbone()

    def draw_backbone(self):
        x, y = 0, 0
        for idx, layer in enumerate(self.config['backbone']):
            name = f"Layer {idx}"
            if layer[2] == "Conv":
                info = f"Conv {layer[3][1]}x{layer[3][1]}, {layer[3][2]}"
                item = LayerItem(name, info, x, y)
                self.addItem(item)
                self.items.append(item)
                y += 60  # Move down for the next layer
                x += 20  # Slightly move to the right for better visualization
            elif layer[2] == "MP":
                info = "Max Pool"
                item = LayerItem(name, info, x, y)
                self.addItem(item)
                self.items.append(item)
                y += 80  # Add extra space for Max Pooling layers
                x += 20
            elif layer[2] == "Concat":
                info = "Concat"
                # Offset the concat module
                concat_x = x + 100
                concat_y = y - 40
                item = LayerItem(name, info, concat_x, concat_y, width=100, height=30)
                self.addItem(item)
                self.items.append(item)
                self.draw_concat_lines(layer[0], item)
                y += 60  # Skip a line to account for the concat
                x += 20

        # Draw connections between consecutive layers
        for i in range(1, len(self.items)):
            if self.items[i].name.startswith("Layer") and self.items[i-1].name.startswith("Layer"):
                self.draw_connection(self.items[i-1], self.items[i])

    def draw_concat_lines(self, indices, concat_item):
        for index in indices:
            source_item = self.items[index]
            source_center = source_item.boundingRect().center() + source_item.pos()
            concat_center = concat_item.boundingRect().center() + concat_item.pos()
            self.draw_curve(source_center, concat_center, QPen(Qt.red))

    def draw_connection(self, from_item, to_item):
        from_center = from_item.boundingRect().center() + from_item.pos()
        to_center = to_item.boundingRect().center() + to_item.pos()
        self.draw_curve(from_center, to_center, QPen(Qt.blue))

    def draw_curve(self, start_point, end_point, pen):
        path = QPainterPath()
        path.moveTo(start_point)
        control_point1 = QPointF((start_point.x() + end_point.x()) / 2, start_point.y())
        control_point2 = QPointF((start_point.x() + end_point.x()) / 2, end_point.y())
        path.cubicTo(control_point1, control_point2, end_point)
        self.addPath(path, pen)

class MainWindow(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.setWindowTitle("YOLOv7 Backbone Visualization")
        self.setGeometry(100, 100, 1200, 800)

        self.scene = BackboneScene(config)
        self.view = QGraphicsView(self.scene, self)
        self.setCentralWidget(self.view)

config = {
    "backbone": [
        [-1, 1, "Conv", [32, 3, 2, None, 1, "nn.LeakyReLU(0.1)"]],
        [-1, 1, "Conv", [64, 3, 2, None, 1, "nn.LeakyReLU(0.1)"]],
        [-1, 1, "Conv", [32, 1, 1, None, 1, "nn.LeakyReLU(0.1)"]],
        [-2, 1, "Conv", [32, 1, 1, None, 1, "nn.LeakyReLU(0.1)"]],
        [-1, 1, "Conv", [32, 3, 1, None, 1, "nn.LeakyReLU(0.1)"]],
        [-1, 1, "Conv", [32, 3, 1, None, 1, "nn.LeakyReLU(0.1)"]],
        [[-1, -2, -3, -4], 1, "Concat", [1]],
        [-1, 1, "Conv", [64, 1, 1, None, 1, "nn.LeakyReLU(0.1)"]],
        [-1, 1, "MP", []],
        [-1, 1, "Conv", [64, 1, 1, None, 1, "nn.LeakyReLU(0.1)"]],
        [-2, 1, "Conv", [64, 1, 1, None, 1, "nn.LeakyReLU(0.1)"]],
        [-1, 1, "Conv", [64, 3, 1, None, 1, "nn.LeakyReLU(0.1)"]],
        [-1, 1, "Conv", [64, 3, 1, None, 1, "nn.LeakyReLU(0.1)"]],
        [[-1, -2, -3, -4], 1, "Concat", [1]],
        [-1, 1, "Conv", [128, 1, 1, None, 1, "nn.LeakyReLU(0.1)"]],
        [-1, 1, "MP", []],
        [-1, 1, "Conv", [128, 1, 1, None, 1, "nn.LeakyReLU(0.1)"]],
        [-2, 1, "Conv", [128, 1, 1, None, 1, "nn.LeakyReLU(0.1)"]],
        [-1, 1, "Conv", [128, 3, 1, None, 1, "nn.LeakyReLU(0.1)"]],
        [-1, 1, "Conv", [128, 3, 1, None, 1, "nn.LeakyReLU(0.1)"]],
        [[-1, -2, -3, -4], 1, "Concat", [1]],
        [-1, 1, "Conv", [256, 1, 1, None, 1, "nn.LeakyReLU(0.1)"]],
        [-1, 1, "MP", []],
        [-1, 1, "Conv", [256, 1, 1, None, 1, "nn.LeakyReLU(0.1)"]],
        [-2, 1, "Conv", [256, 1, 1, None, 1, "nn.LeakyReLU(0.1)"]],
        [-1, 1, "Conv", [256, 3, 1, None, 1, "nn.LeakyReLU(0.1)"]],
        [-1, 1, "Conv", [256, 3, 1, None, 1, "nn.LeakyReLU(0.1)"]],
        [[-1, -2, -3, -4], 1, "Concat", [1]],
        [-1, 1, "Conv", [512, 1, 1, None, 1, "nn.LeakyReLU(0.1)"]]
    ]
}

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow(config)
    window.show()
    sys.exit(app.exec_())
