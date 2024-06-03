video_slider_style = """
QSlider::groove:horizontal {
    border: 1px solid #999999;
    height: 10px; /* 设置进度条的高度 */
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #B1B1B1, stop:1 #c4c4c4);
    margin: 2px 0;
}

QSlider::handle:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #616161, stop:1 #282828);
    border: 1px solid #5c5c5c;
    width: 10px; /* 设置滑块的宽度 */
    margin: -2px 0; /* 设置滑块与进度条的间距 */
    border-radius: 5px; /* 设置滑块的圆角 */
}
"""

# res_list_widget_style = """
# QListWidget {
#     background-color: lightblue;
#     border: 2px solid darkblue;
#     border-radius: 5px;
# }
#
# QListWidget::Item {
#     color: darkblue;
#     padding: 5px;
# }
#
# QListWidget::Item:selected {
#     background-color: blue;
#     color: white;
# }
# """

res_list_widget_style = """

QListView {
    outline:none;
	border-radius: 5px;
}

QListWidget::item {
    background-color: #ffffff;
    color: #000000;
    border: transparent;
    border-bottom: 1px solid #dbdbdb;
    padding: 8px;
}
 
QListWidget::item:hover {
    background-color: #e5e5e5;
}
 
QListWidget::item:selected {
    border-left: 5px solid #244865;
}

"""

combo_box_style = """
/* 未下拉时，QComboBox的样式 */
QComboBox {
    border: 1px solid gray;   /* 边框 */
    border-radius: 3px;   /* 圆角 */
    padding: 1px 18px 1px 3px;   /* 字体填衬 */
    color: #000;
    font: normal normal 15px "Microsoft YaHei";
    background: transparent;
}

/* 下拉后，整个下拉窗体样式 */
QComboBox QAbstractItemView {
    outline: 0px solid gray;   /* 选定项的虚框 */
    border: 1px solid yellow;   /* 整个下拉窗体的边框 */
    background-color: white;   /* 整个下拉窗体的背景色 */
    // selection-background-color: #FDF5E6;   /* 整个下拉窗体被选中项的背景色 */
}

"""

spin_box_style = """

QDoubleSpinBox {
    font-size: 18px; /* 设置内部字体大小 */
    height: 25px; /* 设置高度 */
    padding: 2px; /* 设置内边距 */
    border: 1px solid black; /* 设置边框 */
    border-radius: 5px; /* 设置边框圆角 */
}

"""

tree_scroll = """
    QTreeView {
        background-color: lightblue; 
        border-radius: 10px;
        width: 10px;
    }
    QTreeView::scrollbar {
                width: 15px;
                background-color: #f0f0f0;
                border: 1px solid #d0d0d0;
                margin: 0px 0px 0px 0px;
            }
            QTreeView::scrollbar-thumb {
                background-color: #c3c3c3;
                border-radius: 7px;
            }
            QTreeView::scrollbar-button {
                background-color: #d0d0d0;
            }

"""

# label的各种style
label_style1 = """
QLabel {
    color: blue;
    font-size: 18px;
    font-weight: bold;

}

"""

label_style2 = """
QLabel {
    color: black;
    font-size: 22px;
    font-weight: bold;

}

"""

label_style3 = """
QLabel {
    color: black;
    font-size: 15px;
    font-weight: bold;

}

"""



shishi_btn_style = """
QPushButton {
    background: transparent;
}

QPushButton:hover{
    background-image: url("")
}
QPushButton:pressed{
    background-image: url("")
}


"""
picture_btn_style = """
QPushButton {
    background: transparent;
}

QPushButton:hover{
    background-image: url("../icons/picture_hover.png");
}

"""

video_btn_style = """
QPushButton {
    background: transparent;
}

QPushButton:hover{
    background-image: url("../icons/vid_hover.png");
}

"""