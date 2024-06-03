import argparse
import os.path
import sys
import time

from camera_detect import *
from models.yolo import *
from util.general import *
from util.torch_utils import *


class Example(QWidget):

    def __init__(self):
        super().__init__()
        # 启动鼠标追踪
        self.setMouseTracking(True)

        # 标题
        self.setWindowTitle("目标检测学习")
        self.setWindowIcon(QIcon(TITLE_ICON))
        self.resize(800, 600)

        # 最外层 VerticalLayout
        self.widget1 = QWidget()
        self.widget2 = QWidget()

        # 内部水平布局1  水平布局2  widget1, widget2
        # self.second_layout1 = QHBoxLayout()
        self.second_layout2 = QHBoxLayout()
        self.first_layout = QVBoxLayout()

        # 图片 或 视频是否在label中显示出来
        self.is_img_show = False
        self.is_vid_show = False
        # label 是否被占用
        self.is_label_used = False

        # 布局1 中创建五个按钮
        self.btn_num = BTN_NUM
        self.btn_w = 50
        self.btn_h = 50
        self.top_btn_list = []
        for num in range(self.btn_num):
            tmp_btn = QPushButton(self.widget1)
            tmp_btn.setGeometry(10 + num * (self.btn_w + 10), 10, self.btn_w, self.btn_h)
            # 取消按钮边框
            tmp_btn.setStyleSheet("background: transparent;")
            self.top_btn_list.append(tmp_btn)

        # 按钮[5]是否实时
        self.is_shishi = False

        # 从第六个按钮开始
        self.btn_extra_num = BTN_EXTRA_NUM
        self.top_extra_btn_list = []
        for num2 in range(self.btn_num, self.btn_extra_num + self.btn_num):
            tmp_btn1 = QPushButton(self.widget1)
            tmp_btn1.setGeometry(30 + num2 * (self.btn_w + 10), 10, self.btn_w, self.btn_h)
            tmp_btn1.setStyleSheet("background: transparent;")
            self.top_extra_btn_list.append(tmp_btn1)

        # 最后一个按钮
        self.stop_actions_btn = QPushButton(self.widget1)
        self.stop_actions_btn.setGeometry(self.x() + self.width() - 80, 10, self.btn_w,
                                          self.btn_h)
        self.stop_actions_btn.setStyleSheet("background: transparent;")
        # 是否warm up
        self.is_warm = True

        # 布局2中创建  1个自定义TreeView 1个show_widget 1个data_widget
        self.file_view = MyTreeView()
        self.show_widget = ShowWidget()
        self.image_label = self.show_widget.show_label
        self.data_widget = DataWidget()

        # ========== 初始化方法 ============ #
        self.initUI()
        self.setAttr()

        # 视频捕获对象 cam_num = 0 是摄像头
        self.cam_num = 0
        self.cap = cv2.VideoCapture()
        self.cap2 = cv2.VideoCapture()

        # 视频流的总frames 和 当前frames
        self.total_frames = 0
        self.current_frames = 0

        # 视频的帧率
        self.fps = 24

        # 定时器模块 用于定期获取视频帧  写两个防止混乱
        self.video_timer = QTimer()
        self.video_timer2 = QTimer()
        # 定时器模块 用于将检测后的视频 逐帧播放
        self.video_timer3 = QTimer()

        # 当前指示的文件path
        self.current_file_path = ""

        # iou conf weight coco类别的值
        self.iou_value = 0
        self.conf_value = 0
        self.weight = ""
        self.coco_cls = {}
        # 工作检测线程  iou, conf, weight, img or vid, path, yolo对象
        self.is_vid = 0
        self.yolo7 = YOLOv7()
        self.detect_thread = DetectThread([0, 0, "", {}, {}, self.is_vid, "", self.yolo7, True, ""])
        self.warm_thread = DetectThread([0, 0, "", {}, {}, self.is_vid, "", self.yolo7, True, ""])
        # 检测所耗费的时间 ms
        self.before_t = 0
        self.after_t = 0

        # 处理后的结果目录 返回的原始图像尺寸
        self.result_dir = ""
        self.ret_w, self.ret_h = 640, 480

        # sub window
        self.sub_window = SubVideoWindow()

        # 出场gif
        self.gif1 = QMovie(GIF1)
        self.is_gif1_on = False
        self.gif2 = QMovie(WARM_GIF)

        # camera打开状态
        self.camera_open = False

        self.camera_detect_thread = CameraDetectThread([0.25, 0.45, "yolov7-tiny.pt"])

        # 是否第一次检测 以及 非第一次训练保存的模型
        self.is_first_detected = True
        self.saved_model = ""

        # 检测锁 检测过程中无法进行其余操作
        self.mutex = 0

        # 信号槽 放在最后
        self.signalSlots()

    def initUI(self):
        self.setAttr()

        # 内部layout添加widget
        self.second_layout2.addWidget(self.file_view)
        self.second_layout2.addWidget(self.show_widget)
        self.second_layout2.addWidget(self.data_widget)

        # 给widget2设置水平layout
        self.widget2.setLayout(self.second_layout2)

        # 最外层layout添加widget
        self.first_layout.addWidget(self.widget1)
        self.first_layout.addWidget(self.widget2)

        # 主界面添加layout
        self.setLayout(self.first_layout)

    def setAttr(self):

        self.widget1.setMinimumHeight(80)
        self.widget1.setMaximumHeight(80)
        self.widget1.setStyleSheet("background-color: white; border-radius: 10px")

        # self.widget1.setStyleSheet("background-image: url('./icons/top_bg.png'); background-position: center; "
        #                          "background-repeat: no-repeat;")
        self.first_layout.setContentsMargins(5, 5, 5, 5)
        self.first_layout.setSpacing(0)

        self.second_layout2.setContentsMargins(0, 5, 0, 5)

        self.widget2.setStyleSheet("background-color: rgb(234,232,231);")

        self.image_label.setMinimumWidth(250)

        self.file_view.setMinimumWidth(150)
        self.file_view.setMaximumWidth(250)

        # Label显示录像或图片
        self.image_label.setStyleSheet("background: black; border-radius:10px;")

        # 给按钮加上图片
        self.top_btn_list[3].setIcon(QIcon("./icons/camera_close_icon.png"))
        self.top_btn_list[3].setIconSize(QSize(self.btn_w, self.btn_h))

        self.top_btn_list[0].setIcon(QIcon("./icons/left.png"))
        self.top_btn_list[0].setIconSize(QSize(self.btn_w, self.btn_h))

        self.top_btn_list[1].setIcon(QIcon("./icons/right.png"))
        self.top_btn_list[1].setIconSize(QSize(self.btn_w, self.btn_h))

        self.top_btn_list[2].setIcon(QIcon("./icons/video_icon.png"))
        self.top_btn_list[2].setIconSize(QSize(self.btn_w, self.btn_h))
        self.top_btn_list[2].setStyleSheet(video_btn_style)

        self.top_btn_list[4].setIcon(QIcon("./icons/picture_icon.png"))
        self.top_btn_list[4].setIconSize(QSize(self.btn_w, self.btn_h))
        self.top_btn_list[4].setStyleSheet(picture_btn_style)

        self.top_btn_list[5].setIcon(QIcon(SHISHI_DETECT_OFF))
        self.top_btn_list[5].setIconSize(QSize(self.btn_w, self.btn_h))
        # self.top_btn_list[5].setStyleSheet()

        # ------------------------- top extra button ----------------------- #
        self.top_extra_btn_list[0].setIcon(QIcon(WARM_OFF_ICON))
        self.top_extra_btn_list[0].setIconSize(QSize(self.btn_w, self.btn_h))

        # ------------------------- the last button ----- stop all detections ----#
        self.stop_actions_btn.setIcon(QIcon(STOP_ACTIONS_ICON))
        self.stop_actions_btn.setIconSize(QSize(self.btn_w, self.btn_h))

        # ========================== move ================================== #
        self.move(0, 0)

    # 根据窗口变化动态改变框框的大小
    def resizeEvent(self, event):
        self.image_label.setMinimumWidth(int(self.width() * 2 / 3))
        self.stop_actions_btn.setGeometry(self.x() + self.width() - 80, 10, self.btn_w,
                                          self.btn_h)
        # self.widget1.setStyleSheet("background-image: url('./icons/top_bg.png'); background-position: center; ")

    # 信号和槽
    def signalSlots(self):
        # 第四个按钮是打开摄像头
        self.top_btn_list[3].clicked.connect(self.openCamera)
        # 定时器事件
        self.video_timer.timeout.connect(self.getVideoPic)
        self.video_timer2.timeout.connect(self.getVideoPic2)

        # 文件树双击事件
        self.file_view.doubleClicked.connect(self.handleFileTree)

        # 上一张和下一张
        self.top_btn_list[0].clicked.connect(self.getLastPic)
        self.top_btn_list[1].clicked.connect(self.getNextPic)

        # 视频播放slider值的改变
        self.show_widget.video_proc_slider.valueChanged.connect(self.sliderCtrlVideo)

        # 进行图片检测
        self.top_btn_list[4].clicked.connect(self.detectPicture)

        # 进行视频检测
        self.top_btn_list[2].clicked.connect(self.detectVideo)

        # 是否实时监测
        self.top_btn_list[5].clicked.connect(self.changeShiShiDetect)

        # 进行摄像头实时检测
        self.show_widget.vid_shishi_detect.connect(self.detectCamera)

        # 中止检测
        self.stop_actions_btn.clicked.connect(self.stopDetection)

        # 接收---视频---是否打开信号
        self.show_widget.vid_stopped_signal.connect(self._stop_video)
        self.show_widget.vid_started_signal.connect(self._start_video)

        # 双击QListWidget item高亮box
        self.data_widget.res_list_widget.itemDoubleClicked.connect(self.on_item_double_clicked)

        # 实时改变iou 和 conf的阈值 for NMS
        self.data_widget.on_iou_conf_changed.connect(self.on_iou_conf_changed)

    # -------------------------- 重写paintEvent --------------------------------------------- #
    def paintEvent(self, event):
        pass

    # 打开摄像头 先判断是否处于打开状态
    def openCamera(self):
        if self.video_timer.isActive():
            self.cap.release()
            self.video_timer.stop()
            self.top_btn_list[3].setIcon(QIcon("./icons/camera_close_icon.png"))
            self.top_btn_list[3].setIconSize(QSize(self.btn_w, self.btn_h))
            self.camera_open = False
            return

        self.top_btn_list[3].setIcon(QIcon("./icons/camera_open_icon.png"))
        self.top_btn_list[3].setIconSize(QSize(self.btn_w, self.btn_h))
        self.update()
        # 是否能打开
        is_open = self.cap.open(self.cam_num)
        if not is_open:
            QMessageBox.information(self, "INFO", "该设备未正常连接！", QMessageBox.Ok)
        else:
            self.image_label.setEnabled(True)
            self.video_timer.start(10)
        self.camera_open = True

    # 获取摄像头捕获的视频帧 在定时器内部
    def getVideoPic(self):
        ret, img = self.cap.read()
        if ret:
            # 把BGR模式转换成RGB
            cur_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 视频流的长和宽
            height, width = cur_frame.shape[:2]
            pixmap = QImage(cur_frame, width, height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(pixmap)
            # 获取是视频流和label窗口的长宽比值的最大值，适应label窗口播放，不然显示不全
            self.image_label.setMinimumWidth(int(self.width() * 2 / 3))
            ratio = max(width / self.image_label.width(), height / self.image_label.height())
            pixmap.setDevicePixelRatio(ratio)
            # 视频流置于label中间部分播放
            # self.image_label.setGraphicsEffect(QGraphicsBlurEffect(QLabel.Antialiasing))
            self.image_label.setAlignment(Qt.AlignCenter)
            self.image_label.setPixmap(pixmap)

    # 获取视频捕获的视频帧 在定时器内部
    def getVideoPic2(self):
        ret, img = self.cap2.read()
        if ret:
            # 获取视频帧率
            self.fps = self.cap2.get(cv2.CAP_PROP_FPS)
            # 获取视频总帧数
            self.total_frames = self.cap2.get(cv2.CAP_PROP_FRAME_COUNT)
            # print(f"视频总帧数: {self.total_frames}")
            self.current_frames = self.cap2.get(cv2.CAP_PROP_POS_FRAMES)

            # 根据当前帧数和总帧数更新slider指示位置
            self.updateSliderPos()
            # 更新播放时间label
            self.updateTimeLabel()

            # 把BGR模式转换成RGB
            cur_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 视频流的长和宽
            height, width = cur_frame.shape[:2]
            pixmap = QImage(cur_frame, width, height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(pixmap)
            # 获取是视频流和label窗口的长宽比值的最大值，适应label窗口播放，不然显示不全
            self.image_label.setMinimumWidth(int(self.width() * 2 / 3))
            ratio = max(width / self.image_label.width(), height / self.image_label.height())
            pixmap.setDevicePixelRatio(ratio)
            # 视频流置于label中间部分播放
            # self.image_label.setGraphicsEffect(QGraphicsBlurEffect(QLabel.Antialiasing))
            self.image_label.setAlignment(Qt.AlignCenter)
            self.image_label.setPixmap(pixmap)

    # 改变是否实时检测
    def changeShiShiDetect(self):
        if not self.is_shishi:
            self.is_shishi = True
            self.top_btn_list[5].setIcon(QIcon(SHISHI_DETECT_ON))
            self.top_btn_list[5].setIconSize(QSize(self.btn_w, self.btn_h))
        else:
            self.is_shishi = False
            self.top_btn_list[5].setIcon(QIcon(SHISHI_DETECT_OFF))
            self.top_btn_list[5].setIconSize(QSize(self.btn_w, self.btn_h))

    # 在算法启动之后 逐帧显示图片 --- 弃用 ---
    def openVideo(self):
        pass

    # 处理文件树双击事件
    def handleFileTree(self):
        index = self.file_view.currentIndex()
        file_path = self.file_view.model01.filePath(index)
        # 记录一下
        self.current_file_path = file_path

        # 根据文件路径和属性更新top labels
        self.updateTopLabels()

        if '.' in file_path:
            if file_path.split('.')[-1] in picture_suffix_list:
                # 首先判断是否在检测过程
                if self.mutex:
                    self.warnMessageBox()
                    return
                # 双击图片文件 先检查定时器是否开着 如果开着则关闭定时器 关闭cap
                if self.video_timer.isActive():
                    self.cap.release()
                    self.video_timer.stop()

                if self.video_timer2.isActive():
                    self.cap2.release()
                    self.video_timer2.stop()

                # 关闭 data widget中上一个listWidget显示的内容
                self.data_widget.res_list_widget.clear()
                self.showPicOrImg(file_path, 0)

            elif file_path.split('.')[-1] in video_suffix_list:
                # 首先判断是否在检测过程
                if self.mutex:
                    self.warnMessageBox()
                    return
                # 双击视频文件 先检查定时器是否开着 如果开着则关闭定时器 关闭cap
                if self.video_timer.isActive():
                    self.cap.release()
                    self.video_timer.stop()

                if self.video_timer2.isActive():
                    self.cap2.release()
                    self.video_timer2.stop()

                if not self.video_timer.isActive() and not self.video_timer2.isActive():
                    # 如果没有定时器启动 则开启定时器
                    is_video_open = self.cap2.open(self.current_file_path)
                    if not is_video_open:
                        QMessageBox.Information(self, "", "", QMessageBox.Ok)
                        return
                    # 双击文件树上的视频文件 直接开始播放
                    self.show_widget.current_play_id = 0
                    self.show_widget.changeVideoPlayBtn()

                    self.video_timer2.start(TIME_S)

    # 根据文件路径 把图片或视频展示self.image_label上    -- 双击展示   #
    # 每次展示的时候 都要将 self.current_file_path替换
    def showPicOrImg(self, file_path, detected):

        file_path = os.fsdecode(file_path).replace("\\", "/")
        pix = QPixmap(file_path)

        # 图片适应Label大小
        pix = pix.scaled(self.image_label.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setPixmap(pix)

        # 此处的路径在检测后无需更改
        self.current_file_path = file_path if detected == 0 else self.current_file_path

        # 根据文件路径和属性更新top labels
        self.updateTopLabels()
        # 检测时间
        delta_t = self.after_t - self.before_t
        self.show_widget.setTimeLabel(f"{delta_t:.2f}")

    def showPicOrImg2(self, pixmap1):
        # 图片适应Label大小
        pixmap = pixmap1.scaled(self.image_label.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setPixmap(pixmap)
        self.update()

    def getLastPic(self):
        ret_path = getLastOrNextFile(0, self.current_file_path)
        if not ret_path in error_file_ret:
            self.showPicOrImg(ret_path, 0)
            self.data_widget.res_list_widget.clear()
        else:
            print("WARNNING! Last no picture")

    def getNextPic(self):
        ret_path = getLastOrNextFile(1, self.current_file_path)
        if not ret_path in error_file_ret:
            self.showPicOrImg(ret_path, 0)
            self.data_widget.res_list_widget.clear()
        else:
            print("WARNNING! Next no picture")

    def updateTopLabels(self):
        para1_str = ""
        if self.current_file_path.split('.')[-1] in picture_suffix_list:
            para1_str = "🖼️image🖼️"
        elif self.current_file_path.split('.')[-1] in video_suffix_list:
            para1_str = "📸video📸"
        else:
            para1_str = "🎥camera🎥"

        self.show_widget.setTopAllLabels(para1_str, self.current_file_path, "⏳ none")

    def sliderCtrlVideo(self, value):
        if self.video_timer2.isActive():
            set_frame_number = int(value / 100 * self.total_frames)
            self.cap2.set(cv2.CAP_PROP_POS_FRAMES, set_frame_number)

    def updateSliderPos(self):
        updated_value = int(self.current_frames / self.total_frames * 100)
        self.show_widget.video_proc_slider.setValue(updated_value)

    def updateTimeLabel(self):
        # -------------- 更新slider右侧的时间进度 ------------ #
        total_secs = int(self.total_frames // self.fps)
        current_secs = int(self.current_frames // self.fps)
        total_time_str = getTimeLabelInfo(total_secs)
        current_time_str = getTimeLabelInfo(current_secs)

        self.show_widget.show_time_label_left.setText(f"{current_time_str}")
        self.show_widget.show_time_label_right.setText(f"{total_time_str}")

    def warnMessageBox(self):
        w = MessageBox("警告", "检测尚未结束，禁止多余操作", self)
        if w.exec_():
            print("确认")
        else:
            print("ok")
        w.cancelButton.hide()
        w.buttonLayout.insertStretch(1)

    def on_iou_conf_changed(self, iou_conf_list):
        iou_value, conf_value = iou_conf_list[0], iou_conf_list[1]
        self.detect_thread.setIoU_Conf_(iou_value, conf_value)

    # ------------------------ 打开 停止 vid ----------------------------------------------------- #
    def _start_video(self):
        # 如果已经是启动状态就无需启动
        if self.video_timer2.isActive():
            pass
        else:
            if "." in self.current_file_path and self.current_file_path.split('.')[-1] in video_suffix_list:
                self.video_timer2.start(TIME_S)

    def _stop_video(self):
        if self.video_timer.isActive():
            self.video_timer.stop()
        if self.video_timer2.isActive():
            self.video_timer2.stop()

    # 高亮box
    def on_item_double_clicked(self, item):
        item_txt = item.text()
        numbers = re.findall(r'\b\d+\.\d+|\b\d+\b', item_txt)
        numbers = [float(x) for x in numbers]
        # 左上x y | 右下 x y
        lx, ly, rx, ry = numbers[1], numbers[2], numbers[3], numbers[4]
        ratio1, ratio2 = 0, 0
        pix_x, pix_y = 0, 0
        if not self.image_label.pixmap() is None:
            pw, ph = self.image_label.pixmap().width(), self.image_label.pixmap().height()
            ratio1 = pw / self.ret_w  # 图片展示和原始的缩放比 ratio1--宽  ratio2--高
            ratio2 = ph / self.ret_h
            # pix_x  pix_y 即pixmap相对于image_label的坐标
            pix_x = (self.image_label.width() - pw) / 2
            pix_y = (self.image_label.height() - ph) / 2
        lx = lx * ratio1
        ly = ly * ratio2
        rx = rx * ratio1
        ry = ry * ratio2
        # 增亮  lx ly   rx ry => 框的放缩坐标
        conf = numbers[0]
        cls = str(item_txt).split(' ')[1]
        self.show_widget.highLightBoxes([lx, ly, rx, ry, pix_x, pix_y, conf, cls])

        # ================================= 检测处理部分 ============================================== #

    # 记录原始图片大小
    def recordBoxes(self, pic_list):
        img_shape = pic_list[0]
        self.ret_w, self.ret_h = img_shape[1], img_shape[0]
        print(self.ret_w, self.ret_h)

    def preWarmUp(self):
        # warm up 处理
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMovie(self.gif2)
        self.gif2.start()
        self.yolo7 = YOLOv7()
        self.warm_thread = DetectThread([45, 25, "yolov7-tiny.pt", {}, coco_names_zh, 0, "./icons/img_1.png", self.yolo7,
                                         self.is_first_detected, self.saved_model])
        self.warm_thread.start()
        self.warm_thread.on_detect_finished.connect(self.warmFinished)

    def detectPicture(self):
        is_suffix_pic = False
        self.is_vid = 0
        if '.' not in self.current_file_path:
            QMessageBox.information(self, "6", "目录无法检测", QMessageBox.Ok)
            return
        else:
            is_suffix_pic = self.current_file_path.split('.')[-1] in picture_suffix_list
            if is_suffix_pic is False:
                QMessageBox.information(self, "6", "文件非图片格式 无法检测！", QMessageBox.Ok)

            else:
                # 启动算法detect 图片过程等待时间用gif过渡 算法结束gif消失
                if self.is_gif1_on:
                    self.gif1.stop()
                self.image_label.setMovie(self.gif1)
                self.gif1.start()
                self.is_gif1_on = True

                # 在主线程处理会卡死
                self.mutex = 1
                self.yolo7 = YOLOv7()
                self.iou_value, self.conf_value, self.weight, self.coco_cls = self.data_widget.getIoU_Conf_WValue()
                self.detect_thread = DetectThread(
                    [self.iou_value, self.conf_value, self.weight, self.coco_cls, coco_names_zh, self.is_vid,
                     self.current_file_path, self.yolo7,
                     self.is_first_detected, self.saved_model])

                # 使用多线程来处理
                self.before_t = time.time()
                self.detect_thread.start()

                self.detect_thread.on_pic_trans.connect(self.recordBoxes)
                self.detect_thread.on_detect_finished.connect(self.on_thread_finished)
                self.detect_thread.on_transfer_model.connect(self.on_get_model)

    def detectVideo(self):
        is_suffix_vid = False
        self.is_vid = 1
        if '.' not in self.current_file_path:
            QMessageBox.information(self, "6", "目录无法检测", QMessageBox.Ok)
            return
        else:
            is_suffix_vid = self.current_file_path.split('.')[-1] in video_suffix_list
            if is_suffix_vid is False:
                QMessageBox.information(self, "6", "文件非视频格式 无法检测！", QMessageBox.Ok)
            else:
                # 清除show_widget中float_label的值
                # if self.show_widget.float_label:
                #     self.show_widget.deleteFloatLabel()
                self.mutex = 1
                self.yolo7 = YOLOv7()
                self.sub_window.show()
                self.iou_value, self.conf_value, self.weight, self.coco_cls = self.data_widget.getIoU_Conf_WValue()
                self.detect_thread = DetectThread(
                    [self.iou_value, self.conf_value, self.weight, self.coco_cls, coco_names_zh,
                     self.is_vid, self.current_file_path, self.yolo7,
                     self.is_first_detected, self.saved_model])

                # 使用多线程来处理
                self.detect_thread.start()
                # 实时处理视频帧
                self.detect_thread.on_vid_trans.connect(self.on_func_vid_frame)
                self.detect_thread.on_detect_finished.connect(self.on_thread_finished)
                self.detect_thread.on_transfer_model.connect(self.on_get_model)

    def detectCamera(self):
        conf_value, iou_value = float(self.conf_value) / 100, float(self.iou_value) / 100
        # self.camera_detect_thread = CameraDetectThread([conf_value, iou_value, self.weight])
        self.sub_window.show()
        self.camera_detect_thread = CameraDetectThread([0.25, 0.45, self.weight])
        self.camera_detect_thread.start()
        # 实时处理帧
        self.camera_detect_thread.on_camera_transfer.connect(self.show_camera_frame)

    # detect结束 获取detect处理的相关信息
    def on_detect_finished(self, detect_dict):
        # print(detect_dict)
        pass

    def on_get_model(self, model_list):
        self.saved_model = model_list[0]

    # 热身结束
    def warmFinished(self):
        self.gif2.stop()

    # 线程结束处理 读取图片等等。。。
    def on_thread_finished(self):
        # ------------- 此处用于调用算法进行处理 --------------------------------- #
        self.is_warm = False
        self.is_first_detected = False
        self.mutex = 0
        if self.is_warm:
            self.is_warm = False
            self.top_extra_btn_list[0].setIcon(QIcon(WARM_UP_ICON))
            self.top_extra_btn_list[0].setIconSize(QSize(self.btn_w, self.btn_h))
            return
        result_dir = get_latest_exp()
        result_path_list = []
        for root, dirs, filenames in os.walk(result_dir):
            for filename in filenames:
                result_path_list.append(os.path.join(root, filename))

        if self.is_vid == 0:
            self.after_t = time.time()
            # self.show_widget.top_label3.setText(f"⏳ {delta_t}")
            # 在listWidget里面显示预测框信息
            for res in result_path_list:
                if not os.path.isdir(res) and res.split('.')[-1] in picture_suffix_list:
                    self.showPicOrImg(res, 1)
                elif not os.path.isdir(res) and res.split('.')[-1] == "txt":
                    # 如果exp下的文件后缀是 "txt" 那么就读取txt的内容 显示到dataWidget中
                    self.data_widget.addParamList(res)

        # if video model --> start timer3
        else:
            pass

    # 实时处理视频帧
    def on_func_vid_frame(self, pix_list):
        # 将视频帧展示出来即可
        pixmap = pix_list[0]
        frame = pix_list[1]
        # 如果检测的视频帧 返回后 将原视频从第一帧开始播放
        if frame == 1:
            self.sliderCtrlVideo(1)
            self.updateSliderPos()
            self.updateTimeLabel()
        self.sub_window.setPicture(pixmap)

    # 实时处理摄像头帧
    def show_camera_frame(self, pix_list):
        pixmap = pix_list[0]
        frame = pix_list[1]
        # 先关闭摄像头定时器
        # if self.video_timer.isActive():
        #     self.cap.release()
        #     self.video_timer.stop()
        self.sub_window.setPicture(pixmap)
        # self.showPicOrImg2(pixmap)
        # 接下来可以记录摄像头检测的时间流逝

    # 主线程发出中止检测请求
    def stopDetection(self):
        self.detect_thread.stop_detection()


# ========================================================================================== #
# ====================================== 线程类 ============================================= #
# ========================================================================================== #
class DetectThread(QThread):
    # 一次检测后的信号
    on_pic_size = pyqtSignal(list)
    on_vid_sent = pyqtSignal(list)
    on_vid_trans = pyqtSignal(list)
    on_pic_trans = pyqtSignal(list)
    on_detect_finished = pyqtSignal()
    # ------------------------ #
    on_get_model = pyqtSignal(list)
    on_transfer_model = pyqtSignal(list)

    def __init__(self, info):
        super(DetectThread, self).__init__()
        self.info_list = info
        self.iou_value = info[0]
        self.conf_value = info[1]
        self.weight = info[2]
        self.coco_cls = info[3]
        self.coco_names = info[4]
        self.img_vid = info[5]
        self.path = info[6]
        self.yolo7 = info[7]
        self.is_first_detected = info[8]
        self.saved_model = info[9]
        self.on_vid_sent.connect(self.vid_accepted)
        self.on_pic_size.connect(self.pic_accepted)
        self.on_get_model.connect(self.transfer_model)

        self.start_yolo = 0

    def run(self):
        # print(self.info_list)
        # 开启算法
        self.start_yolo = 1
        self.yolo7.detect(self.path, self.weight, self.iou_value, self.conf_value, self.coco_cls, self.coco_names,
                          self.on_vid_sent,
                          self.on_pic_size,
                          self.is_first_detected, self.saved_model, self.on_get_model)

        # 每次线程结束一次就发射一次信号
        self.on_detect_finished.emit()

    # 算法遍历视频帧时  每一帧图片调用一次该函数
    def vid_accepted(self, pix_list):
        self.on_vid_trans.emit(pix_list)

    def pic_accepted(self, pic_list):
        self.on_pic_trans.emit(pic_list)

    def transfer_model(self, model_list):
        self.on_transfer_model.emit(model_list)

    # 中止操作
    def stop_detection(self):
        self.yolo7.change_process()

    def setIoU_Conf_(self, iou, conf):
        self.iou_value, self.conf_value = iou, conf
        if self.start_yolo == 1:
            self.yolo7.change_iou_conf(self.iou_value, self.conf_value)

# ================================ 算法类 ========================================================= #
# ================================ =============================================================== #
class YOLOv7(QObject):

    def __init__(self):
        super(YOLOv7, self).__init__()
        self.process = True
        self.iou_value = 0.45
        self.conf_value = 0.25

    def change_process(self):
        self.process = False

    def change_iou_conf(self, new_iou, new_conf):
        self.iou_value = new_iou
        self.conf_value = new_conf

    def detect(self, path, weight, iou_val, conf_val, coco_cls, coco_names_list, vid_sent,
               pic_sent, is_first, s_mdl, on_get_model):
        opt = getOpt()
        """
            source: 输入图片路径  weights: 模型权重  view_img:   save_txt: 是否把结果保存为txt格式
            imgsz: 输入图片大小 default (640)  trace: 是否进行trace model
        """
        source, weights, view_img, save_txt, imgsz, trace = \
            opt.source, opt.weights, opt.view_img, opt.save_txt, \
            opt.img_size, not opt.no_trace
        source = path
        weights = weight
        opt.conf_thres = conf_val / 100
        opt.iou_thres = iou_val / 100
        self.conf_value = conf_val / 100
        self.iou_value = iou_val / 100
        save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        # device = select_device(opt.device)
        device = torch.device('cuda:0')
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = ""
        if is_first or (not is_first and model == ""):
            model = attempt_load(weights, map_location=device)  # load FP32 model
            stride = int(model.stride.max())  # model stride  32
            imgsz = check_img_size(imgsz, s=stride)  # check img_size


            # model = TracedModel(model, device, opt.img_size)

            if half:
                model.half()  # to FP16
            on_get_model.emit([model])
        else:
            model = s_mdl

        # Set DataLoader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        t0 = time.time()
        # img--处理后的图片 %RGB通道  %填充pad        im0s--未处理图片 %BGR
        set_frame = -1
        for path, img, im0s, vid_cap in dataset:
            if self.process is False:
                return
            set_frame += 1
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            # 如果图像是三维张量 那么为其增加一维 --> 用于批处理
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (
                    old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_value, self.iou_value, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)
            t3 = time_synchronized()

            # Process detections
            #
            for i, det in enumerate(pred):  # detections per image
                if self.process is False:
                    return
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    store_txt_list = []
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            if len(coco_cls):
                                label2 = coco_names_list[int(cls)]
                                if coco_cls.get(label2):
                                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)],
                                                 line_thickness=1)
                            else:
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)],
                                             line_thickness=1)

                        # 存储基本信息
                        if source.endswith("png") or source.endswith("jpg"):
                            store_txt_list.append({"xyxy": xyxy, "cls": names[int(cls)], "conf": conf})
                    if source.endswith("png") or source.endswith("jpg"):
                        store_txt(store_txt_list, set_frame)

                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        pic_sent.emit([im0.shape])
                        print(f" The image with the result is saved in: {save_path}")
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)
                        # 一帧一帧的图片 保存下来
                        # save_video_img_path = str(save_dir / f"vid_{frame}.jpg")
                        # cv2.imwrite(save_video_img_path, im0)

                        # 把帧的数组信息保存成图片 pixmap形式发送信号
                        # pix_list 存储 [图片, 第几帧]
                        pixmap = numpy_to_pixmap(im0)
                        pix_list = [pixmap, frame]
                        vid_sent.emit(pix_list)

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" \
                if save_txt else ''
            # print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')


ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("myappid")


def getOpt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='F:/test_for_software/vid_test0.mp4',
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    # 界面类
    app = QApplication(sys.argv)
    w = Example()
    w.show()
    w.preWarmUp()
    sys.exit(app.exec_())
