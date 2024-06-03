
from train_plot_selection import *
from train_detail import *
from coordinate_window import *
from train_chart_window import *
from main_window import *
from train_result_window import *


class TrainPlotWindow(QWidget):

    def __init__(self, params={"epoch": 100, "batch-size": 4, "workers": 8}):
        super(TrainPlotWindow, self).__init__()
        # 最外层
        self.outer_layout = QVBoxLayout(self)
        #   {顶部绘图widget}
        #                / top_label_widegt == title_label1 || progressBar || progress_label
        #   top_widget ==                      / scroll1 -> page11 (hlayout11) plot1 || plot2 ...
        #                \ top_stack_widget  ==
        #                                      \ scroll2
        self.top_widget, self.top_label_widget = QWidget(), QWidget()
        self.top_stack_widget = QStackedWidget()
        self.top_layout = QVBoxLayout(self.top_widget)
        # top_label_widget layout
        self.top_label_layout = QHBoxLayout(self.top_label_widget)
        self.top_title_label, self.top_progress_label = QLabel("图像"), QLabel("top_progress")
        self.top_progress_bar = ProgressBar()
        # top_stack_widget1
        self.top_scroll1 = ScrollArea()
        self.page11 = QWidget()
        self.hlayout11 = QHBoxLayout(self.page11)

        #   {底部结果widget}
        self.bottom_widget, self.bottom_label_widget = QWidget(), QWidget()
        self.bottom_stack_widget = QStackedWidget()
        self.bottom_layout = QVBoxLayout(self.bottom_widget)
        # bottom_label_widget layout
        self.bottom_label_layout = QHBoxLayout(self.bottom_label_widget)
        self.bottom_title_label, self.bottom_epoch_label = QLabel("指标"), QLabel("epoch: ?")
        self.bottom_combo = ComboBox()
        # bottom_label_layout
        self.bottom_scroll1 = ScrollArea()
        self.page21 = QWidget()
        self.hlayout21 = QHBoxLayout(self.page21)

        # 保存 plot widget AND result widget
        self.plot_widget_list = []
        self.result_widget_list = []

        # 菜单
        self.menu = RoundMenu(self)
        self.action1 = Action(FluentIcon.ADD, "添加图表")
        self.action2 = Action(QIcon(TRAIN_ICON), "开始训练")
        self.action3 = Action(QIcon(DETECT_ICON), "开始检测")
        self.menu.addActions([
            self.action1,
            self.action2,
            self.action3
        ])

        # 图表选择class and 训练算法
        self.plot_selected_widget = TrainPlotSelectedWidget()
        self.train_widget = testWidget()
        # self.plot_widget_list = []
        # self.result_widget_list = []

        # 需要画哪些图像
        self.plot_list = self.plot_selected_widget.getItemInfo()

        # 范围list | plot name list
        self.y_range_list = [0.20, 0.20, 0.20, 0.10, 0.00]
        self.plot_name_list = ["epoch-Precision", "epoch-Recall", "epoch-mAP0.5", "epoch-mAP", "epoch-Loss"]

        # epoch nums
        self.epoch_nums = 100

        # save all train data || 训练n次的save all train data
        self.train_data_list = []
        self.all_train_data_list = []

        # 进度条maximum   这里的4后面要换掉
        self.bar_max = TRAIN_PICTURES // 4
        self.batch_size = 4

        # 检测界面
        self.detect_window = Example()

        # 结果界面
        self.result_window = TrainResultWidget(all_train_data_list)

        # train_param_dict
        self.train_param_dict = params

        self.initParam()
        self.initStructure()
        self.slotConnection()

    def initStructure(self):
        # top label layout
        self.top_label_layout.addWidget(self.top_title_label)
        self.top_label_layout.addWidget(self.top_progress_bar)
        self.top_label_layout.addWidget(self.top_progress_label)

        # bottom label layout
        self.bottom_label_layout.addWidget(self.bottom_title_label)
        self.bottom_label_layout.addWidget(self.bottom_epoch_label)
        self.bottom_label_layout.addWidget(self.bottom_combo)

        # top stacked layout
        # 动态创建len(self.plot_list)个plot widget
        for i in range(len(self.plot_list)):
            self.hlayout11.addWidget(self.createPlotWidget(i))
        self.top_scroll1.setWidget(self.page11)
        self.top_stack_widget.addWidget(self.top_scroll1)

        # bottom stacked layout
        # 动态创建len(self.plot_list)个result widget
        for i in range(len(self.plot_list)):
            self.hlayout21.addWidget(self.createResultWidget(i))
        self.bottom_scroll1.setWidget(self.page21)
        self.bottom_stack_widget.addWidget(self.bottom_scroll1)

        # top layout
        self.top_layout.addWidget(self.top_label_widget)
        self.top_layout.addWidget(self.top_stack_widget)

        # bottom layout
        self.bottom_layout.addWidget(self.bottom_label_widget)
        self.bottom_layout.addWidget(self.bottom_stack_widget)

        # 最外层 layout
        self.outer_layout.addWidget(self.top_widget)
        self.outer_layout.addWidget(self.bottom_widget)

    def initParam(self):
        self.resize(1000, 800)

        # style sheet
        self.top_widget.setStyleSheet("background-color: white; border-radius: 10px")
        self.bottom_widget.setStyleSheet("background-color: white; border-radius: 10px")

        self.top_progress_bar.setMaximum(self.bar_max)
        # self.top_title_label.setStyleSheet("color:white;")
        # self.top_progress_label.setStyleSheet("color:white;")
        # self.bottom_title_label.setStyleSheet("color:white;")
        # self.bottom_progress_label.setStyleSheet("color:white;")
        self.top_title_label.setFont(self.customFont(weight="1"))
        self.top_progress_label.setFont(self.customFont(weight="1"))
        self.bottom_title_label.setFont(self.customFont(weight="1"))
        self.bottom_epoch_label.setFont(self.customFont(weight="1"))

        self.top_scroll1.setWidgetResizable(True)
        self.top_scroll1.setWidgetResizable(True)

        # bottom widget
        self.bottom_widget.setMaximumHeight(400)

    def paintEvent(self, event):
        # 动态伸缩plot_widget大小
        for plot_widget in self.plot_widget_list:
            plot_widget.setFixedWidth(plot_widget.height())

        # self.top_scroll1.setWidget(self.page11)
        # self.top_stack_widget.addWidget(self.top_scroll1)

    def slotConnection(self):
        # 菜单项事件
        self.action1.triggered.connect(self.openPlotSelectedList)
        self.action2.triggered.connect(self.openTrainWidget)
        self.action3.triggered.connect(self.openDetectWidget)
        # draw
        self.train_widget.on_plot_signal.connect(self.drawPlots)

        # update bar
        self.train_widget.update_progress_bar_signal.connect(self.updateBar)

        # train finished
        # self.train_widget.train_thread.on_train_finished.connect(self.trainFinished)
        self.train_widget.on_train_finished.connect(self.trainFinished)

    # set train params
    def setTrainParams(self, train_params):
        self.train_param_dict = train_params
        batch_size = int(train_params["batch-size"])
        self.top_progress_bar.setMaximum(TRAIN_PICTURES // batch_size)
        self.train_widget.setTrainParams(train_params)

    def contextMenuEvent(self, event):
        action = self.menu.exec_(self.mapToGlobal(event.pos()))

    # 自定义字体属性
    def customFont(self, font_size=16, font_family="Arial", weight=""):
        font_i = QFont(font_family, font_size)
        if len(weight) >= 1:
            font_i.setBold(True)
        return font_i

    # 动态创建训练plot  x: epoch  y:     x:P  y:R   x:epoch y:loss
    def createPlotWidget(self, idx):

        plot_widget = DynamicLineChart((0, self.epoch_nums), (self.y_range_list[idx], 1.00),
                                       self.plot_name_list[idx])
        plot_widget.setMinimumWidth(300)
        plot_widget.setMinimumHeight(300)
        # layout_i = QVBoxLayout(plot_widget)
        # layout_i.addWidget(QLabel("plot"))
        self.plot_widget_list.append(plot_widget)
        return plot_widget

    # 动态创建训练result 包括P R mAP0.5  mAP0.5：0.75 Loss
    def createResultWidget(self, idx):
        result_widget = CircularGauge((self.y_range_list[idx], 1.00),
                                      0, self.plot_name_list[idx])
        result_widget.setMinimumWidth(300)
        result_widget.setMinimumHeight(300)
        # layout_i = QVBoxLayout(result_widget)
        # layout_i.addWidget(QLabel("plot"))
        self.result_widget_list.append(result_widget)
        return result_widget

    # open图表选择list
    def openPlotSelectedList(self):
        self.plot_selected_widget.show()

    def openTrainWidget(self):
        self.train_widget.show()

    def openDetectWidget(self):
        self.detect_window.show()
        self.detect_window.preWarmUp()

    def updateBar(self, value):
        self.top_progress_bar.setValue(value)

    # 根据combo查看之前epoch的数据
    def reshowResult(self, chart_index):
        train_data = self.train_data_list[chart_index]
        m_precision = str(train_data[0])[:4]
        m_recall = str(train_data[1])[:4]
        map_05 = str(train_data[2])[:4]
        map_95 = str(train_data[3])[:4]
        loss = str(train_data[4] + train_data[5] + train_data[6])[:4]
        self.result_widget_list[0].setCurrentValue(float(m_precision))
        self.result_widget_list[1].setCurrentValue(float(m_recall))
        self.result_widget_list[2].setCurrentValue(float(map_05))
        self.result_widget_list[3].setCurrentValue(float(map_95))
        self.result_widget_list[4].setCurrentValue(float(loss))

    # 训练结束的处理
    def trainFinished(self):
        self.showSuccessTip()

        # combo indexChanged
        # self.bottom_combo.currentIndexChanged(self.reshowResult)

        # save all train_data_list
        self.all_train_data_list.append(self.train_data_list)

        # show result window
        self.result_window = TrainResultWidget(self.all_train_data_list)
        self.result_window.show()

    # 结束训练弹窗
    def showSuccessTip(self):
        TeachingTip.create(
            target=self.top_progress_label,
            icon=InfoBarIcon.SUCCESS,
            title='训练',
            content="训练结束咯 快收集数据吧 ！",
            isClosable=True,
            tailPosition=TeachingTipTailPosition.BOTTOM,
            duration=5000,
            parent=self
        )

    # ========================== drawing functions ========================== #
    def drawPlots(self, train_param_list):
        # 可能需要改
        # self.top_progress_bar.setMaximum(TRAIN_PICTURES // self.batch_size)

        self.train_data_list.append(train_param_list)
        # draw plots
        epoch = train_param_list[0]
        results = train_param_list[1]
        maps = train_param_list[2]
        times = train_param_list[3]
        m_precision = str(results[0])[:4]
        m_recall = str(results[1])[:4]
        map_05 = str(results[2])[:4]
        map_95 = str(results[3])[:4]
        loss = str(results[4] + results[5] + results[6])[:4]

        # print(f"{float(m_precision)}  {float(m_recall)}  {float(map_05)}  {float(map_95)}  {float(loss)}")
        # epoch - precision     precision > 0.5 | epoch: 0-100
        self.plot_widget_list[0].add_data_point(int(epoch), float(m_precision))
        # epoch - recall
        self.plot_widget_list[1].add_data_point(int(epoch), float(m_recall))
        # epoch - map0.5
        self.plot_widget_list[2].add_data_point(int(epoch), float(map_05))
        # epoch - map
        self.plot_widget_list[3].add_data_point(int(epoch), float(map_95))
        # epoch - loss
        self.plot_widget_list[4].add_data_point(int(epoch), float(loss))
        self.top_progress_label.setText(f"{epoch} / {99}")

        # epoch - precision/recall/map0.5/map/loss
        self.bottom_epoch_label.setText(f"epoch: {epoch}")
        self.result_widget_list[0].setCurrentValue(float(m_precision))
        self.result_widget_list[1].setCurrentValue(float(m_recall))
        self.result_widget_list[2].setCurrentValue(float(map_05))
        self.result_widget_list[3].setCurrentValue(float(map_95))
        self.result_widget_list[4].setCurrentValue(float(loss))
        self.bottom_combo.addItem(f"epoch-{epoch}")
        self.bottom_combo.setCurrentIndex(int(epoch))

        # update progress bar to 0
        self.top_progress_bar.setValue(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    train_plot_window = TrainPlotWindow()
    train_plot_window.show()
    sys.exit(app.exec_())
