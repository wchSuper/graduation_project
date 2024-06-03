# 所有的图片后缀以及视频后缀
picture_suffix_list = ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif"]
video_suffix_list = ["mp4", "avi", "mov", "wmv", "flv"]

# 错误的文件路径返回
error_file_ret = ["**dir**", "**last-none", "**next-none**", "**error**"]

# 控件大小
VIDEO_PLAY_SIZE = 40

# 顶层按钮总数
BTN_NUM = 6
BTN_EXTRA_NUM = 1

# 测试的检测后的图片路径  根据自己需求更改
test_detect_path = "F:/test_for_software_finished/000000000009.jpg"

# 标题图标
TITLE_ICON = "./icons/visualization.png"
# 实时按钮路径
SHISHI_DETECT_ON = "./icons/shishi_detect_on.png"
SHISHI_DETECT_OFF = "./icons/shishi_detect_off.png"

# warm up 按钮路径
WARM_UP_ICON = "./icons/fire_icon.png"
WARM_OFF_ICON = "./icons/fire_off_icon.png"

# stop action 按钮
STOP_ACTIONS_ICON = "./icons/stop_actions.png"

# 存储目录


# 顶层背景
TOP_BG = "./icons/top_bg.png"

# 开场gif
GIF1 = "./icons/loading.gif"
WARM_GIF = "./icons/warm.gif"

# 定时器2的time
TIME_S = 30

# train  | detect | clear
TRAIN_ICON = "./icons/train.png"
DETECT_ICON = "./icons/detect.png"
CLEAR_ICON = "./icons/clear.png"

TRAIN_PICTURES = 128
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone',
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear',
               'hair drier', 'toothbrush']

param_names1 = ['Precision', 'Recall', 'mAP50', 'mAP95', 'loss']
param_names2 = ['mAP']
param_names3 = ['detection time', 'NMS time', 'all time']

# epoch


# epoch2
epoch2 = [2, (0.5424457877510411, 0.6163198677375248, 0.635327578370867, 0.4038480026557473, 0.048948001116514206,
              0.033025987446308136, 0.011647977866232395),
          [0.49419, 0.31498, 0.17852, 0.72239, 0.69098, 0.64523, 0.48542, 0.33133, 0.16059, 0.13334, 0.40385, 0.62952,
           0.40385, 0.25358, 0.49323, 0.85374, 0.61032, 0.77086, 0.40385, 0.40385, 0.56202, 0.89566, 0.89585, 0.61528,
           0.17433, 0.40148, 0.094018, 0.38025, 0.27206, 0.53577, 0.036197, 0.42669, 0.22788, 0.1947, 0.0044333,
           0.27005, 0.26775, 0.40385, 0.30015, 0.20434, 0.35024, 0.38118, 0.26286, 0.29328, 0.17946, 0.39434, 0.03317,
           0.40385,
           0.67061, 0.32576, 0.15433, 0.46294, 0.72141, 0.65172, 0.67388, 0.60616, 0.24114, 0.62268, 0.37834, 0.53262,
           0.32667, 0.79524, 0.72136, 0.29705, 0, 0.38628, 0.40385, 0.17512, 0.69157, 0.3094, 0.40385, 0.22089,
           0.49936, 0.093968, 0.48492, 0.59705, 0.0055421, 0.27081, 0.40385, 0.33476],
          (2.136608585715294, 1.6508419066667557, 3.7874504923820496, 640, 640, 8)]

epoch1 = [1, (0.5890644242139246, 0.6052107202245444, 0.6355665213355326, 0.39744231733961094, 0.049235280603170395,
              0.032415181398391724, 0.011951200664043427),
          [0.49052, 0.28571, 0.16517, 0.65951, 0.63824, 0.60524, 0.41934, 0.31013, 0.096126, 0.15119, 0.39744, 0.62954,
           0.39744, 0.24205, 0.42605, 0.77086, 0.61823, 0.59659, 0.39744, 0.39744, 0.62251, 0.79613, 0.90198, 0.58247,
           0.18464, 0.41878, 0.091174, 0.38097, 0.65223, 0.69935, 0.099503, 0.41719, 0.12028, 0.17859, 0.0043605,
           0.26122, 0.33658, 0.39744, 0.31977, 0.21016, 0.33515, 0.35769, 0.25195, 0.32711, 0.17069, 0.40967, 0.028435,
           0.39744,
           0.74563, 0.41227, 0.15089, 0.46573, 0.99519, 0.64011, 0.7918, 0.64067, 0.21764, 0.50991, 0.38809, 0.38792,
           0.28271, 0.53615, 0.66276, 0.25643, 0, 0.46585, 0.39744, 0.18748, 0.55902, 0.2807, 0.39744, 0.17507,
           0.46403, 0.10349, 0.44988, 0.46017, 0.0022675, 0.3188, 0.39744, 0.40469],
          (2.1049827337265015, 1.6124453395605087, 3.71742807328701, 640, 640, 8)]

all_train_data_list = [[epoch1, epoch2, epoch1], [epoch1, epoch2, epoch2]]

coco_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
              'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
              'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
              'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
              'cell phone',
              'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
              'teddy bear',
              'hair drier', 'toothbrush']
coco_names_zh = ['人', '自行车', '汽车', '摩托车', '飞机', '公交车', '火车', '卡车', '船', '红绿灯',
                 '消防栓', '停止标志', '停车计时器', '长椅', '鸟', '猫', '狗', '马', '羊', '牛',
                 '大象', '熊', '斑马', '长颈鹿', '背包', '雨伞', '手提包', '领带', '手提箱', '飞盘',
                 '滑雪板', '单板滑雪', '运动球', '风筝', '棒球棒', '棒球手套', '滑板', '冲浪板', '网球拍', '瓶子',
                 '酒杯', '杯子', '叉子', '刀', '勺子', '碗', '香蕉', '苹果', '三明治', '橙子',
                 '西兰花', '胡萝卜', '热狗', '披萨', '甜甜圈', '蛋糕', '椅子', '沙发', '盆栽', '床',
                 '餐桌', '厕所', '电视', '笔记本电脑', '鼠标', '遥控器', '键盘', '手机', '微波炉', '烤箱',
                 '烤面包机', '水槽', '冰箱', '书', '钟', '花瓶', '剪刀', '泰迪熊', '吹风机', '牙刷']
