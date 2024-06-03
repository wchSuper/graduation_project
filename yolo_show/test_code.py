import os


def get_two_levels_up_directory(path):
    # 获取指定目录的上两层目录
    two_levels_up_directory = os.path.abspath(os.path.join(path, "../.."))
    return two_levels_up_directory


if __name__ == "__main__":
    # 示例路径，可以替换为你要获取上两层目录的路径
    path = "E:/YOLOv7_/datasets/coco128/images/train2017"
    two_levels_up_directory = get_two_levels_up_directory(path)
    new_path = os.path.join(two_levels_up_directory, "labels")
    print("Given Path:", path)
    print("Two Levels Up Directory:", two_levels_up_directory)
    print("new path:", new_path)