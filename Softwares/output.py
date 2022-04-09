import cv2
import numpy as np
import add_nomove

def event_merge(sub_event_filename, best_location, merged_output_filename):
    """
    将event子图按照计算出来的最佳的位置叠加到白色背景上，最终图片大小为 1080X1440 与canny图相同大小
    :param sub_event_filename: 事件数据子图路径
    :param  best_location: 计算出来的最好的的叠加位置
    :param merged_output_filename： 最终出图位置, 初始值需要是 1080X1440 的白色图片
    :return: None
    """
    print("正在将event子图按照最佳位置叠加！ ")
    add_nomove.subgraph_add_nomove(sub_event_filename, merged_output_filename, best_location, merged_output_filename, 1)
    print("event子图按照最佳位置叠加完成！ ")

def img2red(merged_output_filename):
    """
    将合成后的图像 1080X1440 的颜色进行处理
    :param merged_output_filename： 最终出图位置
    :return: 处理颜色后的merged图像
    """
    print("正在将fullevent图片颜色统一化")
    blue = np.array([255, 0, 0]) #定义蓝色

    img = cv2.imread(merged_output_filename)
    rows, cols = img.shape[:2]
    for row in range(rows):
        for col in range(cols):
            # 将像素点与蓝色像素做对比
            if (img[row][col] == blue).all():
                # 将蓝色像素换成红色像素
                img[row][col] = np.array([0, 0, 255])

    cv2.imwrite(merged_output_filename, img)
    print("fullevent图片颜色统一化完成！ ")


def fullevent_merge(fullevent_filename, canny_filename, result_filename):
    """
    将移动好的fullevent叠加到canny上，最终图片大小为 1080X1440 与canny图相同大小
    :param fullevent_filename: 事件数据全图路径
    :param canny_filename: canny图路径
    :param result_filename： 最终的叠加结果输出的位置
    :return: None
    """
    print("正在生成fullevent与canny叠加图！ ")
    add_nomove.subgraph_add_nomove(fullevent_filename, canny_filename, [150, 150], result_filename, 0)
    print("fullevent与canny叠加图生成完成！ ")