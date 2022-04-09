import cv2 as cv
import numpy as np
import sys
from PIL import Image


def img_cut(img):

    width, height = int(img.size[0]), int(img.size[1])
    item_width = int(width / 4)
    item_height = int(height / 4)
    box_list = []
    crop_list = []
    for i in range(0, 4):
        for j in range(0, 4):
            box = (j*item_width, i*item_height, (j+1)*item_width, (i+1)*item_height)
            box_list.append(box)
            crop = img.crop(box)
    img_list = [img.crop(box) for box in box_list]

    return img_list

def img_slim(event_filename, sub_event_filename):
    id = 0
    x_delta = 120
    y_delta = 160
    for x in range(0, 480, 120):
        for y in range(0, 640, 160):
            # 1. 读取图片
            img2 = cv.imread(event_filename)  # 读取事件图片
            index = np.full((480, 640), True)
            index[x: x + x_delta, y: y + y_delta] = False

            img2[index] = [255, 255, 255]
            cv.imwrite(sub_event_filename + 'sub_event_{}.png'.format(id), img2)
            id += 1

'''
pixel_num(img)  像素点数量计算(红色)
n  像素点数量

'''
def pixel_num(img):
    """"
    图片4*4分割
    :param img: 分割图片
    :return 4*4 分割结果
    """
    height, width = int(img.shape[0]), int(img.shape[1])

    start_hor = 0
    start_ver = 0

    end_hor = width
    end_ver = height
    
    #统计事件像素点数量
    # npim = np.zeros((height, width, 3), dtype='uint8')
    # npim = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
    num = 0
    for i in range(start_ver,end_ver):
        for j in range(start_hor,end_hor):
            if ((img[i, j, 0] == 0) and (img[i, j, 1] == 0) and (img[i, j, 2] == 255)) or ((img[i, j, 0] == 255) and (img[i, j, 1] == 0) and (img[i, j, 2] == 0)):
                num += 1
    return num

'''
img_save(img_list)  保存图片为PNG文件
img_list  图片列表

'''
def img_save(img_list, event_subgrapg_filename):
    index = 0
    for image in img_list:
        image.save(event_subgrapg_filename+ 'sub_event_{}.png'.format(index), 'PNG')
        index += 1

'''
img_flag(num)  根据像素数量判断分割标志位
flag[][]  分割标志

'''
def img_flag(num):
    flag = np.ones((4,4),dtype = int)
    for i in range(4):
        for j in range(4):
            if (num[i][j] < 150):
                flag[i][j] = 0
    return flag

'''
result_cut[]  4*4 分割结果图片
unit[] 4*4 分割结果图片
num[][]  4*4 像素点数量
flag[][]  4*4  分割标志
'''
def slim_mask(event_filename, sub_event_filename):
    """"
    将事件数据4*4分割, 并输出事件子图
    :param event_filename: 事件图片
    :param sub_event_filename: event子图输出路径
    :return 4*4 mask掩码值
    """
    img_list = []
    index = 0
    unit = []

    #图片分割保存
    img = Image.open(event_filename)
    print("正在切割事件图片！ ")
    img_list = img_cut(img)
    print("正在保存事件子图！ ")
    img_slim(event_filename, sub_event_filename)
    print("事件子图保存完成！ ")
    #像素点数量计算
    num = np.zeros((4, 4), dtype=int)

    for k in range(16):
        unit.append(cv.cvtColor(np.asarray(img_list[k]),cv.COLOR_RGB2BGR))

    for i in range(4):
        for j in range(4):
            num[i][j] = pixel_num(unit[index])
            index += 1
    print("event 4*4子图中所对应的像素数量： {}".format(num))
    #输出分割标志位 4*4
    mask = img_flag(num)
    print("事件子图mask值为： {}".format(mask))
    return mask

