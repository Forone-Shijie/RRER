import cv2
import csv
from numba import jit

#对比所有叠加图片的对比度，并返回
def contrast(img):
    """
    对比叠加图片的对比度，并返回
    :param img: 输入图片
    :return: 对比度
    """
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #彩色转为灰度图片
    global m, n
    m, n = img1.shape
    global rows_ext, cols_ext
    #图片矩阵向外扩展一个像素
    img1_ext = cv2.copyMakeBorder(img1,1,1,1,1,cv2.BORDER_REPLICATE) / 1.0   # 除以1.0的目的是uint8转为float型，便于后续计算
    rows_ext, cols_ext = img1_ext.shape
    return mun(img1_ext)

@jit(nopython=True)
def mun(img1_ext):
    global m,n
    b = 0.0
    for i in range(1,rows_ext-1):
        for j in range(1,cols_ext-1):
            b += ((img1_ext[i,j]-img1_ext[i,j+1])**2 + (img1_ext[i,j]-img1_ext[i,j-1])**2 +
                    (img1_ext[i,j]-img1_ext[i+1,j])**2 + (img1_ext[i,j]-img1_ext[i-1,j])**2)

    cg = b/(4*(m-2)*(n-2)+3*(2*(m-2)+2*(n-2))+2*4) #对应上面48的计算公式
    return cg

def cal_contrast(added_output_filename, img_list, location, first_flag, right_start_flag, fullevent_flag, order_id):
    """
    对比所有叠加图片的对比度，并返回最佳的叠加位置
    :param added_output_filename: 输入图片路径
    :param img_list: 图片列表
    :param location: 叠加位置 [row, col]
    :param first_flag: 第一次叠加标志位
    :param right_start_flag: 右侧首次叠加标志位
    :param fullevent_flag: 是否为全图叠加
    :param order_id :计算顺序
    :return: 最佳对比度所对应位置 [bestrow, bestcol]
    """
    contrast_result = [] #存储对比度
    best_location = []  #存储移动位置信息，并依据索引返回最优值
    index = 0
    #遍历叠加生成的所有图片 8*8
    if (first_flag == True):
        row_start = location[0] - 8
        row_end = location[0] + 8
        col_start = location[1] - 8
        col_end = location[1] + 8

    elif (right_start_flag == True):
        row_start = location[0]-4
        row_end = location[0]+4
        col_start = location[1]-18
        col_end = location[1]

    else:
        row_start = location[0] - 4
        row_end = location[0] + 4
        col_start = location[1] - 14
        col_end = location[1]

    for each_row in range(row_start, row_end, 1):
        for each_col in range(col_start, col_end, 1):
            print("正在计算event与canny叠加图片的对比度, row: {} , col: {} , index: {}".format(each_row, each_col, index))
            #img = cv2.imread(added_output_filename + 'row_{}_col_{}_index_{}.png'.format(each_row, each_col, index))
            img = img_list[index]
            best_location.append([each_row, each_col])
            contrast_result.append(contrast(img))
            print("event与canny叠加图片的对比度计算完成, row: {} , col: {} , index: {}".format(each_row, each_col, index))
            index += 1


    print("所有叠加图片对比度： {}".format(contrast_result)) #所有叠加图片的对比度
    print("最小对比度索引: {}".format(contrast_result.index(min(contrast_result)))) #最小对比度索引
    print("最小对比度为： {}".format(min(contrast_result)))  # 最小对比度
    print("对应索引找到最佳的叠加位置： {}".format(best_location[contrast_result.index(min(contrast_result))])) #按照对应索引找到最佳的叠加位置
    if (fullevent_flag):
        with open("/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/full_result/contrast_{}.csv".format(order_id), "a+") as csvfile:
            writer = csv.writer(csvfile)
            # 写行数据
            writer.writerow([contrast_result])
            writer.writerow([])

        with open("/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/full_result/best_location_{}.csv".format(order_id), "a+", encoding='utf8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # 写行数据
            writer.writerow([best_location[contrast_result.index(min(contrast_result))]])
            csvfile.close()
    elif (fullevent_flag == 0):

        with open("/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/result/contrast_{}.csv".format(order_id), "a+") as csvfile:
            writer = csv.writer(csvfile)
            # 写行数据
            writer.writerow([contrast_result])
            writer.writerow([])

        with open("/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/result/best_location_{}.csv".format(order_id), "a+", encoding='utf8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # 写行数据
            writer.writerow([best_location[contrast_result.index(min(contrast_result))]])
            csvfile.close()

    return best_location[contrast_result.index(min(contrast_result))] #返回最佳的叠加位置 [bestrow, bestcol]