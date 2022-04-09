import add
import add_nomove
import contrast
import output
import cv2
import csv
import os
import shutil
import numpy as np
import time
import pixel_num
import test

def copy(src_dir, dst_dir):
    for root, _, fnames in os.walk(src_dir):
      for fname in sorted(fnames):  # sorted函数把遍历的文件按文件名排序
        fpath = os.path.join(root, fname)
        shutil.copy(fpath, dst_dir)  # 完成文件拷贝
        print(fname + " has been copied!")

for i in range(14):
    #初始化
    src_list = ['/data/sjzhang/RRER/dataset/pre_dataset/image_data/',
                '/data/sjzhang/RRER/dataset/pre_dataset/event_data/',
                '/data/sjzhang/RRER/dataset/pre_dataset/canny_data/']
    dst_list = ['/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/rgb/',
                '/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/event/',
                '/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/canny/']

    event_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/event/event_"
    rgb_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/rgb/rgb_"
    # sub_event_filename = "../Img/sub_event/" #多张事件子图路径
    canny_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/canny/canny_"  # canny图
    # added_output_filename = "../Img/added/" #叠加图片输出位置
    # merged_output_filename = "../Img/result/merged_output_" #最终输出位置
    result_canny_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/result/result_canny_"
    result_rgb_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/result/result_rgb_"

    #数据集生成循环，清除并且重新复制
    shutil.rmtree('/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/')
    shutil.copytree('/data/sjzhang/RRER/dataset/pre_dataset/Img', '/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img')

    for (src_dir, dst_dir) in zip(src_list, dst_list):
        copy(src_dir + '{}/'.format(i), dst_dir)

    file_list = os.listdir(event_filename[:-6])
    test.init(len(file_list))
    print(len(file_list))

    for order_id in range(len(file_list)):

        event_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/event/event_"
        rgb_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/rgb/rgb_"
        sub_event_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/sub_event/"  # 多张事件子图路径
        canny_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/canny/canny_"  # canny图
        added_output_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/added/"  # 叠加图片输出位置
        merged_output_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/result/merged_output_"  # 最终输出位置
        result_canny_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/result/result_canny_"
        result_rgb_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/result/result_rgb_"

        event_filename += '{}.png'.format(order_id)
        rgb_filename += '{}.png'.format(order_id)
        canny_filename += '{}.png'.format(order_id)
        merged_output_filename += '{}.png'.format(order_id)
        result_canny_filename += '{}.png'.format(order_id)
        result_rgb_filename += '{}.png'.format(order_id)

        time_start = time.time() #记录运行时间
        img_list = []#存储图片, 不直接存储照片以提升速度
        location_result = np.zeros([4, 4, 2], dtype= 'int') #构建4*4位置矩阵
        # delivery_rule = np.zeros(4, 4) #构建4*4传递矩阵
        mask = np.ones([4, 4], dtype= 'int') #构建4*4掩码矩阵

        #顺序矩阵
        order_rule = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1], [3, 0], [3, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 2], [2, 3], [3, 2], [3, 3]]

        # 构建4*4传递矩阵, 规则为相近等权重叠加传递
        delivery_rule = np.array([ [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 1], [0, 1]], [[0, 2], [0, 2]],
                                   [[0, 0], [0, 0]], [[1, 0], [0, 1]], [[0, 2], [0, 2]], [[1, 2], [0, 3]],
                                   [[1, 0], [1, 0]], [[2, 0], [1, 1]], [[1, 2], [1, 2]], [[2, 2], [1, 3]],
                                   [[2, 0], [2, 0]], [[3, 0], [2, 1]], [[2, 2], [2, 2]], [[3, 2], [2, 3]] ])
        delivery_rule = delivery_rule.reshape(4, 4, 2, 2)

        location_result[0][0] = [272, 160] #第一个图的默认最优位置

        with open("/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/result/contrast_{}.csv".format(order_id), "w", encoding='utf8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # 写行数据
            writer.writerow(["contrast_result"])
            csvfile.close()

        with open("/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/result/best_location_{}.csv".format(order_id), "w", encoding='utf8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # 写行数据
            writer.writerow(["best_row", "best_col"])
            csvfile.close()

        cv2.setUseOptimized(True)

        if __name__ == "__main__":

            index = 0

            #事件数据切割
            mask = pixel_num.slim_mask(event_filename, sub_event_filename)

            #16个子图索引
            # for row in range(4):
            #     for col in range(4):
            for [row, col] in order_rule:
                index = row * 4 + col * 1
                print("正在进行第{}个事件子图的运算！ ".format(index))
                print(mask[row][col] == 0)
                if (mask[row][col] == 0):

                    best_location = [272, 153]
                    location_result[row][col] = best_location  # 存储最佳位置
                    #存储最佳位置
                    with open("/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/result/best_location_{}.csv".format(order_id), "a+", encoding='utf8', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        # 写行数据
                        writer.writerow([best_location])
                        csvfile.close()

                    # 按照每个子图的最佳位置叠加成完整的事件图
                    output.event_merge(sub_event_filename + 'sub_event_{}.png'.format(index), best_location,
                                       merged_output_filename)

                else:
                    index = row*4 + col*1

                    # 计算出16个子图的叠加最佳位置，并将其存放到location_result中, 4*4, [row, col]
                    if (mask[delivery_rule[row][col][0][0]][delivery_rule[row][col][0][1]] == 0) and (mask[delivery_rule[row][col][1][0]][delivery_rule[row][col][1][1]] == 0) and (row == 0 ) and (col != 0 ): #第一行给临近位置
                        best_location = location_result[delivery_rule[row][col - 1][0][0]][delivery_rule[row][col - 1][0][1]]
                    elif (mask[delivery_rule[row][col][0][0]][delivery_rule[row][col][0][1]] == 0) and (mask[delivery_rule[row][col][1][0]][delivery_rule[row][col][1][1]] == 0) and (row != 0 ) and (col ==0): #第一列给临近位置
                        best_location = location_result[delivery_rule[row - 1][col][0][0]][delivery_rule[row - 1][col][0][1]]
                    elif (mask[delivery_rule[row][col][0][0]][delivery_rule[row][col][0][1]] == 0) and (mask[delivery_rule[row][col][1][0]][delivery_rule[row][col][1][1]] == 0) and (row != 0 ) and (col !=0): #normal位置给对角位置
                        best_location = location_result[delivery_rule[row - 1][col - 1][0][0]][delivery_rule[row - 1][col - 1][0][1]]
                    elif (mask[delivery_rule[row][col][0][0]][delivery_rule[row][col][0][1]] == 0) and (mask[delivery_rule[row][col][1][0]][delivery_rule[row][col][1][1]] == 0) and (row == 0 ) and (col ==0): #第一列给临近位置
                        best_location = location_result[delivery_rule[row][col][0][0]][delivery_rule[row][col][0][1]]

                    elif (mask[delivery_rule[row][col][0][0]][delivery_rule[row][col][0][1]] == 1) and (mask[delivery_rule[row][col][1][0]][delivery_rule[row][col][1][1]] == 1): #两个都为1, 除以二
                        best_location = ((location_result[delivery_rule[row][col][0][0]][delivery_rule[row][col][0][1]]) *
                                         mask[delivery_rule[row][col][0][0]][delivery_rule[row][col][0][1]]
                                         + (location_result[delivery_rule[row][col][1][0]][delivery_rule[row][col][1][1]]) *
                                         mask[delivery_rule[row][col][1][0]][delivery_rule[row][col][1][1]]) / 2

                    elif (mask[delivery_rule[row][col][0][0]][delivery_rule[row][col][0][1]] == 0) and (mask[delivery_rule[row][col][1][0]][delivery_rule[row][col][1][1]] == 1): #一个为0, 除以1
                        best_location = ((location_result[delivery_rule[row][col][0][0]][delivery_rule[row][col][0][1]]) *
                                         mask[delivery_rule[row][col][0][0]][delivery_rule[row][col][0][1]]
                                         + (location_result[delivery_rule[row][col][1][0]][delivery_rule[row][col][1][1]]) *
                                         mask[delivery_rule[row][col][1][0]][delivery_rule[row][col][1][1]])

                    elif (mask[delivery_rule[row][col][0][0]][delivery_rule[row][col][0][1]] == 1) and (mask[delivery_rule[row][col][1][0]][delivery_rule[row][col][1][1]] == 0): #一个为0, 除以1
                        best_location = ((location_result[delivery_rule[row][col][0][0]][delivery_rule[row][col][0][1]]) *
                                         mask[delivery_rule[row][col][0][0]][delivery_rule[row][col][0][1]]
                                         + (location_result[delivery_rule[row][col][1][0]][delivery_rule[row][col][1][1]]) *
                                         mask[delivery_rule[row][col][1][0]][delivery_rule[row][col][1][1]])

                    best_location = best_location.astype(int) #像素点位置转为整形
                    img_list = add.subgraph_add(sub_event_filename + 'sub_event_{}.png'.format(index), canny_filename, best_location, added_output_filename + "{}/".format(index), (index == 0), (index == 2)) #叠加子图
                    best_location = contrast.cal_contrast(added_output_filename + "{}/".format(index), img_list, best_location, (index == 0), (index == 2), 0, order_id) #所有子图计算最小对比度的最佳位置
                    location_result[row][col] = best_location #存储最佳位置
                    img_list.clear()

                    # 按照每个子图的最佳位置叠加成完整的事件图
                    output.event_merge(sub_event_filename + 'sub_event_{}.png'.format(index), best_location, merged_output_filename)


            #output.img2red(merged_output_filename)
            output.fullevent_merge(merged_output_filename, canny_filename, result_canny_filename)
            output.fullevent_merge(merged_output_filename, rgb_filename, result_rgb_filename)

            time_end = time.time() #记录结束时间

            time_sum = time_end - time_start
            with open("/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/result/best_location_{}.csv".format(order_id), "a+", encoding='utf8', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # 写行数据
                writer.writerow(["contrast"])
                img = cv2.imread(result_canny_filename)
                writer.writerow([contrast.contrast(img)])
                csvfile.close()
            with open("/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/result/best_location_{}.csv".format(order_id), "a+", encoding='utf8', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # 写行数据
                writer.writerow(["time"])
                writer.writerow([time_sum])
                csvfile.close()
            print("总共运行时间为: {}".format(time_sum))

    shutil.copytree('/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/result', '/data/sjzhang/RRER/dataset/pre_dataset/result/{}'.format(i))









