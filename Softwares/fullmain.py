import add
import add_nomove
import contrast
import output
import cv2
import csv
import numpy as np
import time
import test_full
import os
import shutil


def copy(src_dir, dst_dir):
    for root, _, fnames in os.walk(src_dir):
      for fname in sorted(fnames):  # sorted函数把遍历的文件按文件名排序
        fpath = os.path.join(root, fname)
        shutil.copy(fpath, dst_dir)  # 完成文件拷贝
        print(fname + " has been copied!")

# event_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/event/event_"
# rgb_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/rgb/rgb_"
# #sub_event_filename = "../Img/sub_event/" #多张事件子图路径
# canny_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/canny/canny_" #canny图
# #added_output_filename = "../Img/full_added/" #叠加图片输出位置
# #merged_output_filename = "../Img/full_result/merged_output_" #最终输出位置
# result_canny_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/full_result/result_canny_"
# result_rgb_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/full_result/result_rgb_"

# test_full.init_full()

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
    # sub_event_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/sub_event/"  # 多张事件子图路径
    canny_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/canny/canny_"  # canny图
    # added_output_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/full_added/"  # 叠加图片输出位置
    # merged_output_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/full_result/merged_output_"  # 最终输出位置
    result_canny_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/full_result/result_canny_"
    result_rgb_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/full_result/result_rgb_"

    #数据集生成循环，清除并且重新复制
    shutil.rmtree('/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/')
    shutil.copytree('/data/sjzhang/RRER/dataset/pre_dataset/Img', '/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img')

    for (src_dir, dst_dir) in zip(src_list, dst_list):
        copy(src_dir + '{}/'.format(i), dst_dir)

    file_list = os.listdir(event_filename[:-6])
    test_full.init_full(len(file_list))
    print(len(file_list))

    for order_id in range(len(file_list)):
        event_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/event/event_"
        rgb_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/rgb/rgb_"
        sub_event_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/sub_event/"  # 多张事件子图路径
        canny_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/canny/canny_"  # canny图
        added_output_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/full_added/"  # 叠加图片输出位置
        merged_output_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/full_result/merged_output_"  # 最终输出位置
        result_canny_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/full_result/result_canny_"
        result_rgb_filename = "/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/full_result/result_rgb_"

        event_filename += '{}.png'.format(order_id)
        rgb_filename += '{}.png'.format(order_id)
        canny_filename += '{}.png'.format(order_id)
        merged_output_filename += '{}.png'.format(order_id)
        result_canny_filename += '{}.png'.format(order_id)
        result_rgb_filename += '{}.png'.format(order_id)

        time_start = time.time() #记录运行时间
        img_list = []#存储图片, 不直接存储照片以提升速度
        best_location = [272, 155] #第一个图的默认最优位置

        with open("/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/full_result/contrast_{}.csv".format(order_id), "w", encoding='utf8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # 写行数据
            writer.writerow(["contrast_result"])
            csvfile.close()

        with open("/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/full_result/best_location_{}.csv".format(order_id), "w", encoding='utf8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # 写行数据
            writer.writerow(["best_row", "best_col"])
            csvfile.close()

        if __name__ == "__main__":

            img_list = add.subgraph_add(event_filename, canny_filename, best_location, added_output_filename, 1, 0)  # 叠加子图
            best_location = contrast.cal_contrast(added_output_filename,img_list, best_location, 1, 0, 1, order_id)  # 事件数据计算最小对比度的最佳位置
            img_list.clear()
            # 按照事件数据的最佳位置叠加成完整的事件图,包含：canny、rgb
            output.event_merge(event_filename, best_location, merged_output_filename)
            output.fullevent_merge(merged_output_filename, canny_filename, result_canny_filename)
            output.fullevent_merge(merged_output_filename, rgb_filename, result_rgb_filename)

            time_end = time.time() #记录结束时间

            time_sum = time_end - time_start
            with open("/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/full_result/best_location_{}.csv".format(order_id), "a+", encoding='utf8', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # 写行数据
                writer.writerow(["contrast"])
                img = cv2.imread(result_canny_filename)
                writer.writerow([contrast.contrast(img)])
                csvfile.close()

            with open("/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/full_result/best_location_{}.csv".format(order_id), "a+", encoding='utf8', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # 写行数据
                writer.writerow(["time"])
                writer.writerow([time_sum])
                csvfile.close()
            print("总共运行时间为: {}".format(time_sum))

    shutil.copytree('/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/full_result', '/data/sjzhang/RRER/dataset/pre_dataset/full_result/{}'.format(i))