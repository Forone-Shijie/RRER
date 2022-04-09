import cv2
import numpy as np

def init(num):
    img1 = np.full((1080, 1440, 3), fill_value=255, dtype='uint8')
    for i in range (num):
        cv2.imwrite('/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/result/merged_output_{}.png'.format(i), img1)

