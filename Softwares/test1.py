import os
import shutil

# rgb_src_dir = '/data/sjzhang/RRER/dataset/pre_dataset/image_data/'
# event_src_dir = '/data/sjzhang/RRER/dataset/pre_dataset/event_data/'
# canny_src_dir = '/data/sjzhang/RRER/dataset/pre_dataset/canny_data/'
#
# rgb_dst_dir = '/data/sjzhang/RRER/dataset/pre_dataset/image_data/'
# event_dst_dir = '/data/sjzhang/RRER/dataset/pre_dataset/event_data/'
# canny_dst_dir = '/data/sjzhang/RRER/dataset/pre_dataset/canny_data/' Z:\data\sjzhang\RRER\dataset\pre_dataset\temp_data\Img\rgb

src_list = ['/data/sjzhang/RRER/dataset/pre_dataset/image_data/', '/data/sjzhang/RRER/dataset/pre_dataset/event_data/', '/data/sjzhang/RRER/dataset/pre_dataset/canny_data/']
dst_list = ['/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/rgb/', '/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/event/', '/data/sjzhang/RRER/dataset/pre_dataset/temp_data/Img/canny/']

def copy(src_dir, dst_dir):
    for root, _, fnames in os.walk(src_dir):
      for fname in sorted(fnames):  # sorted函数把遍历的文件按文件名排序
        fpath = os.path.join(root, fname)
        shutil.copy(fpath, dst_dir)  # 完成文件拷贝
        print(fname + " has been copied!")


if __name__ == '__main__':
    i = 0
    for (src_dir, dst_dir) in zip(src_list, dst_list):
        copy(src_dir + '{}/'.format(i), dst_dir)

