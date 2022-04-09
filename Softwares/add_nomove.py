import cv2
import numpy as np



def subgraph_add_nomove(sub_event_filename, canny_filename, location, added_output_filename, zoom_flag):
    """
    生成多张事件数据子图与canny图叠加的图片,输入
    :param sub_event_filename: 事件数据子图路径
    :param canny_filename: canny图路径
    :param location: 事件数据叠加位置, 确定的位置
    :param added_output_filename： 叠加图像的输出路径
    :return: None
    """
    # 位置参数
    each_row = location[0]
    each_col = location[1]
    print("正在叠加event与canny数据, row: {} , col: {}".format(each_row, each_col))

    # 1. 读取图片
    img1 = cv2.imread(canny_filename) # 读取canny图片
    img2 = cv2.imread(sub_event_filename)  # 读取事件子图
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGBA)  # 转为RGBA模式
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGBA)  # 转为RGBA模式
    #对事件子图进行透明化处理
    white = np.array([0, 0, 0, 255]) #定义白色
    rows, cols = img2.shape[:2]
    for row in range(rows):
        for col in range(cols):
            # 将像素点与白色像素做对比
            if (img2[row][col] == white).all():
                # 将白色像素换成透明
                img2[row][col] = np.array([0, 0, 0, 0])
    #canny图边缘扩充，方便子图叠加
    img1 = cv2.copyMakeBorder(img1, 150, 150, 150, 150, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # 2. 提取感兴趣区域roi
    if (zoom_flag):
        img2 = cv2.resize(img2, (0, 0), fx=2.12, fy=2.12, interpolation=cv2.INTER_NEAREST) #event子图放大
    rows, cols = img2.shape[:2]
    print("canny图大小： {}".format(img1.shape))
    print("event子图大小： {}".format(img2.shape))

    roi = img1[each_row:each_row + rows, each_col:each_col + cols] #在canny图中寻找到与event子图相同大小的roi位置进行叠加， each_row, each_col为叠加位置

    # 3. 创建掩膜mask
    img2gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)  # 将图片灰度化，如果在读取logo时直接灰度化，该步骤可省略
    # cv2.THRESH_BINARY：如果一个像素值低于200，则像素值转换为255（白色色素值），否则转换成0（黑色色素值）
    # 即有内容的地方为黑色0，无内容的地方为白色255.
    # 白色的地方还是白色，除了白色的地方全变成黑色
    ret, mask = cv2.threshold(img2gray, 175, 255, cv2.THRESH_BINARY)  # 阙值操作
    mask_inv = cv2.bitwise_not(mask)  # 与mask颜色相反，白色变成黑色，黑变白

    # 4. logo与感兴趣区域roi融合
    # 保留除logo外的背景
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)
    dst = cv2.add(img1_bg, img2_fg)  # logo与感兴趣区域roi进行融合
    img1[each_row:each_row + rows, each_col:each_col + cols] = dst  # 将融合后的区域放进原图

    img_new_add = img1.copy()  # 对处理后的图像进行拷贝
    img_new_add = img_new_add[150:1080+150:,150: 150+1440]
    # cv2.imwrite(added_output_filename + "row_{}_col_{}.png".format(each_row, each_col), img_new_add)
    img_new_add = cv2.cvtColor(img_new_add, cv2.COLOR_RGBA2BGRA)  # 转为BGRA模式
    cv2.imwrite(added_output_filename, img_new_add)
    print("event与canny数据叠加完成, row: {} , col: {}".format(each_row, each_col))
    cv2.waitKey(0)
