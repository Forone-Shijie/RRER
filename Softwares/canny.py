import cv2
import os

rootdir = "../Img/rgb/"
savedir = "../Img/canny/"

index = 0
for root, dirs, files in os.walk(rootdir):
    for filename in files:
        index += 1
        print(filename)
        original_image = cv2.imread(rootdir + filename, 1)
        cv2.imshow('original_image', original_image)

        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray_image', gray_image)
        print(gray_image)

        gaussian_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
        cv2.imshow('gaussian_image', gaussian_image)

        canny_image = cv2.Canny(gaussian_image, 20, 60)
        cv2.imshow('canny_image', canny_image)

        cv2.imwrite(savedir + 'canny_{}.png'.format(index), canny_image)
        # cv2.waitKey(0)
        # # cv2.destroyAllWindows()
