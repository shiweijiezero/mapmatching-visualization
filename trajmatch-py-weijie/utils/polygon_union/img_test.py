# 轮廓
import numpy as np
import cv2

def get_Contours(file_name, k, width, padding):
    v = int(k * width)

    img = cv2.imread(file_name)
    # img = cv2.imread('X:\\OpenCVtest\\003.jpg')
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 加边框padding
    padded_img = cv2.copyMakeBorder(imgray, padding, padding, padding, padding, cv2.BORDER_CONSTANT,
                                    value=255)  # 上下左右边缘扩充200个像素点

    kernel = np.ones((v, v), np.uint8)

    e_img = cv2.erode(padded_img, kernel, iterations=1)
    cv2.imwrite('erode_reuslt.png', img=e_img)

    d_img = cv2.dilate(e_img, kernel, iterations=1)
    cv2.imwrite('dilate_reuslt.png', img=d_img)

    ret, thresh = cv2.threshold(d_img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img = cv2.drawContours(d_img, contours, -1, 128, 3)
    cv2.imwrite('corridor_reuslt.png', img=img)


    # with open("hierarchy","w") as f2:
    #     f2.write(hierarchy.__str__())

    return contours, imgray, hierarchy


def get_gps_contours(contours, high_per_pixel, width_per_pixel,
                     most_left, most_up, img_shape, padding,start=1):
    gps_contours = []


    for i in range(start,len(contours)):
        contour_temp = []
        temp = contours[i]
        for j in range(len(temp)):
            point = [(temp[j][0][0] - padding) * width_per_pixel + most_left,
                     most_up - (temp[j][0][1] - padding) * high_per_pixel]

            contour_temp.append(point)
        else:
            gps_contours.append(contour_temp)

    # with open("contours","w") as f1:
    #     f1.write(gps_contours.__str__())

    return gps_contours




