from .vector import D2Vector
from .vertex import Vertex
from .utils import make_D2Vector, get_rectangle
from .visualization import draw_segments, draw_rectangles, show, save, init
from .img_test import get_Contours, get_gps_contours
import matplotlib.pyplot as plt


def draw_contours(mee_matched_lines, dpi, k, width=50):
    # mee_matched_lines应该就是 [((lng,lat),(lng,lat)),((lng,lat),(lng,lat))...]这样的格式，p是(lng,lat这样)

    # segments = [[[0,0],[1,1]],
    # 			[[1,1],[2,1]],
    # 			[[2,1],[3,4]],
    # 			[[5,5],[5,6]],
    # 			[[8,8],[8,2]],
    # 			[[8,1],[7,-2]]]

    plt.close()
    segments = mee_matched_lines
    padding = 500
    rectangle_points = []
    most_left = most_right = most_up = most_low = None

    # 得到矩形
    for segment in segments:
        points = get_rectangle(segment, width)
        for point in points:
            if most_left is None or point[0] < most_left:
                most_left = point[0]
            if most_right is None or point[0] > most_right:
                most_right = point[0]
            if most_up is None or point[1] > most_up:
                most_up = point[1]
            if most_low is None or point[1] < most_low:
                most_low = point[1]
        rectangle_points.append(points)

    for points in rectangle_points:
        draw_rectangles(points, round(most_left, 12),
                        round(most_right, 12),
                        round(most_low, 12),
                        round(most_up, 12))

    ##画线段
    # draw_segments(segments)
    save("result", dpi)
    # show()

    ans = {"rectangles": rectangle_points, "left-top": [round(most_left, 12), round(most_up, 12)],
           "right-bottom": [round(most_right, 12), round(most_low, 12)]}
    # print(ans)
    contours, dst_img, hierarchy = get_Contours("result.png", k=k, width=width, padding=padding)
    img_shape = dst_img.shape

    high = most_up - most_low
    width = most_right - most_left

    high_per_pixel = high / img_shape[0]
    width_per_pixel = width / img_shape[1]

    # print("pixel Contours:")
    # print(contours)

    gps_contours = get_gps_contours(contours, high_per_pixel, width_per_pixel,
                                    most_left, most_up, img_shape, padding,start=1)
    gps_contours_outer = get_gps_contours(contours, high_per_pixel, width_per_pixel,
                                    most_left, most_up, img_shape, padding,start=0)

    # print("gps_contours:")
    # print(gps_contours)
    corridor_lines = []
    for i in range(len(gps_contours)):
        for j in range(len(gps_contours[i]) - 1):
            corridor_lines.append([gps_contours[i][j], gps_contours[i][j + 1]])

        if (len(gps_contours[i]) > 0):
            corridor_lines.append([gps_contours[i][-1], gps_contours[i][0]])

    return corridor_lines,gps_contours_outer,hierarchy


# segments = [[[0,0],[1,1]],
# 			[[1,1],[2,1]],
# 			[[2,1],[3,4]],
# 			[[5,5],[5,6]],
# 			[[8,8],[8,2]],
# 			[[8,1],[7,-2]]]

# draw_contours(segments)
