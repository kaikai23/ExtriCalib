import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('images/template.jpg')
window_name = 'Image'
color = (0, 0, 255)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (0, 0, 255)
font_thickness = 1

def getEquidistantPoints(p1, p2, parts=10):
    res = np.concatenate(
        (np.linspace(p1[0], p2[0], parts + 1)[:, None], np.linspace(p1[1], p2[1], parts + 1)[:, None]),
        axis=1)
    return res

def getGridPoints(p_tl, p_br, parts=10):
    x, y = np.meshgrid(np.linspace(p_tl[0], p_br[0], parts), np.linspace(p_tl[1], p_br[1], parts))
    x = x.reshape(parts ** 2)
    y = y.reshape(parts ** 2)
    return np.array([(xx, yy) for xx, yy in zip(x,y)])


def getEquidistantPointsArcExcludeEndPoints(center, radius, parts=10):
    points = []
    for i in range(parts):
        theta = i * (np.pi / (parts - 1))  # Angle ranges from 0 to pi
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        points.append((x, y))
    points.pop(-1)
    points.pop(0)
    return np.array(points)

# predefined_corners = {
#     0: [84, 109],  # 00: top-left
#     1: [875, 109],  # 01: top-right
#     2: [84, 688],  # 02: bottom-left
#     3: [875, 688],  # 03: bottom-right
#     4: [351, 109],  # 04: mid-top-left
#     5: [608, 109],  # 05: mid-top-right
#     6: [351, 415],  # 06: mid-bottom-left
#     7: [608, 415],  # 07: mid-bottom-right
#     8: [412, 175],  # 08: arc-left
#     9: [546, 175],  # 09: arc-right
#     10: [340, 203],  # 10: mid-left-cross1
#     11: [340, 251],  # 11: mid-left-cross2
#     12: [340, 272],  # 12: mid-left-cross3
#     13: [340, 321],  # 13: mid-left-cross4
#     14: [340, 369],  # 14: mid-left-cross5
#     15: [619, 203],  # 15: mid-right-cross1
#     16: [619, 252],  # 16: mid-right-cross2
#     17: [619, 272],  # 17: mid-right-cross3
#     18: [619, 321],  # 18: mid-right-cross4
#     19: [619, 369],  # 19: mid-right-cross5
# }
predefined_corners = {
    0: [0.025, 0.025],  # 00: top-left
    1: [15.075, 0.025],  # 01: top-right
    2: [0.025, 11.075],  # 02: bottom-left
    3: [15.075, 11.075],  # 03: bottom-right
    4: [5.125, 0.025],  # 04: mid-top-left
    5: [9.975, 0.025],  # 05: mid-top-right
    6: [5.125, 5.825],  # 06: mid-bottom-left
    7: [9.975, 5.825],  # 07: mid-bottom-right
    8: [6.275, 1.25],  # 08: arc-left
    9: [8.825, 1.25],  # 09: arc-right
    10: [5.125, 1.825],  # 10: mid-left-cross1
    11: [5.125, 2.7],  # 11: mid-left-cross2
    12: [5.125, 3.1],  # 12: mid-left-cross3
    13: [5.125, 3.975],  # 13: mid-left-cross4
    14: [5.125, 4.875],  # 14: mid-left-cross5
    15: [9.975, 1.825],  # 15: mid-right-cross1
    16: [9.975, 2.7],  # 16: mid-right-cross2
    17: [9.975, 3.1],  # 17: mid-right-cross3
    18: [9.975, 3.975],  # 18: mid-right-cross4
    19: [9.975, 4.875],  # 19: mid-right-cross5
    20: [0.975, 0],  # 20
    21: [14.125, 0]  # 21
}

line_segs = [[0, 1], [1, 3], [3, 2], [2, 0], [4, 6], [6, 7], [7, 5]]

predefined_cube1 = np.array([predefined_corners[7], predefined_corners[5], predefined_corners[6], predefined_corners[4],
                            predefined_corners[7], predefined_corners[5], predefined_corners[6], predefined_corners[4]])
predefined_cube2 = np.array([predefined_corners[3], predefined_corners[1], predefined_corners[2], predefined_corners[0],
                            predefined_corners[3], predefined_corners[1], predefined_corners[2], predefined_corners[0]])

predefined_cube1 = np.hstack((predefined_cube1, np.array([[0], [0], [0], [0], [3.05], [3.05], [3.05], [3.05]])))
predefined_cube2 = np.hstack((predefined_cube2, np.array([[0], [0], [0], [0], [3.05], [3.05], [3.05], [3.05]])))
predefined_z_points = np.array([[7.5, 1.625, 3.05]])


def draw_cube_on_img(cube_pts, K, d, rvec, tvec, img, color=(255, 0, 0)):
    cube_pts = cube_pts.astype(np.float32)
    cube_2d, _ = cv2.projectPoints(cube_pts, rvec, tvec, K, d)
    cube_2d = cube_2d.reshape(-1, 2).astype(int)
    # for pt in cube_2d:
    #     cv2.circle(img, tuple(pt), 5, (0, 0, 255), -1)
    cv2.line(img, cube_2d[0], cube_2d[1], color, 1)
    cv2.line(img, cube_2d[0], cube_2d[2], color, 1)
    cv2.line(img, cube_2d[0], cube_2d[4], color, 1)
    cv2.line(img, cube_2d[1], cube_2d[3], color, 1)
    cv2.line(img, cube_2d[1], cube_2d[5], color, 1)
    cv2.line(img, cube_2d[2], cube_2d[3], color, 1)
    cv2.line(img, cube_2d[2], cube_2d[6], color, 1)
    cv2.line(img, cube_2d[3], cube_2d[7], color, 1)
    cv2.line(img, cube_2d[4], cube_2d[5], color, 1)
    cv2.line(img, cube_2d[4], cube_2d[6], color, 1)
    cv2.line(img, cube_2d[5], cube_2d[7], color, 1)
    cv2.line(img, cube_2d[6], cube_2d[7], color, 1)


def draw_distorted_cube_on_img(cube_pts, K, d, rvec, tvec, img, map1, map2, new_K, old_K, old_d, color=(255, 0, 0)):
    def draw_distorted_line(original_image, pt1, pt2):
        pt1 = np.array(pt1, dtype=np.float32).reshape(1, 1, 2)
        pt2 = np.array(pt2, dtype=np.float32).reshape(1, 1, 2)
        undistorted_pts = cv2.undistortPoints(np.concatenate([pt1, pt2], axis=0), old_K, old_d, P=new_K)
        pt1_undistorted = undistorted_pts[0][0]
        pt2_undistorted = undistorted_pts[1][0]
        num_points = 100
        line_points_undistorted = np.linspace(pt1_undistorted, pt2_undistorted, num_points).reshape(-1, 2)
        x_undist = line_points_undistorted[:, 0]
        y_undist = line_points_undistorted[:, 1]
        x_dist = map1[np.clip(y_undist.astype(int), 0, map1.shape[0] - 1), np.clip(x_undist.astype(int), 0, map1.shape[1] - 1)]
        y_dist = map2[np.clip(y_undist.astype(int), 0, map2.shape[0] - 1), np.clip(x_undist.astype(int), 0, map2.shape[1] - 1)]
        line_points_distorted = np.stack((x_dist, y_dist), axis=1)
        # line_points_distorted = cv2.projectPoints(np.concatenate((line_points_undistorted, np.ones((line_points_undistorted.shape[0], 1, 1))), axis=2),
        #                                           np.zeros((3,)), np.zeros((3,)), np.eye(3), d)[0][:, 0, :]
        for i in range(len(line_points_distorted) - 1):
            pt1 = tuple(line_points_distorted[i].astype(int))
            pt2 = tuple(line_points_distorted[i + 1].astype(int))
            cv2.line(original_image, pt1, pt2, color, 2)
        return original_image

    cube_pts = cube_pts.astype(np.float32)
    cube_2d, _ = cv2.projectPoints(cube_pts, rvec, tvec, K, d)
    cube_2d = cube_2d.reshape(-1, 2).astype(int)
    draw_distorted_line(img, cube_2d[0], cube_2d[1])
    draw_distorted_line(img, cube_2d[0], cube_2d[2])
    draw_distorted_line(img, cube_2d[0], cube_2d[4])
    draw_distorted_line(img, cube_2d[1], cube_2d[3])
    draw_distorted_line(img, cube_2d[1], cube_2d[5])
    draw_distorted_line(img, cube_2d[2], cube_2d[3])
    draw_distorted_line(img, cube_2d[2], cube_2d[6])
    draw_distorted_line(img, cube_2d[3], cube_2d[7])
    draw_distorted_line(img, cube_2d[4], cube_2d[5])
    draw_distorted_line(img, cube_2d[4], cube_2d[6])
    draw_distorted_line(img, cube_2d[5], cube_2d[7])
    draw_distorted_line(img, cube_2d[6], cube_2d[7])


def draw_axis_on_img(K, d, rvec, tvec, img):
    origin = cv2.projectPoints(np.array([0., 0., 0.]), rvec, tvec, K, d)[0].reshape(2).astype(int)
    x = cv2.projectPoints(np.array([1., 0., 0.]), rvec, tvec, K, d)[0].reshape(2).astype(int)
    y = cv2.projectPoints(np.array([0., 1., 0.]), rvec, tvec, K, d)[0].reshape(2).astype(int)
    z = cv2.projectPoints(np.array([0., 0., 1.]), rvec, tvec, K, d)[0].reshape(2).astype(int)
    cv2.arrowedLine(img, origin, x, (0, 0, 255), 3)
    cv2.arrowedLine(img, origin, y, (0, 255, 0), 3)
    cv2.arrowedLine(img, origin, z, (255, 0, 0), 3)


predefined_lines = [[predefined_corners[0], predefined_corners[1]],
                    [predefined_corners[1], predefined_corners[2]],
                    [predefined_corners[2], predefined_corners[3]],
                    [predefined_corners[3], predefined_corners[0]],
                    [predefined_corners[4], predefined_corners[5]],
                    [predefined_corners[5], predefined_corners[7]],
                    [predefined_corners[7], predefined_corners[6]]]

center = [7.55, 1.625]
radius1 = 1.275
radius2 = 6.725
connect_line1 = ((center[0]-radius2, center[1]), (center[0]-radius2, 0.025))
connect_line2 = ((center[0]+radius2, center[1]), (center[0]+radius2, 0.025))

arc_points = []
# points.append(getGridPoints(predefined_corners[0], predefined_corners[3]))
arc_points.append(getEquidistantPointsArcExcludeEndPoints(center, radius1, 4))
arc_points.append(getEquidistantPointsArcExcludeEndPoints(center, radius2, 21))
arc_points = np.concatenate(arc_points, axis=0)

annot_parts = 10
annot_points = []
# annot_points.append(np.array([predefined_corners[4], predefined_corners[5],
#                               predefined_corners[6], predefined_corners[7]]))
annot_points.append(getGridPoints(predefined_corners[0], predefined_corners[3], parts=annot_parts))
annot_points = np.concatenate(annot_points, axis=0)



radius = 1
thickness = 1

if __name__ == "__main__":
    a = np.zeros((2000,1800,3))
    annot_points *= 100
    annot_points += 100
    annot_points = annot_points.astype(int)

    def show(annot_points):
        a = np.zeros((2000, 1800, 3))
        for idx, pt in enumerate(annot_points):
            a = cv2.circle(a, pt, 8, (255,255,255), 2)
            text = str(idx)
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = pt[0] - text_size[0] // 2
            text_y = pt[1] - radius - 5  # Adjust the vertical position as needed
            a = cv2.putText(a, text, (text_x, text_y), font, 2, (255,255,255), 2)
        plt.imshow(a)
        plt.show()
    show(annot_points)

    idxs = np.meshgrid(np.linspace(0, annot_parts-1, annot_parts), np.linspace(0, annot_parts-1, annot_parts))
    flip_idx0 = np.flip(idxs[0], -1)
    idxs = flip_idx0 + idxs[1] * annot_parts
    idxs = idxs.reshape(annot_parts**2, ).astype(int)
    print(idxs)
    annot_points = annot_points[idxs]
    show(annot_points)
    # for idx, coord in predefined_corners.items():
    #     image = cv2.circle(image, coord, radius, color, thickness)
    #
    #     text = str(idx)
    #     text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    #     text_x = coord[0] - text_size[0] // 2
    #     text_y = coord[1] - radius - 5  # Adjust the vertical position as needed
    #     image = cv2.putText(image, text, (text_x, text_y), font, font_scale, font_color, font_thickness)
    #
    #
    # cv2.imwrite('images/template_corners1.png', image)


