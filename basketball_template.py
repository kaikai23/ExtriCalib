import cv2
import numpy as np
import matplotlib.pyplot as plt

class CourtTemplate():
    """
    球场模版模型,
    单位: 厘米
    """

    def __init__(self):
        self.baseline_top = np.array(((2.5, 2.5), (1507.5, 2.5)))
        self.baseline_bottom = np.array(((2.5, 1107.5), (1507.5, 1107.5)))
        self.net = np.array(((286, 1748), (1379, 1748)))
        self.left_court_line = np.array(((2.5, 2.5), (2.5, 1107.5)))
        self.right_court_line = np.array(((1507.5, 2.5), (1507.5, 1107.5)))
        self.left_inner_line = np.array(((512.5, 2.5), (512.5, 582.5)))
        self.right_inner_line = np.array(((997.5, 2.5), (997.5, 582.5)))
        self.bottom_inner_line = np.array(((512.5, 582.5), (997.5, 582.5)))
        self.left_3point_line = np.array(((97.5, 2.5), (97.5, 303.75)))
        self.right_3point_line = np.array(((1412.5, 2.5), (1412.5, 303.75)))
        self.basket_center = np.array((755, 162.5))
        self.three_point_radius = 672.5
        self.three_point_ellipse_axes = (672.5, 672.5)
        self.three_point_arc_startangle = 12.124
        self.three_point_arc_endangle = 167.876
        self.key_points = np.array([*self.baseline_top, *self.baseline_bottom,
                                    *self.left_inner_line, *self.right_inner_line,
                                    self.left_3point_line[0], self.right_3point_line[0]])
        self.predefined_points = np.array((*self.baseline_top, *self.baseline_bottom,
                                           self.left_inner_line[0], self.right_inner_line[0], *self.bottom_inner_line,
                                           ))
        self.line_width = 5
        self.court_width = 1510
        self.court_height = 1110
        self.top_boarder = 195
        self.bottom_boarder = 95
        self.left_boarder = self.right_boarder = 145
        self.base_shift = np.array((self.left_boarder, self.top_boarder))
        self.court_total_width = self.court_width + self.left_boarder + self.right_boarder
        self.court_total_height = self.court_height + self.top_boarder + self.bottom_boarder

        self._build_court_reference()
        self._build_annot_points(parts=10)
        self._build_masks()

        #  convert gray to RGB, (0, 255, 0)
        self.court = np.dstack((np.zeros_like(self.court), self.court * 255, np.zeros_like(self.court)))

    def _build_court_reference(self):
        """
        Create court reference image using the lines positions
        """
        court = np.zeros((self.court_total_height, self.court_total_width), dtype=np.uint8)
        base_shift = self.base_shift
        cv2.line(court, *(self.baseline_top + base_shift).astype(int), 1, 1)
        cv2.line(court, *(self.baseline_bottom + base_shift).astype(int), 1, 1)
        cv2.line(court, *(self.left_court_line + base_shift).astype(int), 1, 1)
        cv2.line(court, *(self.right_court_line + base_shift).astype(int), 1, 1)
        cv2.line(court, *(self.left_inner_line + base_shift).astype(int), 1, 1)
        cv2.line(court, *(self.right_inner_line + base_shift).astype(int), 1, 1)
        cv2.line(court, *(self.bottom_inner_line + base_shift).astype(int), 1, 1)
        cv2.line(court, *(self.left_3point_line + base_shift).astype(int), 1, 1)
        cv2.line(court, *(self.right_3point_line + base_shift).astype(int), 1, 1)
        cv2.ellipse(court, (self.basket_center + base_shift).astype(int),
                    [int(x) for x in self.three_point_ellipse_axes],
                    0, self.three_point_arc_startangle, self.three_point_arc_endangle, 1, 1)
        court = cv2.dilate(court, np.ones((self.line_width, self.line_width), dtype=np.uint8))
        self.court = court

    def _build_annot_points(self, parts=10):
        """
        Create annotation points by uniformly sampling the court area
        """
        self.annot_points = self._getGridPoints(self.baseline_top[0], self.baseline_bottom[1], parts=parts)
        self.annot_points += self.base_shift

    def _build_masks(self):
        self.mask_all = self._create_whole_mask()
        self.mask_3points = self._create_three_points_mask()
        self.mask_inner_square = self._create_inner_square_mask()

    def _getGridPoints(self, p_tl, p_br, parts=10):
        x, y = np.meshgrid(np.linspace(p_tl[0], p_br[0], parts), np.linspace(p_tl[1], p_br[1], parts))
        x = x.reshape(parts ** 2)
        y = y.reshape(parts ** 2)
        return np.array([(xx, yy) for xx, yy in zip(x, y)])

    def _create_whole_mask(self):
        court = np.zeros((self.court_total_height, self.court_total_width), dtype=np.uint8)
        cv2.line(court, *(self.baseline_top + self.base_shift).astype(int), 1, 1)
        cv2.line(court, *(self.baseline_bottom + self.base_shift).astype(int), 1, 1)
        cv2.line(court, *(self.left_court_line + self.base_shift).astype(int), 1, 1)
        cv2.line(court, *(self.right_court_line + self.base_shift).astype(int), 1, 1)
        return self.fill_contours(court)

    def _create_three_points_mask(self):
        court = np.zeros((self.court_total_height, self.court_total_width), dtype=np.uint8)
        cv2.line(court, (self.left_3point_line[0] + self.base_shift).astype(int),
                 (self.right_3point_line[0] + self.base_shift).astype(int), 1, 1)
        cv2.line(court, *(self.left_3point_line + self.base_shift).astype(int), 1, 1)
        cv2.line(court, *(self.right_3point_line + self.base_shift).astype(int), 1, 1)
        cv2.ellipse(court, (self.basket_center + self.base_shift).astype(int),
                    [int(x) for x in self.three_point_ellipse_axes],
                    0, self.three_point_arc_startangle, self.three_point_arc_endangle, 1, 1)
        return self.fill_contours(court)

    def _create_inner_square_mask(self):
        court = np.zeros((self.court_total_height, self.court_total_width), dtype=np.uint8)
        cv2.line(court, (self.left_inner_line[0] + self.base_shift).astype(int),
                 (self.right_inner_line[0] + self.base_shift).astype(int), 1, 1)
        cv2.line(court, (self.left_inner_line[1] + self.base_shift).astype(int),
                 (self.right_inner_line[1] + self.base_shift).astype(int), 1, 1)
        cv2.line(court, *(self.left_inner_line + self.base_shift).astype(int), 1, 1)
        cv2.line(court, *(self.right_inner_line + self.base_shift).astype(int), 1, 1)
        return self.fill_contours(court)

    def fill_contours(self, arr):
        return np.maximum.accumulate(arr, 1) & \
               np.maximum.accumulate(arr[:, ::-1], 1)[:, ::-1]

    def draw_court_on_warped_image(self, warped_image, color=(0, 255, 0)):
        assert warped_image.shape[0] == self.court.shape[0]
        assert warped_image.shape[1] == self.court.shape[1]
        base_shift = self.base_shift
        cv2.line(warped_image, *(self.baseline_top + base_shift).astype(int), color, self.line_width)
        cv2.line(warped_image, *(self.baseline_bottom + base_shift).astype(int), color, self.line_width)
        cv2.line(warped_image, *(self.left_court_line + base_shift).astype(int), color, self.line_width)
        cv2.line(warped_image, *(self.right_court_line + base_shift).astype(int), color, self.line_width)
        cv2.line(warped_image, *(self.left_inner_line + base_shift).astype(int), color, self.line_width)
        cv2.line(warped_image, *(self.right_inner_line + base_shift).astype(int), color, self.line_width)
        cv2.line(warped_image, *(self.bottom_inner_line + base_shift).astype(int), color, self.line_width)
        cv2.line(warped_image, *(self.left_3point_line + base_shift).astype(int), color, self.line_width)
        cv2.line(warped_image, *(self.right_3point_line + base_shift).astype(int), color, self.line_width)
        cv2.ellipse(warped_image, (self.basket_center + base_shift).astype(int),
                    [int(x) for x in self.three_point_ellipse_axes],
                    0, self.three_point_arc_startangle, self.three_point_arc_endangle, color, self.line_width)

courtTemplate = CourtTemplate()


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



