import numpy as np
import cv2

class PingPangTableTemplate:
    """
    乒乓球球桌模版模型,
    单位: 厘米
    """

    def __init__(self):
        self.boundary_line_top = np.array(((0, 0), (152.5, 0)))
        self.boundary_line_bottom = np.array(((0, 274), (152.5, 274)))
        self.middle_line = np.array(((-15.25, 137), (167.75, 137), (0, 137), (152.5, 137)))
        self.longitudinal_line = np.array(((76.25, 0), (76.25, 274)))
        self.predefined_points = np.array((*self.boundary_line_top,
                                           *self.boundary_line_bottom,
                                           *self.middle_line,
                                           *self.longitudinal_line,
                                           ))

        self.top_boarder = self.bottom_boarder = self.left_boarder = self.right_boarder = 0
        self.base_shift = np.array((self.left_boarder, self.top_boarder))

        self._build_annot_points(parts=10)

    def _build_annot_points(self, parts=10):
        """
        Create annotation points by uniformly sampling the court area
        """
        self.annot_points = self._getGridPoints(self.boundary_line_top[0], self.boundary_line_bottom[-1], parts=parts)

    def _getGridPoints(self, p_tl, p_br, parts=10):
        x, y = np.meshgrid(np.linspace(p_tl[0], p_br[0], parts), np.linspace(p_tl[1], p_br[1], parts))
        x = x.reshape(parts ** 2)
        y = y.reshape(parts ** 2)
        return np.array([(xx, yy) for xx, yy in zip(x, y)])


courtTemplate = PingPangTableTemplate()
predefined_corners = {i: corner * 0.01 for i, corner in enumerate(courtTemplate.predefined_points)}
annot_points = courtTemplate.annot_points * 0.01
line_segs = [[0, 8], [8, 1], [0, 6], [6, 2], [1, 7], [7, 3], [2, 9], [9, 3], [4, 5], [8, 9]]
predefined_cubes = []
predefined_cube1 = np.array([predefined_corners[7], predefined_corners[1], predefined_corners[6], predefined_corners[0], predefined_corners[7], predefined_corners[1], predefined_corners[6], predefined_corners[0]])
predefined_cube2 = np.array([predefined_corners[3], predefined_corners[7], predefined_corners[2], predefined_corners[6], predefined_corners[3], predefined_corners[7], predefined_corners[2], predefined_corners[6]])
predefined_cube3 = np.array([predefined_corners[5], predefined_corners[5], predefined_corners[4], predefined_corners[4], predefined_corners[5], predefined_corners[5], predefined_corners[4], predefined_corners[4]])
predefined_cube1 = np.hstack((predefined_cube1, np.array([[0], [0], [0], [0], [0.1525], [0.1525], [0.1525], [0.1525]])))
predefined_cube2 = np.hstack((predefined_cube2, np.array([[0], [0], [0], [0], [0.1525], [0.1525], [0.1525], [0.1525]])))
predefined_cube3 = np.hstack((predefined_cube3, np.array([[0], [0], [0], [0], [0.1525], [0.1525], [0.1525], [0.1525]])))
predefined_cubes.append(predefined_cube1)
predefined_cubes.append(predefined_cube2)
predefined_cubes.append(predefined_cube3)
predefined_z_points = np.array([])
arc_points = None

def draw_axis_on_img(K, d, rvec, tvec, img):
    origin = cv2.projectPoints(np.array([0., 0., 0.]), rvec, tvec, K, d)[0].reshape(2).astype(int)
    x = cv2.projectPoints(np.array([1., 0., 0.]), rvec, tvec, K, d)[0].reshape(2).astype(int)
    y = cv2.projectPoints(np.array([0., 1., 0.]), rvec, tvec, K, d)[0].reshape(2).astype(int)
    z = cv2.projectPoints(np.array([0., 0., 1.]), rvec, tvec, K, d)[0].reshape(2).astype(int)
    cv2.arrowedLine(img, origin, x, (0, 0, 255), 3)
    cv2.arrowedLine(img, origin, y, (0, 255, 0), 3)
    cv2.arrowedLine(img, origin, z, (255, 0, 0), 3)


def draw_cube_on_img(cube_pts, K, d, rvec, tvec, img, color=(255, 0, 0), line_width=2):
    cube_pts = cube_pts.astype(np.float32)
    cube_2d, _ = cv2.projectPoints(cube_pts, rvec, tvec, K, d)
    cube_2d = cube_2d.reshape(-1, 2).astype(int)
    # for pt in cube_2d:
    #     cv2.circle(img, tuple(pt), 5, (0, 0, 255), -1)
    cv2.line(img, cube_2d[0], cube_2d[1], color, line_width)
    cv2.line(img, cube_2d[0], cube_2d[2], color, line_width)
    cv2.line(img, cube_2d[0], cube_2d[4], color, line_width)
    cv2.line(img, cube_2d[1], cube_2d[3], color, line_width)
    cv2.line(img, cube_2d[1], cube_2d[5], color, line_width)
    cv2.line(img, cube_2d[2], cube_2d[3], color, line_width)
    cv2.line(img, cube_2d[2], cube_2d[6], color, line_width)
    cv2.line(img, cube_2d[3], cube_2d[7], color, line_width)
    cv2.line(img, cube_2d[4], cube_2d[5], color, line_width)
    cv2.line(img, cube_2d[4], cube_2d[6], color, line_width)
    cv2.line(img, cube_2d[5], cube_2d[7], color, line_width)
    cv2.line(img, cube_2d[6], cube_2d[7], color, line_width)

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


# import matplotlib.pyplot as plt
# from copy import deepcopy
# court_img = courtTemplate.court
# for i, pt in predefined_corners.items():
#     pt = pt + courtTemplate.base_shift
#     cv2.circle(court_img, pt, 5, (255, 0, 0), -1)
#     cv2.putText(court_img, f'{i}', pt, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 2)
# plt.imshow(courtTemplate.court)
# plt.savefig('./images/badminton_corners.png')
# plt.show()
# for pt in courtTemplate.annot_points:
#     courtTemplate.court = cv2.circle(courtTemplate.court, tuple(pt.astype(int)), 10, (0, 255, 0), -1)
# plt.imshow(courtTemplate.court)
# plt.show()