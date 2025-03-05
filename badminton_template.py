import numpy as np
import cv2

class BadmintonCourtTemplate:
    """
    羽毛球场模版模型,
    单位: 厘米
    """

    def __init__(self):
        self.boundary_line_top = np.array(((2, 2), (48, 2), (305, 2), (562, 2), (608, 2)))
        self.long_service_line_top = np.array(((2, 78), (48, 78), (305, 78), (562, 78), (608, 78)))
        self.short_service_line_top = np.array(((2, 470), (48, 470), (305, 470), (562, 470), (608, 470)))
        self.short_service_line_bottom = np.array(((2, 870), (48, 870), (305, 870), (562, 870), (608, 870)))
        self.long_service_line_bottom = np.array(((2, 1262), (48, 1262), (305, 1262), (562, 1262), (608, 1262)))
        self.boundary_line_bottom = np.array(((2, 1338), (48, 1338), (305, 1338), (562, 1338), (608, 1338)))
        self.predefined_points = np.array((*self.boundary_line_top, *self.long_service_line_top,
                                           *self.short_service_line_top, *self.short_service_line_bottom,
                                           *self.long_service_line_bottom, *self.boundary_line_bottom))
        self.line_width = 4
        self.court_width = 610
        self.court_height = 1340

        self.top_boarder = self.bottom_boarder = self.left_boarder = self.right_boarder = 100
        self.base_shift = np.array((self.left_boarder, self.top_boarder))
        self.court_total_width = self.court_width + self.left_boarder + self.right_boarder
        self.court_total_height = self.court_height + self.top_boarder + self.bottom_boarder

        self._build_court_reference()
        self._build_annot_points(parts=10)

        #  convert gray to RGB, in green (0, 255, 0)
        self.court = np.dstack((np.zeros_like(self.court), self.court * 255, np.zeros_like(self.court)))

    def _build_court_reference(self):
        """
        Create court reference image using the lines positions
        """
        court = np.zeros((self.court_total_height, self.court_total_width), dtype=np.uint8)
        base_shift = self.base_shift
        cv2.line(court, self.boundary_line_top[0] + base_shift, self.boundary_line_top[-1] + base_shift, 1, 1)
        cv2.line(court, self.long_service_line_top[0] + base_shift, self.long_service_line_top[-1] + base_shift, 1, 1)
        cv2.line(court, self.short_service_line_top[0] + base_shift, self.short_service_line_top[-1] + base_shift, 1, 1)
        cv2.line(court, self.short_service_line_bottom[0] + base_shift, self.short_service_line_bottom[-1] + base_shift, 1, 1)
        cv2.line(court, self.long_service_line_bottom[0] + base_shift, self.long_service_line_bottom[-1] + base_shift, 1, 1)
        cv2.line(court, self.boundary_line_bottom[0] + base_shift, self.boundary_line_bottom[-1] + base_shift, 1, 1)
        cv2.line(court, self.boundary_line_top[0] + base_shift, self.boundary_line_bottom[0] + base_shift, 1, 1)
        cv2.line(court, self.boundary_line_top[1] + base_shift, self.boundary_line_bottom[1] + base_shift, 1, 1)
        cv2.line(court, self.boundary_line_top[3] + base_shift, self.boundary_line_bottom[3] + base_shift, 1, 1)
        cv2.line(court, self.boundary_line_top[4] + base_shift, self.boundary_line_bottom[4] + base_shift, 1, 1)
        cv2.line(court, self.boundary_line_top[2] + base_shift, self.short_service_line_top[2] + base_shift, 1, 1)
        cv2.line(court, self.short_service_line_bottom[2] + base_shift, self.boundary_line_bottom[2] + base_shift, 1, 1)
        court = cv2.dilate(court, np.ones((self.line_width, self.line_width), dtype=np.uint8))
        self.court = court

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

    def draw_court_on_warped_image(self, warped_image, color=(0, 255, 0)):
        assert warped_image.shape[0] == self.court.shape[0]
        assert warped_image.shape[1] == self.court.shape[1]
        base_shift = self.base_shift
        cv2.line(warped_image, self.boundary_line_top[0] + base_shift, self.boundary_line_top[-1] + base_shift, 1, 1)
        cv2.line(warped_image, self.long_service_line_top[0] + base_shift, self.long_service_line_top[-1] + base_shift, 1, 1)
        cv2.line(warped_image, self.short_service_line_top[0] + base_shift, self.short_service_line_top[-1] + base_shift, 1, 1)
        cv2.line(warped_image, self.short_service_line_bottom[0] + base_shift, self.short_service_line_bottom[-1] + base_shift, 1, 1)
        cv2.line(warped_image, self.long_service_line_bottom[0] + base_shift, self.long_service_line_bottom[-1] + base_shift, 1, 1)
        cv2.line(warped_image, self.boundary_line_bottom[0] + base_shift, self.boundary_line_bottom[-1] + base_shift, 1, 1)
        cv2.line(warped_image, self.boundary_line_top[0] + base_shift, self.boundary_line_bottom[0] + base_shift, 1, 1)
        cv2.line(warped_image, self.boundary_line_top[1] + base_shift, self.boundary_line_bottom[1] + base_shift, 1, 1)
        cv2.line(warped_image, self.boundary_line_top[3] + base_shift, self.boundary_line_bottom[3] + base_shift, 1, 1)
        cv2.line(warped_image, self.boundary_line_top[4] + base_shift, self.boundary_line_bottom[4] + base_shift, 1, 1)
        cv2.line(warped_image, self.boundary_line_top[2] + base_shift, self.short_service_line_top[2] + base_shift, 1, 1)
        cv2.line(warped_image, self.short_service_line_bottom[2] + base_shift, self.boundary_line_bottom[2] + base_shift, 1, 1)

courtTemplate = BadmintonCourtTemplate()
predefined_corners = {i: corner * 0.01 for i, corner in enumerate(courtTemplate.predefined_points)}
annot_points = courtTemplate.annot_points * 0.01
line_segs = [[0, 4], [0, 25], [4, 29], [25, 29], [5, 9], [10, 14], [15, 19], [20, 24], [25, 29], [1, 26], [2, 12], [3, 28], [17, 27]]
predefined_cube1 = np.array([predefined_corners[29], predefined_corners[4], predefined_corners[25], predefined_corners[0], predefined_corners[29], predefined_corners[4], predefined_corners[25], predefined_corners[0]])
predefined_cube2 = np.array([predefined_corners[18], predefined_corners[13], predefined_corners[16], predefined_corners[11], predefined_corners[18], predefined_corners[13], predefined_corners[16], predefined_corners[11]])
predefined_cube1 = np.hstack((predefined_cube1, np.array([[0], [0], [0], [0], [1.55], [1.55], [1.55], [1.55]])))
predefined_cube2 = np.hstack((predefined_cube2, np.array([[0], [0], [0], [0], [1.55], [1.55], [1.55], [1.55]])))
# predefined_cube3 = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
predefined_z_points = np.array([[0.02, 6.7, 1.55], [6.08, 6.7, 1.55]])

def draw_axis_on_img(K, d, rvec, tvec, img):
    origin = cv2.projectPoints(np.array([0., 0., 0.]), rvec, tvec, K, d)[0].reshape(2).astype(int)
    x = cv2.projectPoints(np.array([1., 0., 0.]), rvec, tvec, K, d)[0].reshape(2).astype(int)
    y = cv2.projectPoints(np.array([0., 1., 0.]), rvec, tvec, K, d)[0].reshape(2).astype(int)
    z = cv2.projectPoints(np.array([0., 0., 1.]), rvec, tvec, K, d)[0].reshape(2).astype(int)
    cv2.arrowedLine(img, origin, x, (0, 0, 255), 3)
    cv2.arrowedLine(img, origin, y, (0, 255, 0), 3)
    cv2.arrowedLine(img, origin, z, (255, 0, 0), 3)


def draw_cube_on_img(cube_pts, K, d, rvec, tvec, img, color=(255, 0, 0)):
    cube_pts = cube_pts.astype(np.float32)
    cube_2d, _ = cv2.projectPoints(cube_pts, rvec, tvec, K, d)
    cube_2d = cube_2d.reshape(-1, 2).astype(int)
    # for pt in cube_2d:
    #     cv2.circle(img, tuple(pt), 5, (0, 0, 255), -1)
    try:
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
    except:
        print('Error when drawing cv2.line, probably due to an invalid value.')



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
