import numpy as np
import cv2

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
