import cv2
import os
import numpy as np
from utils.camera_utils import FileStorage, read_camera
import matplotlib.pyplot as plt
from badminton_template import predefined_corners, annot_points, predefined_cube1, predefined_cube2, draw_cube_on_img, \
	line_segs, draw_axis_on_img


def transform(points, homography):
	points = np.asarray(points)
	if homography is not None:
		res = cv2.perspectiveTransform(np.asarray(points).reshape((-1, 1, 2)).astype(np.float32), homography)
		out_pts: np.ndarray = res.reshape(points.shape).astype(int)
		return out_pts
	return None


def draw_court_on_frame(frame, K, d, Rvec, tvec, h):
	draw_pts = {id: transform(pt, h) for id, pt in predefined_corners.items()}
	for line_id1, line_id2 in line_segs:
		cv2.line(frame, tuple(draw_pts[line_id1]), tuple(draw_pts[line_id2]), (255, 0, 0), 2)
	draw_cube_on_img(predefined_cube1[:, [1, 0, 2]], K, d, Rvec, tvec, frame, color=(0, 0, 255))
	draw_cube_on_img(predefined_cube2[:, [1, 0, 2]], K, d, Rvec, tvec, frame, color=(0, 255, 255))
	draw_axis_on_img(K, d, Rvec, tvec, frame)
	return None

if __name__ == "__main__":
	frame = cv2.imread('/Users/yifei/Desktop/Research/z-/game_images/S--a--01--TY29.jpg')
	cameras = read_camera(intri_name='/Users/yifei/Desktop/Research/z-/game_camparam/intri_TY29.yml',
						  extri_name='/Users/yifei/Desktop/Research/z-/game_camparam/extri_TY29.yml')
	cameras.pop('basenames')
	h = np.load('/Users/yifei/Desktop/Research/z-/save_to/h_S--a--01--TY29.jpg.npy')
	draw_court_on_frame(frame, K=cameras['01']['K'], d=cameras['01']['dist'], Rvec=cameras['01']['Rvec'], tvec=cameras['01']['T'], h=h)
	plt.imshow(frame[:, :, ::-1])
	plt.show()
	pass
