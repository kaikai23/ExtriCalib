from typing import Optional, Tuple
import cv2
import numpy as np
from .opencv_utils import MousePointsClick, OpenCVWindow
import tkinter as tk


def get_h_from_images(
    image: np.ndarray,
    predefined_corners,
    pts_size: int = 5,
    color: Tuple[int, int, int] = (255, 0, 0),
    num_rect_pts=4,
) -> Optional[np.ndarray]:

    min_required_pts = 4
    pts1 = []  # video frame points
    pts2 = []  # template points
    pt_ids = []  # chosen template point ids

    root = tk.Tk()
    root.title("Choose point id in template")
    label = tk.Label(root, text="Choose correspondence point in template", font=("Helvetica", 14))
    label.pack()
    button_list = []
    num_buttons = len(predefined_corners)
    global clicked_num
    def on_button_click(key):
        global clicked_num
        button = button_list[key]
        clicked_num = int(button['text'])
        print(f'{clicked_num} is clicked')
        pt_ids.append(clicked_num)
        root.quit()
        button.pack_forget()  # Hide the clicked button
        # buttons_alive.remove(key)
    global NEXT_IMAGE
    NEXT_IMAGE = False
    def next_image():
        global NEXT_IMAGE
        NEXT_IMAGE = True
        root.quit()
        root.destroy()

    for key in range(num_buttons):
        button = tk.Button(root, text=str(key), command=lambda k=key: on_button_click(k))
        button.pack()
        button_list.append(button)
    button2 = tk.Button(root, text='next image', command=next_image)
    button2.pack()
    for _ in range(num_rect_pts):
        root.mainloop()
        if NEXT_IMAGE:
            return None, None, None, None
        pts2.append(predefined_corners[clicked_num])
        frame_calibration_window = OpenCVWindow("Image")
        correspondences_1 = MousePointsClick(
            frame_calibration_window, 1)
        correspondences_1.get_points(
            frame_calibration_window, image, pts_size=pts_size, color=color
        )
        pts1.append(correspondences_1.points[0])
    root.destroy()

    if len(pts1) < min_required_pts:
        print(
            f"Not enough points selected for window {frame_calibration_window.opencv_winname}. Exiting"
        )
        return None
    else:
        print(f"Points from image 1: {pts1}")

    if len(pts2) < min_required_pts:
        print(
            f"Not enough points selected for window {frame_calibration_window.opencv_winname}. Exiting"
        )
    else:
        print(f"Points from image 2: {pts2}")

    # Get H from points
    pts1 = np.asarray(pts1, dtype=np.float32)
    pts2 = np.asarray(pts2, dtype=np.float32)

    if len(pts1) == min_required_pts and len(pts2) == min_required_pts:
        h0 = cv2.getPerspectiveTransform(pts2, pts1)
    else:
        # with ransac
        h0, _ = cv2.findHomography(pts2, pts1, method=0)

    return h0, pts1, pts2, pt_ids


def transform(points, homography):
    if homography is not None:
        res = cv2.perspectiveTransform(
            np.asarray(points).reshape(
                (-1, 1, 2)).astype(np.float32), homography
        )
        out_pts: np.ndarray = res.reshape(points.shape).astype(np.int)
        return out_pts
    return None


def cameraCalibFromHomography(H, u0, v0):
    """
    Assume fx = fy, camera intri f, and extrinsic R, T can be recovered from a single view.
    If fx ≠ fy, only the scale from fx to fy can be found.
    Parameters
    ----------
    H: Homography (3x3)
    u0: principle point (1, )
    v0: principle point(1, )

    Returns
    rvec: (3x1)
    tvec: (3x1)
    -------

    """
    h11 = H[0 ,0]
    h12 = H[0, 1]
    h21 = H[1, 0]
    h22 = H[1, 1]
    h31 = H[2, 0]
    h32 = H[2, 1]
    f = np.sqrt(((h12-u0*h32)**2 + (h22-v0*h32)**2 - (h11-u0*h31)**2 - (h21-v0*h31)**2 ) / (h31**2 - h32**2))
    K_inv = np.array([[1/f, 0, -u0/f], [0, 1/f, -v0/f], [0, 0, 1]])
    R_col1col2_T = K_inv @ H
    r1 = R_col1col2_T[:, 0:1]
    norm = np.linalg.norm(r1)
    r1 = r1 / norm
    r2 = R_col1col2_T[:, 1:2]
    r2 = r2 / np.linalg.norm(r2)
    r3 = np.cross(r1, r2, axis=0)
    tvec = R_col1col2_T[:, 2:] / norm
    R = np.hstack((r1, r2, r3))
    rvec = cv2.Rodrigues(R)[0]
    return rvec, tvec
