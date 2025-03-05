# SPORTS = 'basketball'
# SPORTS = 'badminton'
SPORTS = 'pingpang'

import sys, os
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    application_path = sys._MEIPASS
else:
    application_path = os.path.dirname(os.path.abspath(__file__))
print(f'{application_path=}')
import argparse
import cv2
import numpy as np
from utils.homography_utils import get_h_from_images, transform
import tkinter as tk
if SPORTS == 'basketball':
    from predefined_corners import predefined_corners, arc_points, annot_points, predefined_cube1, predefined_cube2, draw_cube_on_img, line_segs, draw_axis_on_img, predefined_z_points, draw_distorted_cube_on_img
    template_plane = application_path + '/images/template_corners.png'
    from basketball_court_template import courtTemplate
elif SPORTS == 'badminton':
    from badminton_template import predefined_corners, arc_points, annot_points, predefined_cube1, predefined_cube2, draw_cube_on_img, line_segs, draw_axis_on_img, predefined_z_points
    arc_points = np.array([])
    template_plane = application_path + '/images/badminton_corners.png'
    from badminton_template import courtTemplate
elif SPORTS == 'pingpang':
    from pingpang_template import predefined_corners, arc_points, annot_points, predefined_cubes, draw_cube_on_img, line_segs, draw_axis_on_img, predefined_z_points, draw_distorted_cube_on_img
    template_plane = application_path + '/images/pingpang_corners.png'
from scipy.optimize import minimize

import glob
import re
import tqdm
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))


from utils.camera_utils import FileStorage
from copy import deepcopy

if SPORTS == 'basketball':
    H_meter_to_centimeter_and_base_shift = cv2.getPerspectiveTransform(
        np.array([[0.025, 0.025], [15.075, 0.025],
                  [0.025, 11.075], [15.075, 11.075]], dtype=np.float32),
        np.array([[2.5, 2.5], [1507.5, 2.5],
                  [2.5, 1107.5], [1507.5, 1107.5]], dtype=np.float32) + np.array((145., 195.), dtype=np.float32)
    )
elif SPORTS == 'badminton':
    H_meter_to_centimeter_and_base_shift = cv2.getPerspectiveTransform(
        np.array([[0.02, 0.02], [6.08, 0.02],
                  [0.02, 13.38], [6.08, 13.38]], dtype=np.float32),
        np.array([[2, 2], [608, 2],
                  [2, 1338], [608, 1338]], dtype=np.float32) + np.array((100., 100.), dtype=np.float32)
    )
elif SPORTS == 'pingpang':
    H_meter_to_centimeter_and_base_shift = cv2.getPerspectiveTransform(
        np.array([[0, 0], [1.525, 0],
                  [0, 2.74], [1.525, 2.74]], dtype=np.float32),
        np.array([[0, 0], [152.5, 0],
                  [0, 274], [152.5, 274]], dtype=np.float32) + np.array((100., 100.), dtype=np.float32)
    )



def transform(points, homography):
    points = np.asarray(points)
    if homography is not None:
        res = cv2.perspectiveTransform(
            np.asarray(points).reshape((-1, 1, 2)).astype(np.float32), homography
        )
        out_pts: np.ndarray = res.reshape(points.shape).astype(int)
        return out_pts
    return None


def reproj_error(x, points3d, points2d):
    """ used for scipy.optimize.minimize """
    assert len(x) == 15
    fx, fy, cx, cy = x[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    d = x[4:9].reshape(1, 5)
    rvec = x[9:12].reshape(3, )
    tvec = x[12:15].reshape(1, 3)
    proj_pts2d = cv2.projectPoints(points3d, rvec, tvec, K, d)[0].reshape(points2d.shape)
    return np.sqrt(((points2d - proj_pts2d) ** 2).sum(-1)).sum()  # MSE


def compute_RTd_from_fixedK(image_points, object_points, K, d, resolution=(1920, 1080), method=1):
    image_points = image_points.astype(np.float32)
    object_points = object_points.astype(np.float32)
    K_ = deepcopy(K)
    d_ = deepcopy(d)
    if method == 0:  # estimate K, d, rvec, tvec
        assert K is None and d is None
        rms, K_, d_, rvec, tvec = cv2.calibrateCamera(object_points.reshape(1, -1, 3), image_points.reshape(1, -1, 2),
                                                    resolution, None, None)
        rvec = rvec[0]
        tvec = tvec[0]
    elif method == 1:  # optimize rvec, tvec, and distortion
        rms, K_, d_, rvec, tvec = cv2.calibrateCamera(object_points.reshape(1, -1, 3),
                                                      image_points.reshape(1, -1, 2), resolution,
                                                      K_, d_,
                                                      # None, None,
                                                      flags=cv2.CALIB_USE_INTRINSIC_GUESS
                                                            + cv2.CALIB_FIX_PRINCIPAL_POINT
                                                            + cv2.CALIB_FIX_FOCAL_LENGTH
                                                            # + cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6
                                                            # + cv2.CALIB_FIX_S1_S2_S3_S4
                                                            # + cv2.CALIB_FIX_TAUX_TAUY
                                                      )
        rvec = rvec[0]
        tvec = tvec[0]
        assert (K == K_).all()
        if not (K == K_).all():
            K = K_
            print(f'K changed, K={K}')
        if not (d == d_).all():
            d = d_
            print(f'd changed, d={d}')
    elif method == 2:  # optimize rvec and tvec
        ret, rvec, tvec = cv2.solvePnP(object_points, image_points, K, distCoeffs=d)
        if ret is not True:
            print('Error: PnP fails.')
            return None, None, d
    else:
        raise NotImplementedError
    return K_, d_, rvec, tvec

def yolobbox2bbox(yolobbox):
    x, y, w, h = yolobbox
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return np.array([x1, y1, x2, y2])

def bbox2yolobbox(bbox):
    x1, y1, x2, y2 = bbox
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return np.array([x, y, w, h])


def main(args):
    pattern_img = img = cv2.imread("./2024-summer-olympics-summer-olympic-games-paris-paralympic-games-paris-1a43a5452ac5e594c5838d3719273f28.png", cv2.IMREAD_UNCHANGED)
    # Our Sony
    # A = np.array([[1665.185, 0.000, 900.682], [0.000, 1669.339, 482.780], [0.000, 0.000, 1.000]])
    # d = np.array([0.038, -0.063, -0.016, -0.009, 0.000])
    # HK 01
    # A = np.array([[1190.588429, 0.000000, 638.752197], [0.000000, 1193.261599, 370.206628], [0.000000, 0.000000, 1.000000]])
    # d = np.array([0.260055, -1.156552, -0.003080, -0.000129, 0.000000])
    # HK 01
    # A = np.array([[1607.540004, 0.000000, 951.666319], [0.000000, 1618.957960, 541.034859], [0.000000, 0.000000, 1.000000]])
    # d = np.array([0.167417, -0.841929, -0.010231, 0.000609, 0.000000])
    # HK 011
    # A = np.array([[2368.316843, 0.000000, 979.728509], [0.000000, 2369.335610, 551.095322], [0.000000, 0.000000, 1.000000]])
    # d = np.array([0.183906, -0.772709, 0.004902, 0.004141, 0.000000])
    # HK 02
    # A = np.array([[1770.510274, 0.000000, 610.787811], [0.000000, 1793.453895, 378.412656], [0.000000, 0.000000, 1.000000]])
    # d = np.array([0.302733, -1.669009, -0.007565, -0.008188, 0.000000])
    # d = np.array([0., 0., 0., 0., 0.])
    # HK 03
    # A = np.array([[1776.285610, 0.000000, 568.529545], [0.000000, 1796.687371, 373.358316], [0.000000, 0.000000, 1.000000]])
    # d = np.array([0.265612, -0.818744, -0.010422, -0.021752, 0.000000])
    # HK 03
    # A = np.array([[2521.454126, 0.000000, 848.174331], [0.000000, 2463.960134, 474.762322], [0.000000, 0.000000, 1.000000]])
    # d = np.array([0.229715, -1.043221, -0.010522, -0.020102, 0.000000])
    # HK 031
    # A = np.array([[2249.602528, 0.000000, 979.015892], [0.000000, 2273.964966, 568.482705], [0.000000, 0.000000, 1.000000]])
    # d = np.array([0.226443, -1.130630, -0.001476, 0.002800, 0.000000])
    # HK 04
    # A = np.array([[1049.815412, 0.000000, 610.495696], [0.000000, 1038.442621, 296.089499], [0.000000, 0.000000, 1.000000]])
    # d = np.array([0.013008, 0.024371, -0.007840, -0.007817, 0.000000])
    # HK 04
    # A = np.array([[1711.088061, 0.000000, 950.036290], [0.000000, 1710.126883, 407.124657], [0.000000, 0.000000, 1.000000]])
    # d = np.array([-0.051384, 1.170450, -0.018059, 0.002315, 0.000000])

    # 01
    # A = np.array([[4986.828, 0.000, 1120.826], [0.000, 4983.206, 784.349], [0.000, 0.000, 1.000]])
    # d = np.array([-0.438, 0.308, -0.004, 0.002, 0.000])

    # 02
    # A = np.array([[4809.657, 0.000, 954.318], [0.000, 4811.186, 565.969], [0.000, 0.000, 1.000]])
    # d = np.array([-0.481, 1.427, -0.002, -0.001, 0.000])

    # 03
    # A = np.array([[5843.372, 0.000, 1035.574], [0.000, 5852.271, 660.748], [0.000, 0.000, 1.000]])
    # d = np.array([-0.655, 2.158, -0.010, -0.008, 0.000])

    # 04
    # A = np.array([[5728.166, 0.000, 1053.162], [0.000, 5737.774, 618.119], [0.000, 0.000, 1.000]])
    # d = np.array([-0.638, 2.620, -0.001, -0.008, 0.000])

    # 05
    # A = np.array([[6159.727, 0.000, 747.682], [0.000, 6170.912, 704.519], [0.000, 0.000, 1.000]])
    # d = np.array([-0.000, -1.993, 0.011, -0.007, 0.000])

    # HK court-calibrated 01
    # A = np.array([[1607.540004, 0.000000, 951.666319], [0.000000, 1618.957960, 541.034859], [0.000000, 0.000000, 1.000000]])
    # d = np.array([0.167417, -0.841929, -0.010231, 0.000609, 0.000000])

    # HK court-calibrated 02
    # A = np.array([[2268.115, 0.000000, 958.077], [0.000000, 2215.973, 484.028], [0.000000, 0.000000, 1.000000]])
    # d = np.array([7.12661454e-03, -3.00857508e-02, -6.97557299e-05, 4.18023672e-03, 2.84115651e-02])

    # HK court-calibrated 03
    # A = np.array([[2521.454126, 0.000000, 848.174331], [0.000000, 2463.960134, 474.762322], [0.000000, 0.000000, 1.000000]])
    # d = np.array([0.229715, -1.043221, -0.010522, -0.020102, 0.000000])

    # HK court-calibrated 04
    # A = np.array([[1405.455, 0, 971.399], [0, 1366.042, 565.276], [0, 0, 1]])
    # d = np.array([-2.73318004e-03,  1.13196964e-02,  3.16243012e-05,  2.46444567e-03, -2.05885167e-02])

    # 畸变 13.jpg
    # A = np.array([[2298.565, 0.000, 1890.145], [0.000, 2310.545, 1007.662], [0.000, 0.000, 1.000]])
    # d = np.array([-0.356, 0.093, 0.001, 0.002, 0.000])

    # 畸变 14.jpg
    # A = np.array([[2314.974, 0.000, 1903.618], [0.000, 2323.658, 1118.859], [0.000, 0.000, 1.000]])
    # d = np.array([-0.373, 0.106, -0.004, 0.001, 0.000])

    # Hungary 00
    # A = np.array([[2631.383602, 0.000000, 654.712990], [0.000000, 2631.525464, 933.734921], [0.000000, 0.000000, 1.000000]])
    # d = np.array([0.270719, -1.486459, -0.004432, 0.023437, 0.000000])

    # HUngary 01
    # A = np.array([[2974.730884, 0.000000, 684.111305], [0.000000, 2988.992050, 827.621365], [0.000000, 0.000000, 1.000000]])
    # d = np.array([0.305068, -0.999809, -0.025084, 0.028039, 0.000000])

    # Hungary 00 and 01: calibrated by this tool
    # A = np.array([[2760.041321, 0.000000, 943.692204], [0.000000, 2700.149122, 477.896512], [0.000000, 0.000000, 1.000000]])
    # d = np.array([-0.007690, 0.010348, 0.000508, -0.002196, 0.028867])

    # A = None
    # d = None
    A = args.K
    d = args.d
    if args.read_intri_from_file:
        intri_file_abspath = args.read_intri_from_file
        assert os.path.exists(intri_file_abspath)
        camname, camparam_fname = os.path.splitext(args.img_base)[0].split('_')
        intri = FileStorage(intri_file_abspath, isWrite=False)
        assert camname in intri.read('names', dt='list')
        A = intri.read('K_{}'.format(camname))
        d = intri.read('dist_{}'.format(camname))
        # d[:] = 0
        intri.__del__()
        print(f'Read intrinsic from {intri_file_abspath}\n {A=}\n {d=}')



    camera = args.input_dir + '/' + args.img_base
    assert os.path.isfile(camera), f"File not found: {camera}"
    assert os.path.isfile(template_plane), f"File not found: {template_plane}"

    floor = cv2.imread(template_plane)

    wname = "frame_" + args.img_base
    wname_p = "template"

    cap = cv2.VideoCapture(camera)

    assert cap.isOpened(), f'Unable to open the camera: "{camera}"!'

    ret, frame = cap.read()
    H, W = frame.shape[:2]
    if not ret:
        raise NotImplementedError()
    # if args.undistort:
    #     newcameramtx, roi = cv2.getOptimalNewCameraMatrix(A, d, (W, H), 1, (W, H))
    #     frame_undist = cv2.undistort(frame, A, d, None, newcameramtx)
    #     frame = frame_undist
    #     A = newcameramtx
    #     d = np.array([0., 0., 0., 0., 0.])
    if args.undistort:
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(A, d, (W, H), 1, (W, H))
        oldcameramtx = A.copy()
        old_d = d.copy()
        print(f'{oldcameramtx=}\n, {old_d=}')
        frame_undist = cv2.undistort(frame, A, d, None, newcameramtx)
        # frame = frame_undist
        # A = newcameramtx
        # d = np.array([0., 0., 0., 0., 0.])
        map1, map2 = cv2.initUndistortRectifyMap(A, d, None, newcameramtx, (W, H), cv2.CV_32FC1)

    # Function to map points from undistorted frame to original frame
    def map_points_to_original(points_undist, map1, map2):
        # Convert points to float32
        points_undist = np.array(points_undist, dtype=np.float32)

        # Extract x and y coordinates
        x_undist = points_undist[:, 0]
        y_undist = points_undist[:, 1]

        # Map x and y coordinates using the reverse mapping
        x_orig = map1[np.clip(y_undist.astype(int), 0, map1.shape[0]-1), np.clip(x_undist.astype(int), 0, map1.shape[1]-1)]
        y_orig = map2[np.clip(y_undist.astype(int), 0, map2.shape[0]-1), np.clip(x_undist.astype(int), 0, map2.shape[1]-1)]
        # x_orig = cv2.remap(map1, x_undist, y_undist, interpolation=cv2.INTER_LINEAR)
        # y_orig = cv2.remap(map2, x_undist, y_undist, interpolation=cv2.INTER_LINEAR)

        # Combine x and y coordinates back into a single array
        points_orig = np.stack((x_orig, y_orig), axis=1)

        return points_orig


    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)

    save_dir = args.save_annot_to
    h_file = f"{save_dir}/h_{args.img_base}.npy"
    pts1_file = f"{save_dir}/pts1_{args.img_base}.npy"
    pts2_file = f"{save_dir}/pts2_{args.img_base}.npy"
    pt_ids_file = f"{save_dir}/pt_ids_{args.img_base}.npy"
    yolotxt_file = f"{save_dir}/{args.img_base}.txt"
    # camera_name, camparam_fname = args.img_base.split('--')[2:4]
    # camera_name = '0'
    # camparam_fname = args.img_base
    camera_name, camparam_fname = args.img_base.split('_')
    camname, camparam_fname = os.path.splitext(args.img_base)[0].split('_')
    extri_fname = 'extri_' + camparam_fname + '.yml'
    intri_fname = 'intri_' + camparam_fname + '.yml'
    draw_pts = {}

    h = pts1 = pts2 = pt_ids = None
    if os.path.isfile(h_file):
        h = np.load(h_file)
        pts1 = np.load(pts1_file)
        pts2 = np.load(pts2_file)
        pt_ids = list(np.load(pt_ids_file))
        print('Loaded an annotated homography')
    elif args.init_extri_from_preceding_img and args.PRE_img_base is not None:
        PRE_h_file = f"{save_dir}/h_{args.PRE_img_base}.npy"
        PRE_pts1_file = f"{save_dir}/pts1_{args.PRE_img_base}.npy"
        PRE_pts2_file = f"{save_dir}/pts2_{args.PRE_img_base}.npy"
        PRE_pt_ids_file = f"{save_dir}/pt_ids_{args.PRE_img_base}.npy"
        if os.path.isfile(PRE_h_file):
            h = np.load(PRE_h_file)
            pts1 = np.load(PRE_pts1_file)
            pts2 = np.load(PRE_pts2_file)
            pt_ids = list(np.load(PRE_pt_ids_file))
            print('Loaded homography from preceding image')

    ori_frame = frame.copy()
    ori_frame_undist = frame_undist.copy() if args.undistort else None
    global REFINE_STATUS, SHIFT_SPEED
    REFINE_STATUS = False
    show_points_status = 0  # 0: markable points, 1: annot points
    SHIFT_SPEED = 5
    TOPDOWN_VIEW = False
    UNDIST_VIEW = False
    CONVERGE = False
    METHOD = 2  # 0: estimate K, d, R, T;   # 1: calibrate dist + RT;   2: PnP (only R,T);
    if METHOD == 0:
        A = None
        d = None
    USE_Z_POINT = 0  # 0: off;,   # 1: use z point 1;   # 2: use z point 2;
    pts1_z_points = {}


    while True:

        key = cv2.waitKey(10)

        if h is not None:
            draw_pts = {id: transform(pt, h) for id, pt in predefined_corners.items()}

        if h is None and key == -1:
            h, pts1, pts2, pt_ids = get_h_from_images(frame, predefined_corners, num_rect_pts=4) if not args.undistort else \
                                    get_h_from_images(frame_undist, predefined_corners, num_rect_pts=4)
            if h is None and pt_ids is None:
                break
            print(f'points1: {pts1}, points2: {pts2}')
            print(f'Homography is {h}')

        elif key == ord('m'):
            root = tk.Tk()
            root.title("Set maxmimun shift speed (default is 20 pixels)")
            button_list = []
            button_nums = [5, 20, 50, 100, 300, 3000, 30000, 3000000]
            def on_button_click(key):
                global SHIFT_SPEED
                SHIFT_SPEED= key
                root.quit()
            for key in button_nums:
                button = tk.Button(root, text=str(key), command=lambda k=key: on_button_click(k))
                button.pack()
                button_list.append(button)
            root.mainloop()
            root.destroy()
            print(f'SHIFT_SPEED = {SHIFT_SPEED}')

        elif key == ord('r'):
            CONVERGE = 1 - CONVERGE
            shift = np.array([0, 0])
            print(f'CONVERGE set to {CONVERGE}')
        elif key == ord('z'):
            USE_Z_POINT = (USE_Z_POINT + 1) % (len(predefined_z_points) + 1)
            shift = np.array([0, 0])
            print(f'USE_Z_POINT set to {USE_Z_POINT}')

        # Adjust points
        elif key == 13 or REFINE_STATUS:  # 13:"Enter"
            assert h is not None
            REFINE_STATUS = True
            root = tk.Tk()
            root.title("Choose point id to adjust")
            label = tk.Label(root, text="1. Click point id you want to adjuste\n "
                                        "2. Press ↑↓←→ arrows to adjust\n "
                                        "3. Press 'Enter' and click next point\n "
                                        "4. Click 'Finish' when you want to stop"
                                        , font=("Helvetica", 14))
            label.pack()
            button_list = []
            num_buttons = len(predefined_corners)
            global clicked_num
            def detect_esc_press(event_):
                global REFINE_STATUS
                REFINE_STATUS = False
                root.quit()
            def on_button_click(key):
                global clicked_num
                global REFINE_STATUS
                button = button_list[key]
                if button['text'] == 'Finish':
                    print('Finish refinement')
                    REFINE_STATUS = False
                else:
                    clicked_num = int(button['text'])
                    print(f'{clicked_num} is clicked')
                root.quit()
            # create buttons
            for key in range(num_buttons):
                button = tk.Button(root, text=str(key), command=lambda k=key: on_button_click(k))
                button.pack()
                button_list.append(button)
            button = tk.Button(root, text='Finish', command=lambda k=-1: on_button_click(k))
            button.pack()
            button_list.append(button)
            root.bind('<Escape>', detect_esc_press)
            root.mainloop()
            root.destroy()
            last_key = None
            multiplier = 1
            while True and REFINE_STATUS:
                key = cv2.waitKey(100)
                if last_key == key and key != -1:
                    multiplier = min(SHIFT_SPEED, multiplier * 2)
                else:
                    multiplier = 1
                last_key = key
                if key in [81, 2]:  # "←"
                    shift = np.array([-1, 0]) * multiplier
                elif key in [82, 0]:  # "↑"
                    shift = np.array([0, -1]) * multiplier
                elif key in [83, 3]:  # "→"
                    shift = np.array([1, 0]) * multiplier
                elif key in [84, 1]:  # "↓"
                    shift = np.array([0, 1]) * multiplier
                elif key == 13:  # "Enter"
                    print('next point')
                    break
                elif key == ord('v'):
                    TOPDOWN_VIEW = 1 - TOPDOWN_VIEW
                    shift = np.array([0, 0])
                elif key == ord('u'):
                    UNDIST_VIEW = 1 - UNDIST_VIEW
                    shift = np.array([0, 0])
                elif key == ord('r'):
                    CONVERGE = 1 - CONVERGE
                    shift = np.array([0, 0])
                    print(f'CONVERGE={CONVERGE}')
                else:
                    continue
                if clicked_num in pt_ids:
                    i = pt_ids.index(clicked_num)
                    pts1[i] = pts1[i] + shift
                else:
                    p2 = predefined_corners[clicked_num]
                    p1 = transform(p2, h)
                    p1 = p1 + shift
                    pts1 = np.vstack((pts1, p1))
                    pts2 = np.vstack((pts2, p2))
                    pt_ids = pt_ids + [clicked_num]
                pts1 = np.asarray(pts1, dtype=np.float32)
                pts2 = np.asarray(pts2, dtype=np.float32)
                if pts1.shape[0] == 4:
                    h = cv2.getPerspectiveTransform(pts2, pts1)
                else:
                    h, _ = cv2.findHomography(pts2, pts1, method=0)
                print(f'points1\n{repr(pts1)}\npoints2\n{repr(pts2)}\npt_ids\n{pt_ids}')
                print(f'Homography\n{h}')
                draw_pts = {id: transform(pt, h) for id, pt in predefined_corners.items()}
                overlay = ori_frame.copy() if not UNDIST_VIEW else ori_frame_undist.copy()
                for line_id1, line_id2 in line_segs:
                    cv2.line(overlay, tuple(draw_pts[line_id1]), tuple(draw_pts[line_id2]), (255, 0, 0), 2)
                alpha = 0.45
                frame = cv2.addWeighted(overlay, alpha, ori_frame, 1 - alpha, 0) if not UNDIST_VIEW else \
                        cv2.addWeighted(overlay, alpha, ori_frame_undist, 1 - alpha, 0)
                for id, pt in draw_pts.items():
                    cv2.putText(frame, f'{id}', pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.circle(frame, tuple(pt), 4, (0, 0, 255), 2)
                assert 'image_points' in locals() and 'object_points' in locals()
                valid_id = (0 < image_points[:, 0]) * (image_points[:, 0] < W) * (0 < image_points[:, 1]) * (image_points[:, 1] < H)
                image_points = image_points[valid_id]
                object_points = object_points[valid_id]
                image_points_final = map_points_to_original(image_points, map1, map2) if args.undistort else image_points
                if args.undistort:
                    for pt in image_points:
                        cv2.circle(frame, tuple(pt.astype(int)), 5, (0, 255, 0), -1) if UNDIST_VIEW else None
                    for pt in image_points_final:
                        cv2.circle(frame, tuple(pt.astype(int)), 5, (0, 255, 0), -1) if not UNDIST_VIEW else None
                A, d, rvec1, tvec1= compute_RTd_from_fixedK(image_points_final, object_points, A, d, method=METHOD)
                METHOD = 1 if A is not None and METHOD == 0 else METHOD
                for color_idx, predefined_cube in enumerate(predefined_cubes):
                    colors = [(255, 0, 255), (255, 255, 0), (0, 255, 255)]
                    draw_cube_on_img(predefined_cube[:, [1, 0, 2]], A, d, rvec1, tvec1, frame, color=colors[color_idx])
                draw_axis_on_img(A, d, rvec1, tvec1, overlay)
                for pt in pts1:
                    cv2.drawMarker(frame, tuple(pt.astype(int)), (0, 255, 0), markerType=cv2.MARKER_CROSS)
                if not TOPDOWN_VIEW:
                    cv2.imshow(wname, frame)
                else:
                    h_img2court = H_meter_to_centimeter_and_base_shift @ np.linalg.inv(h)
                    print(f'Homography (to court in cm)\n{h_img2court}')
                    frame_bev = cv2.warpPerspective(frame, h_img2court,
                                                    (courtTemplate.court.shape[1], courtTemplate.court.shape[0]))
                    courtTemplate.draw_court_on_warped_image(frame_bev)
                    # h1, w1 = frame.shape[:2]
                    # h2, w2 = frame_bev.shape[:2]
                    # frame_toshow = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
                    # frame_toshow[:h1, :w1, :] = frame
                    # frame_toshow[:h2, w1:, :] = frame_bev
                    cv2.imshow(wname, frame_bev)




                # A = np.array([[1920.00000, 0, 960.0], [0, 1920.00000, 540.0], [0, 0, 1]])
                # object_points = np.array(list(predefined_corners.values()))
                # object_points = np.concatenate((object_points, np.zeros((object_points.shape[0], 1))), axis=1).astype(np.float32)
                # image_points = np.array(list(draw_pts.values())).astype(np.float32)
                # ret, rvec1, tvec1 = cv2.solvePnP(object_points, image_points, A, np.zeros((4, 1)))
                # from utils.homography_utils import cameraCalibFromHomography
                # rvec2, tvec2 = cameraCalibFromHomography(h, 960, 540)
                # rms, K, dist, rvec3, tve3 = cv2.calibrateCamera(object_points.reshape(1, -1, 3),
                #                                              image_points.reshape(1, -1, 2), (1920, 1080), None, None)




                # rms, K, dist, rvec3, tvec3 = cv2.calibrateCamera(object_points.reshape(1, -1, 3),
                #                                              image_points.reshape(1, -1, 2), (1920, 1080), None, None)
                print(f'rvec1:{rvec1}\n tvec1: {tvec1}')
                # print(f'K: {repr(K)}\n  dist:{dist}\n rvec3:{rvec3}\n tvec3: {tvec3}')

        elif key == ord("c"):
            show_points_status = 1 - show_points_status
        elif key == ord("v"):
            TOPDOWN_VIEW = 1 - TOPDOWN_VIEW
        elif key == ord("u"):
            UNDIST_VIEW = 1 - UNDIST_VIEW
        # Exit
        elif key == ord("q"):
            break
        elif key == ord("s"):
            np.save(h_file, h)
            np.save(pts1_file, pts1)
            np.save(pts2_file, pts2)
            np.save(pt_ids_file, pt_ids)
            print(f'saved to {pts1_file}\n{pts2_file}\n{h_file}\n{pt_ids_file}')

            yolobox = bbox2yolobbox(bbox)
            yolobox_ = yolobox * np.array([1/W, 1/H, 1/W, 1/H])
            annot_points_ = transform(annot_points, h) * np.array([1/W, 1/H])
            visible = ((annot_points_[:, 0] > 0) * (annot_points_[:, 0] < 1) *
                       (annot_points_[:, 1] > 0) * (annot_points_[:, 1] < 1)).astype(int)
            annot_points_ = annot_points_ * visible[..., None] + np.zeros_like(annot_points_) * (1 - visible[..., None])
            annot_pts__ = np.hstack((annot_points_,visible[:, None]))

            message = ['0'] + [str(x) for x in yolobox_] + [str(x) for kpt in annot_pts__ for x in kpt]
            with open(yolotxt_file, 'w') as f:
                f.write(' '.join(message)+'\n')
            print(f'write annotation file to {yolotxt_file}')
            # show_bbox_kpts_from_yoloannot_txt(camera, yolotxt_file, kpt_shape=annot_pts__.shape)

            # Write Extrinsics
            extri_fname_abs = args.save_camparam_to + '/' + extri_fname
            cameras = {}
            if os.path.exists(extri_fname_abs):
                extri = FileStorage(extri_fname_abs, isWrite=False)
                camnames = extri.read('names', dt='list')
                for key in camnames:
                    cam = {}
                    cam['name'] = key
                    cam['R'] = extri.read('R_{}'.format(key))
                    cam['T'] = extri.read('T_{}'.format(key))
                    cameras[key] = cam
                extri.__del__()
            extri = FileStorage(extri_fname_abs, isWrite=True)
            this_cam = {}
            this_cam['name'] = camera_name
            this_cam['R'] = rvec1
            this_cam['T'] = tvec1
            cameras[camera_name] = this_cam
            extri.write("names", list(cameras.keys()), dt='list')
            for cam in cameras.values():
                extri.write('R_{}'.format(cam['name']), cam['R'])
                extri.write('T_{}'.format(cam['name']), cam['T'])
            extri.__del__()
            print(f'write extri to {extri_fname_abs}')

            # Write Intrinsic
            intri_fname_abs = args.save_camparam_to + '/' + intri_fname
            cameras = {}
            if os.path.exists(intri_fname_abs):
                intri = FileStorage(intri_fname_abs, isWrite=False)
                camnames = intri.read('names', dt='list')
                for key in camnames:
                    cam = {}
                    cam['name'] = key
                    cam['K'] = intri.read('K_{}'.format(key))
                    cam['dist'] = intri.read('dist_{}'.format(key))
                    cameras[key] = cam
                intri.__del__()
            intri = FileStorage(intri_fname_abs, isWrite=True)
            this_cam = {}
            this_cam['name'] = camera_name
            this_cam['K'] = A
            this_cam['dist'] = d.reshape(1, 5)
            cameras[camera_name] = this_cam
            intri.write("names", list(cameras.keys()), dt='list')
            for cam in cameras.values():
                intri.write('K_{}'.format(cam['name']), cam['K'])
                intri.write('dist_{}'.format(cam['name']), cam['dist'])
            intri.__del__()
            print(f'write intri to {intri_fname_abs}')



        overlay = ori_frame.copy() if not UNDIST_VIEW else ori_frame_undist.copy()
        if len(draw_pts) == 0:
            draw_pts = {id: transform(pt, h) for id, pt in predefined_corners.items()}
        # Draw in original image
        if show_points_status == 0:
            # for id, pt in draw_pts.items():
            #     cv2.circle(overlay, tuple(pt), 5, (0, 0, 255), -1)
            if arc_points is not None:
                for pt in arc_points:
                    cv2.circle(overlay, tuple(transform(pt, h)), 5, (0, 0, 255), -1)
        else:
            for pt in annot_points:
                cv2.circle(overlay, tuple(transform(pt, h)), 5, (0, 0, 255), -1)
        for line_id1, line_id2 in line_segs:
            cv2.line(overlay, tuple(draw_pts[line_id1]), tuple(draw_pts[line_id2]), (255, 0, 0), 2)
        x1, y1 = np.max([np.array(list(draw_pts.values())).min(0), np.array([0, 0])], axis=0)
        x2, y2 = np.min([np.array(list(draw_pts.values())).max(0), np.array([W-1, H-1])], axis=0)
        bbox = np.array([x1,y1,x2,y2])
        # cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if 'image_points' not in locals() and 'object_points' not in locals():
            # outer_points = np.array((predefined_corners[0], predefined_corners[1], predefined_corners[2], predefined_corners[3]))
            # image_points = transform(outer_points, h).astype(np.float32)
            # object_points = np.concatenate((outer_points, np.zeros((outer_points.shape[0], 1))), axis=1).astype(np.float32)
            image_points = transform(annot_points, h).astype(np.float32)
            object_points = np.concatenate((annot_points, np.zeros((annot_points.shape[0], 1))), axis=1).astype(np.float32)[:, [1, 0, 2]]
            valid_id = (0 < image_points[:, 0]) * (image_points[:, 0] < W) * (0 < image_points[:, 1]) * (image_points[:, 1] < H)
            image_points = image_points[valid_id]
            object_points = object_points[valid_id]
            bev_ref_points = annot_points * 100 + np.array((145., 195.))
            # image_points = pts1.astype(np.float32)
            # object_points = np.concatenate((pts2, np.zeros((pts2.shape[0], 1))), axis=1).astype(np.float32)
            # image_points = np.vstack(list(draw_pts.values()))
            # object_points = np.concatenate((np.vstack(list(predefined_corners.values())), np.zeros((len(predefined_corners), 1))), axis=1)[:, [1, 0, 2]]
            # bev_ref_points = np.vstack(list(predefined_corners.values())) * 100 + np.array((145., 195.))
            print(len(image_points))

        else:
            assert 'image_points' in locals() and 'object_points' in locals()
            image_points = image_points
            object_points = object_points
        image_points_final = map_points_to_original(image_points, map1, map2) if args.undistort else image_points
        if args.undistort:
            for pt in image_points:
                cv2.circle(frame, tuple(pt.astype(int)), 5, (0, 255, 0), -1) if UNDIST_VIEW else None
            for pt in image_points_final:
                cv2.circle(frame, tuple(pt.astype(int)), 5, (0, 255, 0), -1) if not UNDIST_VIEW else None
        A, d, rvec1, tvec1 = compute_RTd_from_fixedK(image_points_final, object_points, A, d, method=METHOD)
        METHOD = 1 if A is not None and METHOD == 0 else METHOD
        if 'rvec1' in locals() and 'tvec1' in locals():
            colors = [(0, 0, 255), (0, 255, 0), (0, 255, 255)]
            if not args.undistort:
                for color_idx, predefined_cube in enumerate(predefined_cubes):
                    draw_cube_on_img(predefined_cube[:, [1, 0, 2]], A, d, rvec1, tvec1, overlay, color=colors[color_idx])
            else:
                for color_idx, predefined_cube in enumerate(predefined_cubes):
                    draw_distorted_cube_on_img(predefined_cube[:, [1, 0, 2]], A, d, rvec1, tvec1, overlay, map1, map2, newcameramtx, oldcameramtx, old_d, color=colors[color_idx])
            draw_axis_on_img(A, d, rvec1, tvec1, overlay)
        alpha = 0.9
        frame = cv2.addWeighted(overlay, alpha, ori_frame, 1 - alpha, 0) if not UNDIST_VIEW else \
                cv2.addWeighted(overlay, alpha, ori_frame_undist, 1 - alpha, 0)
        # Compute the point idx which has the biggest reproj error in the bird's eye view
        if CONVERGE:
            assert 'image_points' in locals() and 'object_points' in locals()
            if 'h_img2court' not in locals():
                h_img2court = H_meter_to_centimeter_and_base_shift @ np.linalg.inv(h)
            bev_projected_pts = cv2.perspectiveTransform(
                cv2.projectPoints(object_points, rvec1, tvec1, A, d)[0].reshape((-1, 1, 2)), h_img2court).reshape(
                (-1, 2))
            distances = np.linalg.norm(bev_projected_pts - bev_ref_points, axis=1)
            add_id = np.argmax(distances)
            image_points = np.vstack((image_points, image_points[add_id]))
            object_points = np.vstack((object_points, object_points[add_id]))
            bev_ref_points = np.vstack((bev_ref_points, bev_ref_points[add_id]))
        if USE_Z_POINT:
            clicked_coords = {'x': None, 'y': None} if 'clicked_coords' not in locals() else clicked_coords
            def get_coordinates(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    print("Coordinates:", x, y)
                    param['x'] = x
                    param['y'] = y
            cv2.setMouseCallback(wname, get_coordinates, param=clicked_coords)
            if clicked_coords['x'] is not None:
                pts1_z_points[USE_Z_POINT - 1] = np.array([clicked_coords['x'], clicked_coords['y']])
                clicked_coords = {'x': None, 'y': None}

            for pt in pts1_z_points.values():
                cv2.circle(frame, tuple(pt), 4, (0, 0, 255), 2)
            pts1_planar = transform(annot_points, h).astype(np.float32)
            pts2_planar_3d = np.concatenate((annot_points, np.zeros((annot_points.shape[0], 1))), axis=1).astype(np.float32)[:, [1, 0, 2]]
            if len(pts1_z_points) > 0:
                pts1_z = np.array([pt for i, pt in pts1_z_points.items()])
                pts2_z_3d = np.array([predefined_z_points[i] for i, pt in pts1_z_points.items()]).astype(np.float32)[:, [1, 0, 2]]
                pts1_all = np.concatenate((pts1_planar, pts1_z))
                pts1_all = map_points_to_original(pts1_all, map1, map2) if args.undistort else pts1_all
                pts2_all_3d = np.concatenate((pts2_planar_3d, pts2_z_3d))
                x0 = np.concatenate((np.array((A[0, 0], A[1, 1], A[0, 2], A[1, 2])), d.flatten(), rvec1.flatten(), tvec1.flatten()))
                res = minimize(reproj_error, x0, args=(pts2_all_3d, pts1_all), method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
                x0 = res.x
                fx, fy, cx, cy = x0[:4]
                A = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                d = x0[4:9].reshape(5,)
                rvec1 = x0[9:12].reshape(3, 1)
                tvec1 = x0[12:15].reshape(3, 1)
                print(f'{fx=}, {fy=}, {cx=}, {cy=}, {d=}')


        draw_axis_on_img(K=A, d=d, rvec=rvec1, tvec=tvec1, img=frame)
        if not TOPDOWN_VIEW:
            cv2.imshow(wname, frame)
        else:
            h_img2court = H_meter_to_centimeter_and_base_shift @ np.linalg.inv(h)
            frame_bev = cv2.warpPerspective(frame, h_img2court,
                                            (courtTemplate.court.shape[1], courtTemplate.court.shape[0]))
            courtTemplate.draw_court_on_warped_image(frame_bev)
            cv2.imshow(wname, frame_bev)
        cv2.imshow(wname_p, floor)
        # print(f'{A=}, {d=}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=application_path+'/game_images',
                        help="The input directory")
    parser.add_argument("--selected_imgs", type=str, default=None,
                        help="regular expression for selected images in the input directory."
                             "For example: Stop_Hungary*--aligned--*--*.jpg")
    parser.add_argument("--save_annot_to", type=str, default=application_path+'/save_to')
    parser.add_argument("--save_camparam_to", type=str, default=application_path+'/game_camparam')
    parser.add_argument("--K", type=float, nargs=9, default=None, help="intrinsic matrix")
    parser.add_argument("--d", type=float, nargs=5, default=None, help="distortion coefficients")
    parser.add_argument("--init_extri_from_preceding_img", action="store_true",
                        help='labeling assistance: inherit marked points from last image')
    parser.add_argument("--undistort", type=bool, default=False,
                        help='only enabled when the distortion is very apparent')
    parser.add_argument("--read_intri_from_file", type=str, default=False,
                        help='Read intrisics from file, please make sure the file already exist.'
                             'Priority is highest.')
    image_extensions = ['jpg', 'jpeg', 'png']


    args = parser.parse_args()
    args.K = np.array(args.K).reshape(3, 3) if args.K is not None else None
    args.d = np.array(args.d).reshape(5, ) if args.d is not None else None
    args.PRE_img_base = None
    PRE_img_base = None
    if args.selected_imgs is not None:
        print(f'{args.selected_imgs=}')
        tmp = args.input_dir + '/' + args.selected_imgs
        print(f'search {tmp}')
        img_paths = glob.glob(args.input_dir + '/' + args.selected_imgs)
        img_paths = [img_paths[i] for i in range(len(img_paths))
                     # if int(os.path.basename(img_paths[i]).rstrip('.jpg').rstrip('.png')) % 6 == 0
                     # and int(os.path.basename(img_paths[i]).rstrip('.jpg').rstrip('.png')) > 612
                     ]
        print(f'len(img_paths)={len(img_paths)}')
    else:
        img_paths = []
        for ext in image_extensions:
            img_paths.extend(glob.glob(args.input_dir + '/*.' + ext))

    for i, img_path in enumerate(tqdm.tqdm(sorted(img_paths))):
        args.img_base = os.path.basename(img_path)
        if i > 0:
            args.PRE_img_base = PRE_img_base
        root_ = tk.Tk()
        root_.title('main')
        print(args.img_base)
        label = tk.Label(root_, text=f"{args.img_base}流程\n"
                                     "1.选择4个角点的id并标记图中位置\n"
                                     "(若已有保存值，会自动跳过此步,直接显示蓝色球场线)\n"
                                     "2.微调4个对应点的位置：Enter键\n"
                                     "3.保存/覆盖存档：S键\n"
                                     "4.退出当前image：Q键"
                         , font=("Helvetica", 14))
        label.pack()
        main(args)
        root_.quit()
        root_.destroy()
        PRE_img_base = args.img_base
