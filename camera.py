"""Define camera object and multiview camera here"""
import os
from typing import List, Dict, Union, Optional

import numpy as np
import cv2
import yaml
import json


def load_camera_from_yaml(
    intri_path: str, extri_path: str, camera_ids: List[str] = None
) -> Dict:
    """
    load intrinsics and extrinsics from yaml file, the intri yaml file should be like this:
        %YAML:1.0
        ---
        K_01: !!opencv-matrix
        rows: 3
        cols: 3
        dt: d
        data: [2714.137788, 0.000000, 911.738749, 0.000000, 2683.085000, 588.723669, 0.000000, 0.000000, 1.000000]
        dist_01: !!opencv-matrix
        rows: 1
        cols: 5
        dt: d
        data: [0.000000, 0.000000, 0.000000, -0.000000, 0.000000]
        names:
        - "01"

    if given camera_id, the function will only load the given camera's intrinsics and extrinsics, otherwise, it will load all the cameras' intrinsics and extrinsics

    Args:
        intri_path: path to the intrinsics yaml file
        extri_path: path to the extrinsics yaml file
        camera_id: camera id
    """
    fs_intri = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_READ)
    fs_extri = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)

    intri_data = {}
    extri_data = {}

    # 读取内参数据
    for name in fs_intri.root().keys():
        if name == "names":
            intri_data[name] = []
            for i in range(fs_intri.getNode(name).size()):
                intri_data[name].append(fs_intri.getNode(name).at(i).string())
            continue
        intri_data[name] = fs_intri.getNode(name).mat()

    # 读取外参数据
    for name in fs_extri.root().keys():
        if name == "names":
            extri_data[name] = []
            for i in range(fs_extri.getNode(name).size()):
                extri_data[name].append(fs_extri.getNode(name).at(i).string())
            continue
        extri_data[name] = fs_extri.getNode(name).mat()

    fs_intri.release()
    fs_extri.release()

    if "names" not in intri_data or "names" not in extri_data:
        raise ValueError("Invalid intrinsics or extrinsics yaml file")

    # check exist camera_id, should be in both intri_data and extri_data
    intri_camera_ids = intri_data["names"]
    extri_camera_ids = extri_data["names"]
    file_camera_ids = list(set(intri_camera_ids) & set(extri_camera_ids))

    if camera_ids is None:
        camera_ids = file_camera_ids
    else:
        # check if all camera_ids are in file_camera_ids
        if not all(camera_id in file_camera_ids for camera_id in camera_ids):
            raise ValueError(f"Invalid camera id: {camera_ids}")

    camera_params = {}
    # for each camera_id, load the intrinsics and extrinsics
    for camera_id in camera_ids:
        param = {}
        param["K"] = intri_data[f"K_{camera_id}"]
        param["dist"] = intri_data[f"dist_{camera_id}"]
        param["Rvec"] = extri_data[f"R_{camera_id}"]
        param["T"] = extri_data[f"T_{camera_id}"]
        camera_params[camera_id] = param

    return camera_params


def load_camera_from_json(json_path: Union[str, List[Dict]]) -> Dict:
    """
    Load camera parameters from json file
    the json file should be like this:
    [
        {
            "camera_id": "01",
            "K": [2714.137788, 0.000000, 911.738749, 0.000000, 2683.085000, 588.723669, 0.000000, 0.000000, 1.000000],
            "dist": [0.000000, 0.000000, 0.000000, -0.000000, 0.000000],
            "R": [1.000000, 0.000000, 0.000000],
            "T": [0.000000, 0.000000, 0.000000]
        }
    ]
    """
    if isinstance(json_path, str):
        with open(json_path, "r") as f:
            raw_camera_params = json.load(f)
    elif isinstance(json_path, list):
        raw_camera_params = json_path
    else:
        raise ValueError(f"Invalid json path: {json_path}")

    camera_ids = []
    camera_params = {}
    for param in raw_camera_params:
        camera_id = param["camera_id"]
        camera_ids.append(camera_id)
        camera_params[camera_id] = {
            "K": param["K"],
            "dist": param["dist"],
            "Rvec": param["R"],
            "T": param["T"],
        }

    return camera_ids, camera_params


def compute_C_up_forward(R, T):
    C = -R.T @ T
    C = C[:, 0]
    C = C[[1, 2, 0]]
    C[2] = -C[2]
    R = R.T
    right, up, forward = R[:, 0], R[:, 1], R[:, 2]
    right = right[[1, 2, 0]]
    right[2] = -right[2]
    up = up[[1, 2, 0]]
    up[2] = -up[2]
    forward = forward[[1, 2, 0]]
    forward[2] = -forward[2]
    return C, right, up, forward

def compute_sensor_size_fromK(K, f_mm=18, width=1920, height=1080):
    assert K.shape == (3, 3)
    ax = K[0, 0]
    ay = K[1, 1]
    x0 = K[0, 2]
    y0 = K[1, 2]
    f = f_mm
    sizeX = f * width / ax
    sizeY = f * height / ay
    shiftX = -(x0-width/2) / width
    shiftY = (y0-height/2) / height

    sensorSize = (sizeX, sizeY)
    focalLength = f
    lensShift = (shiftX, shiftY)
    return focalLength, sensorSize, lensShift


class Camera:
    """
    Camera Object, support loading from yaml file or directly from parameters, if loading from yaml file, the camera_id must be provided
    Args:
        camera_id: camera id
        K: intrinsic matrix
        dist: distortion coefficients
        Rvec: rotation vector
        T: translation matrix
        intri_path: path to the intrinsics yaml file
        extri_path: path to the extrinsics yaml file
    """

    def __init__(
        self,
        camera_id: str = "",
        K: Union[List, np.ndarray] = None,
        dist: Union[List, np.ndarray] = None,
        Rvec: Union[List, np.ndarray] = None,
        T: Union[List, np.ndarray] = None,
        intri_path: str = None,
        extri_path: str = None,
    ):
        self.camera_id = camera_id

        if intri_path and extri_path:
            assert (
                camera_id is not None and camera_id != ""
            ), "camera_id must be provided if intri_path and extri_path are provided"
            self.intri_path = intri_path
            self.extri_path = extri_path
            self.load_from_yaml(self.intri_path, self.extri_path)
        else:
            self.load_parameters(K, dist, Rvec, T)

    def load_parameters(
        self,
        K: Union[List, np.ndarray] = None,
        dist: Union[List, np.ndarray] = None,
        Rvec: Union[List, np.ndarray] = None,
        T: Union[List, np.ndarray] = None,
    ):
        # convert List to np.ndarray
        if isinstance(K, list):
            K = np.array(K)
        self.K = K.reshape((3, 3)).astype(np.float32) if K is not None else None

        if isinstance(dist, list):
            dist = np.array(dist)
        self.dist = (
            dist.reshape((1, 5)).astype(np.float32) if dist is not None else None
        )

        if isinstance(Rvec, list):
            Rvec = np.array(Rvec).reshape((3, 1))
        R = cv2.Rodrigues(Rvec)[0] if Rvec is not None else None

        self.R = R.reshape((3, 3)).astype(np.float32) if R is not None else None
        self.Rvec = (
            Rvec.reshape((3, 1)).astype(np.float32) if Rvec is not None else None
        )

        if isinstance(T, list):
            T = np.array(T)
        self.T = T.reshape((3, 1)).astype(np.float32) if T is not None else None

        if K is not None:
            self._invK = np.linalg.inv(self.K)

        if self.R is not None and self.T is not None:
            self._RT = np.hstack([self.R, self.T])

        if self.K is not None and self.R is not None and self.T is not None:
            self._P = self.K @ self.RT

        if self.RT is not None and self.invK is not None:
            self._ground_inv_P = np.linalg.inv(self.RT[:, [0, 1, 3]]) @ self.invK

    @property
    def invK(self):
        return getattr(self, "_invK", None)

    @property
    def RT(self):
        return getattr(self, "_RT", None)

    @property
    def P(self):
        return getattr(self, "_P", None)

    @property
    def ground_inv_P(self):
        return getattr(self, "_ground_inv_P", None)

    def project_points(self, points_3d: Union[List, np.ndarray], only_valid=False) -> np.ndarray:
        if self.P is None:
            return None
        # points_3d: (N, 3), dtype=np.float32
        points_3d = np.array(points_3d, dtype=np.float32)
        # shape = points_3d.shape
        if only_valid:
            invalid_index = points_3d[:,0] == 0
        # check the shape of points_3d
        if points_3d.ndim == 1:
            points_3d = points_3d[np.newaxis, :]
        shape = points_3d.shape
        if points_3d.shape[-1] > 3:
            ext_score = points_3d[:,3:]
        if points_3d.shape[-1] < 3:
            raise ValueError(f"Invalid shape of points_3d: {points_3d.shape}")
        else:
            points_3d = points_3d[..., :3]
        points_2d, _ = cv2.projectPoints(
            points_3d.astype('float64'), self.Rvec, self.T, self.K, self.dist
        )
        points_2d = points_2d.reshape(*shape[:-1], 2)
        if shape[-1] > 3:
            points_2d = np.concatenate([points_2d, ext_score], axis=-1)
        if only_valid:
            points_2d[invalid_index,:] *= 0
        return points_2d

    def undistort_points(self, points_2d: Union[List, np.ndarray]) -> np.ndarray:
        if self.K is None or self.dist is None:
            return None
        return cv2.undistortPoints(points_2d.astype(float), self.K, self.dist, P=self.K)

    def unproject2ground(self, points_2d: Union[List, np.ndarray]) -> np.ndarray:
        """unproject points from image coordinate to world coordinate on ground plane (z=0)
        Args:
            points_2d: np.ndarray, shape (N, 2)
        Returns:
            np.ndarray, shape (N, 2)
        """
        undistorted_points_2d = self.undistort_points(points_2d)
        homo_points_2d = cv2.convertPointsToHomogeneous(undistorted_points_2d)[
            :, 0, :, None
        ]
        points_on_ground = self.ground_inv_P @ homo_points_2d
        points_on_ground = points_on_ground[..., 0]
        points_on_ground = (
            points_on_ground[..., :2] / points_on_ground[..., 2, np.newaxis]
        )
        return points_on_ground

    def load_from_yaml(self, intri_path: str, extri_path: str):
        camera_params = load_camera_from_yaml(intri_path, extri_path, self.camera_id)
        self.load_parameters(**camera_params[self.camera_id])


    def convert_to_unity(self, width=1920, height=1080):
        focal_length, sensor_size, lens_shift = compute_sensor_size_fromK(self.K, width=width, height=height)
        C, right, up, forward = compute_C_up_forward(self.R, self.T)
        return {
            "focalLength": focal_length,
            "sensorSize": sensor_size,
            "lensShift": lens_shift,
            "C": C.tolist(),
            "right": right.tolist(),
            "up": up.tolist(),
            "forward": forward.tolist(),
        }

class MultiViewCamera:
    """
    MultiViewCamera Object, is a collection of Camera objects, used to represent a multi-view camera system
    """

    def __init__(
        self, camera_ids: List[str], camera_params: Dict[str, Union[Dict, Camera]]
    ):
        self.camera_ids: List[str] = camera_ids
        self.camera_params = camera_params
        self.cameras: Dict[str, Camera] = {
            camera_id: (
                Camera(camera_id, **camera_params[camera_id])
                if isinstance(camera_params[camera_id], dict)
                else camera_params[camera_id]
            )
            for camera_id in camera_ids
        }

    def __getitem__(self, camera_id: str):
        return self.cameras[camera_id]

    @property
    def P_all(self) -> np.ndarray:
        # concat all the P matrix of the cameras
        return np.stack(
            [self.cameras[camera_id].P for camera_id in self.camera_ids], axis=0
        )

    def save_to_json(self, json_path: str):
        camera_params = []
        for camera_id in self.camera_ids:
            param = {
                "camera_id": camera_id,
                "K": self.cameras[camera_id].K.reshape(-1).tolist(),
                "dist": self.cameras[camera_id].dist.reshape(-1).tolist(),
                "R": self.cameras[camera_id].Rvec.reshape(-1).tolist(),
                "T": self.cameras[camera_id].T.reshape(-1).tolist(),
            }
            camera_params.append(param)
        with open(json_path, "w") as f:
            json.dump(camera_params, f, indent=4)

    def save_to_unity(self, unity_json_path:str, id_key):
        unity_camers = {}
        for camera_id in self.camera_ids:
            unity_camers[id_key[camera_id]] = self.cameras[camera_id].convert_to_unity()
        os.makedirs(os.path.dirname(unity_json_path), exist_ok=True)
        with open(unity_json_path, "w") as f:
            json.dump(unity_camers, f, indent=4)


def build_multi_view_camera(
    intri_path: str, extri_path: str, camera_ids: List[str],json_camera_path:str=None
) -> MultiViewCamera:
    if json_camera_path:
        file_camera_ids, camera_params = load_camera_from_json(json_camera_path)
        # check if all camera_ids are in camera_params
        assert all([cid in file_camera_ids for cid in camera_ids]), f"Invalid camera_ids, {json_camera_path} not contain all camera"
    else:
        camera_params = load_camera_from_yaml(intri_path, extri_path, camera_ids)
    return MultiViewCamera(camera_ids, camera_params)


def build_multi_view_camera_from_path(
        camera_path: str, camera_ids: List[str]
):
    intri_path = os.path.join(camera_path, "intri.yml")
    extri_path = os.path.join(camera_path, "extri.yml")
    if os.path.exists(intri_path) and os.path.exists(extri_path):
        return build_multi_view_camera(intri_path, extri_path, camera_ids)
    json_path = os.path.join(camera_path, "camera_params.json")
    if os.path.exists(json_path):
        return build_multi_view_camera(None, None, camera_ids, json_path)
    raise ValueError(f"Invalid camera path: {camera_path}")


def build_multi_view_camera_from_json(
    json_path: Union[str, List[Dict]]
) -> MultiViewCamera:
    camera_ids, camera_params = load_camera_from_json(json_path)
    return MultiViewCamera(camera_ids, camera_params)


if __name__ == "__main__":
    camera_ids = ["0"]
    # camera_params = {
    #     "01": {
    #         "K": np.array(
    #             [
    #                 [2714.137788, 0.000000, 911.738749],
    #                 [0.000000, 2683.085000, 588.723669],
    #                 [0.000000, 0.000000, 1.000000],
    #             ]
    #         ),
    #         "dist": np.array([0.000000, 0.000000, 0.000000, -0.000000, 0.000000]),
    #         "Rvec": np.array(
    #             [
    #                 [
    #                     1,
    #                 ],
    #                 [
    #                     0,
    #                 ],
    #                 [
    #                     0,
    #                 ],
    #             ],
    #             dtype=np.float32,
    #         ),
    #         "T": np.array([0.000000, 0.000000, 0.000000]),
    #     },
    #     "02": {
    #         "K": np.array(
    #             [
    #                 [2714.137788, 0.000000, 911.738749],
    #                 [0.000000, 2683.085000, 588.723669],
    #                 [0.000000, 0.000000, 1.000000],
    #             ]
    #         ),
    #         "dist": np.array([0.000000, 0.000000, 0.000000, -0.000000, 0.000000]),
    #         "Rvec": np.array(
    #             [
    #                 [
    #                     1,
    #                 ],
    #                 [
    #                     0,
    #                 ],
    #                 [
    #                     0,
    #                 ],
    #             ],
    #             dtype=np.float32,
    #         ),
    #         "T": np.array([0.000000, 0.000000, 0.000000]),
    #     },
    #     "03": {
    #         "K": np.array(
    #             [
    #                 [2714.137788, 0.000000, 911.738749],
    #                 [0.000000, 2683.085000, 588.723669],
    #                 [0.000000, 0.000000, 1.000000],
    #             ]
    #         ),
    #         "dist": np.array([0.000000, 0.000000, 0.000000, -0.000000, 0.000000]),
    #         "Rvec": np.array(
    #             [
    #                 [
    #                     1,
    #                 ],
    #                 [
    #                     0,
    #                 ],
    #                 [
    #                     0,
    #                 ],
    #             ],
    #             dtype=np.float32,
    #         ),
    #         "T": np.array([0.000000, 0.000000, 0.000000]),
    #     },
    # }
    # multi_view_camera = MultiViewCamera(camera_ids, camera_params)
    # print(multi_view_camera.P_all)

    # test load_from_yaml
    intri_path = "/Users/yifei/Desktop/Research/DynamicView/z-/game_camparam/intri_01278.yml"
    extri_path = "/Users/yifei/Desktop/Research/DynamicView/z-/game_camparam/extri_01278.yml"
    # camera_params = load_camera_from_yaml(intri_path, extri_path)
    # multi_view_camera = MultiViewCamera(camera_ids, camera_params)
    multi_view_camera = build_multi_view_camera(intri_path, extri_path, camera_ids)
    save_path = "data/demo_videos/0903_clip_02/unity_camera_params.json"
    multi_view_camera.save_to_unity(save_path)
    # print(multi_view_camera.P_all)
